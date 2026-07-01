// Stage 0 correctness test for the SYCL CvtColor port (Family A: channel
// swap / alpha add-drop, codes 0-5). Exact match vs CPU gold (tolerance 0 —
// Family A is pure channel reorder). Negative tests: 16F alpha-add rejected,
// unsupported code rejected, wrong channels rejected.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   
//   test/test_cvt.cpp \
//   -o test/test_cvt
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_cvt

#include "../colorcvt.hpp"
#include "cvt_gold.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

using namespace sycl;
using cvcuda::cvt::CvtColor;
using cvcuda::cvt::DType;

// code -> (sch, dch, alpha_only)
static void code_dims(int code, int& sch, int& dch, bool& alpha_only) {
  sch = (code == 1 || code == 3 || code == 5) ? 4 : 3;
  dch = (code == 0 || code == 2 || code == 5) ? 4 : 3;
  alpha_only = (code == 0 || code == 1);  // bidx==0
}

template <typename T>
static bool run_case(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int sch, dch;
  bool alpha_only;
  code_dims(code, sch, dch, alpha_only);

  DType dt;
  if constexpr (std::is_same_v<T, uint8_t>) dt = DType::U8;
  else if constexpr (std::is_same_v<T, uint16_t>) dt = DType::U16;
  else if constexpr (std::is_same_v<T, int32_t>) dt = DType::S32;
  else if constexpr (std::is_same_v<T, float>) dt = DType::F32;
  else if constexpr (std::is_same_v<T, double>) dt = DType::F64;

  size_t nin = (size_t)N * H * W * sch, nout = (size_t)N * H * W * dch;
  std::vector<T> src(nin), gpu(nout), cpu(nout);

  std::mt19937 rng(12345 + code);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::cvt::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    for (auto& v : src) v = (T)d(rng);
  }

  T* d_in = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nout, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();

  bool ok = op.forward(d_in, d_out, N, H, W, sch, dch, dt, code, &q);
  if (!ok) {
    std::cout << "  [code " << code << "] forward returned false (unexpected)\n";
    free(d_in, q); free(d_out, q);
    return false;
  }
  q.wait();
  q.memcpy(gpu.data(), d_out, nout * sizeof(T)).wait();

  cvt_gold::gold_family_a<T>(cpu.data(), src.data(), (size_t)N * H * W, sch, dch, alpha_only);

  int mismatch = 0;
  for (size_t i = 0; i < nout; ++i)
    if (gpu[i] != cpu[i]) ++mismatch;
  std::cout << "  [code " << code << " " << (alpha_only ? "alpha" : "swap") << " sch=" << sch
            << " dch=" << dch << "] mismatch=" << mismatch << (mismatch ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return mismatch == 0;
}

// tolerance for |gpu-cpu| (double; 0 = exact). Returns mismatch count.
template <typename T>
static int count_mismatch_tol(const std::vector<T>& gpu, const std::vector<T>& cpu, double tol) {
  int m = 0;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs((double)gpu[i] - (double)cpu[i]) > tol) ++m;
  return m;
}

// Family B: BGR->GRAY (codes 6,7,10,11). dtypes U8/U16/F32; tol U8=1, U16=2, F32=ERR1_4.
template <typename T>
static bool run_case_gray(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int bidx = (code == 7 || code == 11) ? 2 : 0;
  int sch  = (code == 10 || code == 11) ? 4 : 3;
  DType dt;
  double tol;
  if constexpr (std::is_same_v<T, uint8_t>) { dt = DType::U8; tol = 1.0; }
  else if constexpr (std::is_same_v<T, uint16_t>) { dt = DType::U16; tol = 2.0; }
  else { dt = DType::F32; tol = 1.22e-4; }

  size_t nin = (size_t)N * H * W * sch, nout = (size_t)N * H * W;
  std::vector<T> src(nin), gpu(nout), cpu(nout);
  std::mt19937 rng(999 + code);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::cvt::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    for (auto& v : src) v = (T)d(rng);
  }
  T* d_in = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nout, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();
  bool ok = op.forward(d_in, d_out, N, H, W, sch, 1, dt, code, &q);
  if (!ok) { std::cout << "  [gray code " << code << "] forward=false (unexpected)\n"; free(d_in,q); free(d_out,q); return false; }
  q.wait();
  q.memcpy(gpu.data(), d_out, nout * sizeof(T)).wait();
  cvt_gold::gold_bgr_to_gray<T>(cpu.data(), src.data(), (size_t)N * H * W, sch, bidx);
  int m = count_mismatch_tol<T>(gpu, cpu, tol);
  std::cout << "  [gray code " << code << " sch=" << sch << "] mismatch=" << m << " (tol=" << tol << ")"
            << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

// Family B: GRAY->BGR/BGRA (codes 8,9). Pure replicate -> exact (tol 0).
template <typename T>
static bool run_case_gray2bgr(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int dch = (code == 9) ? 4 : 3;
  DType dt;
  if constexpr (std::is_same_v<T, uint8_t>) dt = DType::U8;
  else if constexpr (std::is_same_v<T, uint16_t>) dt = DType::U16;
  else if constexpr (std::is_same_v<T, float>) dt = DType::F32;
  else if constexpr (std::is_same_v<T, double>) dt = DType::F64;

  size_t nin = (size_t)N * H * W, nout = (size_t)N * H * W * dch;
  std::vector<T> src(nin), gpu(nout), cpu(nout);
  std::mt19937 rng(777 + code);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::cvt::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    for (auto& v : src) v = (T)d(rng);
  }
  T* d_in = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nout, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();
  bool ok = op.forward(d_in, d_out, N, H, W, 1, dch, dt, code, &q);
  if (!ok) { std::cout << "  [g2b code " << code << "] forward=false (unexpected)\n"; free(d_in,q); free(d_out,q); return false; }
  q.wait();
  q.memcpy(gpu.data(), d_out, nout * sizeof(T)).wait();
  cvt_gold::gold_gray_to_bgr<T>(cpu.data(), src.data(), (size_t)N * H * W, dch);
  int m = count_mismatch_tol<T>(gpu, cpu, 0.0);
  std::cout << "  [g2b  code " << code << " dch=" << dch << "] mismatch=" << m << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

// Family C: packed YUV (codes 82 BGR2YUV, 83 RGB2YUV, 84 YUV2BGR, 85 YUV2RGB).
// dtypes U8/U16/F32. Tolerances: U8=1; U16 RGB2YUV=2, YUV2BGR=1; F32=ERR1_4.
template <typename T>
static bool run_case_yuv(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int bidx = (code == 82 || code == 84) ? 0 : 2;
  DType dt;
  double tol;
  if constexpr (std::is_same_v<T, uint8_t>) { dt = DType::U8; tol = 1.0; }
  else if constexpr (std::is_same_v<T, uint16_t>) { dt = DType::U16; tol = (code <= 83) ? 2.0 : 1.0; }
  else { dt = DType::F32; tol = 1.22e-4; }

  size_t n = (size_t)N * H * W * 3;
  std::vector<T> src(n), gpu(n), cpu(n);
  std::mt19937 rng(555 + code);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::cvt::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    for (auto& v : src) v = (T)d(rng);
  }
  T* d_in = malloc_device<T>(n, q);
  T* d_out = malloc_device<T>(n, q);
  q.memcpy(d_in, src.data(), n * sizeof(T)).wait();
  bool ok = op.forward(d_in, d_out, N, H, W, 3, 3, dt, code, &q);
  if (!ok) { std::cout << "  [yuv code " << code << "] forward=false (unexpected)\n"; free(d_in,q); free(d_out,q); return false; }
  q.wait();
  q.memcpy(gpu.data(), d_out, n * sizeof(T)).wait();
  if (code <= 83)
    cvt_gold::gold_bgr_to_yuv<T>(cpu.data(), src.data(), (size_t)N * H * W, bidx);
  else
    cvt_gold::gold_yuv_to_bgr<T>(cpu.data(), src.data(), (size_t)N * H * W, bidx);
  int m = count_mismatch_tol<T>(gpu, cpu, tol);
  std::cout << "  [yuv  code " << code << "] mismatch=" << m << " (tol=" << tol << ")"
            << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

// circular hue mismatch count for HSV output (H mod range, S/V linear).
template <typename T>
static int count_mismatch_hsv(const std::vector<T>& gpu, const std::vector<T>& cpu, size_t npx,
                              double range, double tol) {
  int m = 0;
  for (size_t i = 0; i < npx; ++i) {
    size_t b = i * 3;
    double dh = std::fabs((double)gpu[b] - (double)cpu[b]);
    if (dh > range / 2.0) dh = range - dh;
    double ds = std::fabs((double)gpu[b + 1] - (double)cpu[b + 1]);
    double dv = std::fabs((double)gpu[b + 2] - (double)cpu[b + 2]);
    if (dh > tol || ds > tol || dv > tol) ++m;
  }
  return m;
}

// Family D: BGR/RGB -> HSV (codes 40,41,66,67). U8 tol 1, F32 tol ERR2_3.
// Hue compared circularly mod range.
template <typename T>
static bool run_case_bgr2hsv(CvtColor& op, queue& q, int code, int N, int H, int W) {
  bool isFullRange = (code == 66 || code == 67);
  int bidx = (code == 40 || code == 66) ? 0 : 2;
  DType dt;
  double tol, range;
  if constexpr (std::is_same_v<T, uint8_t>) {
    dt = DType::U8; tol = 1.0; range = isFullRange ? 256.0 : 180.0;
  } else {
    dt = DType::F32; tol = 1.95e-3; range = 360.0;
  }
  size_t n = (size_t)N * H * W * 3;
  std::vector<T> src(n), gpu(n), cpu(n);
  std::mt19937 rng(321 + code);
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::cvt::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    for (auto& v : src) v = (T)d(rng);
  }
  T* d_in = malloc_device<T>(n, q);
  T* d_out = malloc_device<T>(n, q);
  q.memcpy(d_in, src.data(), n * sizeof(T)).wait();
  bool ok = op.forward(d_in, d_out, N, H, W, 3, 3, dt, code, &q);
  if (!ok) { std::cout << "  [b2h code " << code << "] forward=false (unexpected)\n"; free(d_in,q); free(d_out,q); return false; }
  q.wait();
  q.memcpy(gpu.data(), d_out, n * sizeof(T)).wait();
  if (isFullRange)
    cvt_gold::gold_bgr_to_hsv<T, true>(cpu.data(), src.data(), (size_t)N * H * W, bidx);
  else
    cvt_gold::gold_bgr_to_hsv<T, false>(cpu.data(), src.data(), (size_t)N * H * W, bidx);
  int m = count_mismatch_hsv<T>(gpu, cpu, (size_t)N * H * W, range, tol);
  std::cout << "  [b2h  code " << code << "] mismatch=" << m << " (tol=" << tol << ",circ range=" << range << ")"
            << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

// Family D: HSV -> BGR/RGB (codes 54,55,70,71). Cout=3. U8 tol 1, F32 tol ERR1_4.
// Output is RGB (non-circular, linear compare).
template <typename T>
static bool run_case_hsv2bgr(CvtColor& op, queue& q, int code, int N, int H, int W) {
  bool isFullRange = (code == 70 || code == 71);
  int bidx = (code == 54 || code == 70) ? 0 : 2;
  DType dt;
  double tol, range;
  if constexpr (std::is_same_v<T, uint8_t>) {
    dt = DType::U8; tol = 1.0; range = isFullRange ? 256.0 : 180.0;
  } else {
    dt = DType::F32; tol = 1.22e-4; range = 360.0;
  }
  size_t n = (size_t)N * H * W * 3;
  std::vector<T> src(n), gpu(n), cpu(n);
  std::mt19937 rng(432 + code);
  if constexpr (std::is_integral_v<T>) {
    int hmax = (int)range;
    std::uniform_int_distribution<int> dh(0, hmax - 1);
    std::uniform_int_distribution<int> dsv(0, (int)cvcuda::cvt::TypeMax<T>());
    for (size_t i = 0; i < (size_t)N * H * W; ++i) {
      src[i * 3 + 0] = (T)dh(rng);
      src[i * 3 + 1] = (T)dsv(rng);
      src[i * 3 + 2] = (T)dsv(rng);
    }
  } else {
    std::uniform_real_distribution<double> dh(0.0, range);
    std::uniform_real_distribution<double> dsv(0.0, 1.0);
    for (size_t i = 0; i < (size_t)N * H * W; ++i) {
      src[i * 3 + 0] = (T)dh(rng);
      src[i * 3 + 1] = (T)dsv(rng);
      src[i * 3 + 2] = (T)dsv(rng);
    }
  }
  T* d_in = malloc_device<T>(n, q);
  T* d_out = malloc_device<T>(n, q);
  q.memcpy(d_in, src.data(), n * sizeof(T)).wait();
  bool ok = op.forward(d_in, d_out, N, H, W, 3, 3, dt, code, &q);
  if (!ok) { std::cout << "  [h2b code " << code << "] forward=false (unexpected)\n"; free(d_in,q); free(d_out,q); return false; }
  q.wait();
  q.memcpy(gpu.data(), d_out, n * sizeof(T)).wait();
  if (isFullRange)
    cvt_gold::gold_hsv_to_bgr<T, true>(cpu.data(), src.data(), (size_t)N * H * W, bidx, 3);
  else
    cvt_gold::gold_hsv_to_bgr<T, false>(cpu.data(), src.data(), (size_t)N * H * W, bidx, 3);
  int m = count_mismatch_tol<T>(gpu, cpu, tol);
  std::cout << "  [h2b  code " << code << "] mismatch=" << m << " (tol=" << tol << ")"
            << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

// Family E: YUV420 (codes 90-106, 127-134, 140-147). 8U only. H,W must be even
// (planar needs H%4==0 -> use H multiple of 4). Tolerances: YUV->BGR=2, BGR->YUV=1,
// gray=0.
static bool run_case_yuv420(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int yuv_h = H + H / 2;
  std::mt19937 rng(888 + code);
  std::uniform_int_distribution<int> d(0, 255);
  auto fill = [&](uint8_t* p, size_t n) { for (size_t i=0;i<n;++i) p[i]=(uint8_t)d(rng); };

  if (code == 106) {  // YUV2GRAY_420
    size_t nin = (size_t)N * yuv_h * W, nout = (size_t)N * H * W;
    std::vector<uint8_t> src(nin), gpu(nout), cpu(nout);
    fill(src.data(), nin);
    uint8_t* di = malloc_device<uint8_t>(nin, q);
    uint8_t* dou = malloc_device<uint8_t>(nout, q);
    q.memcpy(di, src.data(), nin).wait();
    bool ok = op.forward(di, dou, N, H, W, 1, 1, DType::U8, code, &q);
    if (!ok) { std::cout << "  [y420 code 106] forward=false\n"; free(di,q); free(dou,q); return false; }
    q.wait(); q.memcpy(gpu.data(), dou, nout).wait();
    cvt_gold::gold_yuv420_to_gray<uint8_t>(cpu.data(), src.data(), W, H, N);
    int m = count_mismatch_tol<uint8_t>(gpu, cpu, 0.0);
    std::cout << "  [y420 code 106 GRAY] mismatch=" << m << (m ? " FAIL" : " PASS") << "\n";
    free(di,q); free(dou,q); return m==0;
  }

  if (code >= 90 && code <= 105) {  // YUV420 -> BGR (Cout=3)
    bool isSemi = (code <= 97);
    int bidx = (code % 2 == 0) ? 2 : 0;
    int uidx = isSemi ? ((code % 4 >= 2) ? 0 : 1) : ((code % 4 >= 2) ? 1 : 0);
    size_t nin = (size_t)N * yuv_h * W, nout = (size_t)N * H * W * 3;
    std::vector<uint8_t> src(nin), gpu(nout), cpu(nout);
    fill(src.data(), nin);
    uint8_t* di = malloc_device<uint8_t>(nin, q);
    uint8_t* dou = malloc_device<uint8_t>(nout, q);
    q.memcpy(di, src.data(), nin).wait();
    bool ok = op.forward(di, dou, N, H, W, 1, 3, DType::U8, code, &q);
    if (!ok) { std::cout << "  [y420 code " << code << "] forward=false\n"; free(di,q); free(dou,q); return false; }
    q.wait(); q.memcpy(gpu.data(), dou, nout).wait();
    if (isSemi) cvt_gold::gold_yuv420_to_bgr_nv12<uint8_t>(cpu.data(), src.data(), W, H, N, 3, bidx, uidx);
    else        cvt_gold::gold_yuv420_to_bgr_planar<uint8_t>(cpu.data(), src.data(), W, H, N, 3, bidx, uidx);
    int m = count_mismatch_tol<uint8_t>(gpu, cpu, 2.0);
    std::cout << "  [y420 code " << code << " YUV->BGR] mismatch=" << m << " (tol=2)"
              << (m ? " FAIL" : " PASS") << "\n";
    free(di,q); free(dou,q); return m==0;
  }

  // BGR -> YUV420 (Cin=3)
  bool isSemi = (code >= 140);
  int bidx = isSemi ? ((code % 2 == 0) ? 2 : 0) : ((code % 2 == 0) ? 0 : 2);
  int uidx = isSemi ? ((code % 4 < 2) ? 0 : 1) : ((code <= 130) ? 0 : 1);
  size_t nin = (size_t)N * H * W * 3, nout = (size_t)N * yuv_h * W;
  std::vector<uint8_t> src(nin), gpu(nout), cpu(nout);
  fill(src.data(), nin);
  uint8_t* di = malloc_device<uint8_t>(nin, q);
  uint8_t* dou = malloc_device<uint8_t>(nout, q);
  q.memcpy(di, src.data(), nin).wait();
  bool ok = op.forward(di, dou, N, H, W, 3, 1, DType::U8, code, &q);
  if (!ok) { std::cout << "  [y420 code " << code << "] forward=false\n"; free(di,q); free(dou,q); return false; }
  q.wait(); q.memcpy(gpu.data(), dou, nout).wait();
  if (isSemi) cvt_gold::gold_bgr_to_yuv420_nv12<uint8_t>(cpu.data(), src.data(), W, H, N, 3, bidx, uidx);
  else        cvt_gold::gold_bgr_to_yuv420_planar<uint8_t>(cpu.data(), src.data(), W, H, N, 3, bidx, uidx);
  int m = count_mismatch_tol<uint8_t>(gpu, cpu, 1.0);
  std::cout << "  [y420 code " << code << " BGR->YUV] mismatch=" << m << " (tol=1)"
            << (m ? " FAIL" : " PASS") << "\n";
  free(di,q); free(dou,q); return m==0;
}

// Family F: YUV422 (codes 107-124 except 109,110,113,114). 8U only, W%2==0.
// Tolerances: YUV->BGR=2, gray=0.
static bool run_case_yuv422(CvtColor& op, queue& q, int code, int N, int H, int W) {
  int yuv_w = 2 * W;
  bool is_gray = (code == 123 || code == 124);
  int bidx = is_gray ? 0 : ((code % 2 == 1) ? 2 : 0);
  int yidx = (code == 107 || code == 108 || code == 111 || code == 112 || code == 123) ? 1 : 0;
  int uidx = (code == 117 || code == 118 || code == 121 || code == 122) ? 2 : 0;
  std::mt19937 rng(2024 + code);
  std::uniform_int_distribution<int> d(0, 255);
  auto fill = [&](uint8_t* p, size_t n) { for (size_t i=0;i<n;++i) p[i]=(uint8_t)d(rng); };

  if (is_gray) {
    size_t nin = (size_t)N * H * yuv_w, nout = (size_t)N * H * W;
    std::vector<uint8_t> src(nin), gpu(nout), cpu(nout);
    fill(src.data(), nin);
    uint8_t* di = malloc_device<uint8_t>(nin, q);
    uint8_t* dou = malloc_device<uint8_t>(nout, q);
    q.memcpy(di, src.data(), nin).wait();
    bool ok = op.forward(di, dou, N, H, W, 1, 1, DType::U8, code, &q);
    if (!ok) { std::cout << "  [y422 code " << code << "] forward=false\n"; free(di,q); free(dou,q); return false; }
    q.wait(); q.memcpy(gpu.data(), dou, nout).wait();
    cvt_gold::gold_yuv422_to_gray<uint8_t>(cpu.data(), src.data(), W, H, N, yidx);
    int m = count_mismatch_tol<uint8_t>(gpu, cpu, 0.0);
    std::cout << "  [y422 code " << code << " GRAY] mismatch=" << m << (m ? " FAIL" : " PASS") << "\n";
    free(di,q); free(dou,q); return m==0;
  }

  size_t nin = (size_t)N * H * yuv_w, nout = (size_t)N * H * W * 3;
  std::vector<uint8_t> src(nin), gpu(nout), cpu(nout);
  fill(src.data(), nin);
  uint8_t* di = malloc_device<uint8_t>(nin, q);
  uint8_t* dou = malloc_device<uint8_t>(nout, q);
  q.memcpy(di, src.data(), nin).wait();
  bool ok = op.forward(di, dou, N, H, W, 1, 3, DType::U8, code, &q);
  if (!ok) { std::cout << "  [y422 code " << code << "] forward=false\n"; free(di,q); free(dou,q); return false; }
  q.wait(); q.memcpy(gpu.data(), dou, nout).wait();
  cvt_gold::gold_yuv422_to_bgr<uint8_t>(cpu.data(), src.data(), W, H, N, 3, bidx, yidx, uidx);
  int m = count_mismatch_tol<uint8_t>(gpu, cpu, 2.0);
  std::cout << "  [y422 code " << code << " YUV->BGR] mismatch=" << m << " (tol=2)"
            << (m ? " FAIL" : " PASS") << "\n";
  free(di,q); free(dou,q); return m==0;
}

int main() {
  auto op = cvcuda::cvt::create_cvt();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

  bool all = true;
  int N = 2, H = 6, W = 8;
  std::cout << "--- Family A: U8 ---\n";
  for (int c = 0; c <= 5; ++c) all &= run_case<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family A: U16 ---\n";
  for (int c = 0; c <= 5; ++c) all &= run_case<uint16_t>(*op, q, c, N, H, W);
  std::cout << "--- Family A: S32 ---\n";
  for (int c = 0; c <= 5; ++c) all &= run_case<int32_t>(*op, q, c, N, H, W);
  std::cout << "--- Family A: F32 ---\n";
  for (int c = 0; c <= 5; ++c) all &= run_case<float>(*op, q, c, N, H, W);
  std::cout << "--- Family A: F64 ---\n";
  for (int c = 0; c <= 5; ++c) all &= run_case<double>(*op, q, c, N, H, W);

  // ---- Family B: gray ----
  std::cout << "--- Family B: BGR->GRAY U8 (tol 1) ---\n";
  for (int c : {6, 7, 10, 11}) all &= run_case_gray<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family B: BGR->GRAY U16 (tol 2) ---\n";
  for (int c : {6, 7, 10, 11}) all &= run_case_gray<uint16_t>(*op, q, c, N, H, W);
  std::cout << "--- Family B: BGR->GRAY F32 (tol ERR1_4) ---\n";
  for (int c : {6, 7, 10, 11}) all &= run_case_gray<float>(*op, q, c, N, H, W);
  std::cout << "--- Family B: GRAY->BGR U8 (exact) ---\n";
  for (int c : {8, 9}) all &= run_case_gray2bgr<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family B: GRAY->BGR U16 (exact) ---\n";
  for (int c : {8, 9}) all &= run_case_gray2bgr<uint16_t>(*op, q, c, N, H, W);
  std::cout << "--- Family B: GRAY->BGR F32 (exact) ---\n";
  for (int c : {8, 9}) all &= run_case_gray2bgr<float>(*op, q, c, N, H, W);
  std::cout << "--- Family B: GRAY->BGR F64 (exact) ---\n";
  for (int c : {8, 9}) all &= run_case_gray2bgr<double>(*op, q, c, N, H, W);

  // ---- Family C: packed YUV ----
  std::cout << "--- Family C: BGR<->YUV U8 (tol 1) ---\n";
  for (int c : {82, 83, 84, 85}) all &= run_case_yuv<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family C: BGR<->YUV U16 ---\n";
  for (int c : {82, 83, 84, 85}) all &= run_case_yuv<uint16_t>(*op, q, c, N, H, W);
  std::cout << "--- Family C: BGR<->YUV F32 (tol ERR1_4) ---\n";
  for (int c : {82, 83, 84, 85}) all &= run_case_yuv<float>(*op, q, c, N, H, W);

  // ---- Family D: HSV ----
  std::cout << "--- Family D: BGR->HSV U8 (tol 1, circ hue) ---\n";
  for (int c : {40, 41, 66, 67}) all &= run_case_bgr2hsv<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family D: BGR->HSV F32 (tol ERR2_3, circ hue) ---\n";
  for (int c : {40, 41, 66, 67}) all &= run_case_bgr2hsv<float>(*op, q, c, N, H, W);
  std::cout << "--- Family D: HSV->BGR U8 (tol 1) ---\n";
  for (int c : {54, 55, 70, 71}) all &= run_case_hsv2bgr<uint8_t>(*op, q, c, N, H, W);
  std::cout << "--- Family D: HSV->BGR F32 (tol ERR1_4) ---\n";
  for (int c : {54, 55, 70, 71}) all &= run_case_hsv2bgr<float>(*op, q, c, N, H, W);

  // ---- Family E: YUV420 (H,W multiples of 4 for planar) ----
  {
    int He = 8, We = 8;  // %4==0 satisfies both semi and planar
    std::cout << "--- Family E: YUV420->BGR U8 (tol 2) ---\n";
    for (int c : {90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105})
      all &= run_case_yuv420(*op, q, c, 2, He, We);
    std::cout << "--- Family E: YUV2GRAY_420 U8 (exact) ---\n";
    all &= run_case_yuv420(*op, q, 106, 2, He, We);
    std::cout << "--- Family E: BGR->YUV420 U8 (tol 1) ---\n";
    for (int c : {127, 128, 129, 130, 131, 132, 133, 134, 140, 141, 142, 143, 144, 145, 146, 147})
      all &= run_case_yuv420(*op, q, c, 2, He, We);
  }

  // ---- Family F: YUV422 (W even) ----
  {
    int Hf = 6, Wf = 8;
    std::cout << "--- Family F: YUV422->BGR U8 (tol 2) ---\n";
    for (int c : {107, 108, 111, 112, 115, 116, 117, 118, 119, 120, 121, 122})
      all &= run_case_yuv422(*op, q, c, 2, Hf, Wf);
    std::cout << "--- Family F: YUV2GRAY_422 U8 (exact) ---\n";
    for (int c : {123, 124}) all &= run_case_yuv422(*op, q, c, 2, Hf, Wf);
  }

  // ---- negative tests ----
  std::cout << "--- negative ---\n";
  uint8_t* dummy = malloc_device<uint8_t>(16, q);
  q.wait();
  // 16F + alpha-add (code 0: sch3->dch4) must be rejected.
  bool r1 = op->forward(dummy, dummy, 1, 2, 2, 3, 4, DType::F16, 0, &q);
  std::cout << "  [F16 + BGR2BGRA (alpha-add)] rejected=" << (!r1) << (r1 ? " FAIL" : " PASS") << "\n";
  all &= !r1;
  // unsupported code (50 = RGB2Lab, not implemented by CV-CUDA).
  bool r2 = op->forward(dummy, dummy, 1, 2, 2, 3, 3, DType::U8, 50, &q);
  std::cout << "  [unsupported code 50] rejected=" << (!r2) << (r2 ? " FAIL" : " PASS") << "\n";
  all &= !r2;
  // wrong channels (code 4 BGR2RGB expects sch=3,dch=3; pass sch=4).
  bool r3 = op->forward(dummy, dummy, 1, 2, 2, 4, 3, DType::U8, 4, &q);
  std::cout << "  [wrong channels] rejected=" << (!r3) << (r3 ? " FAIL" : " PASS") << "\n";
  all &= !r3;
  free(dummy, q);

  std::cout << (all ? "=== ALL TESTS PASSED ===" : "=== TESTS FAILED ===") << "\n";
  return all ? 0 : 1;
}
