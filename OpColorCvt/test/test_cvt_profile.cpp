// Per-kernel profiling test for the SYCL CvtColor port. VOX_PROFILE=1 (set
// self-contained before construction). Measures one representative code per
// family at a realistic image size (1280x720, N=1), reporting best kernel ms
// (SYCL event), best wall ms, and throughput (Mpix/s).
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   
//   test/test_cvt_profile.cpp \
//   -o test/test_cvt_profile
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_cvt_profile

#include "../colorcvt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;
using cvcuda::cvt::CvtColor;
using cvcuda::cvt::DType;

// Compute in/out element counts + Cin/Cout for a code given RGB shape (N,H,W).
static void sizes(int code, int N, int H, int W, int& Cin, int& Cout, size_t& in_elems,
                  size_t& out_elems) {
  int yuv_h = H + H / 2;
  int yuv_w = 2 * W;
  auto bgr = [&](int ci, int co) { Cin = ci; Cout = co; in_elems = (size_t)N * H * W * ci; out_elems = (size_t)N * H * W * co; };
  if (code >= 90 && code <= 105) { Cin = 1; Cout = 3; in_elems = (size_t)N * yuv_h * W; out_elems = (size_t)N * H * W * 3; }       // YUV420->BGR
  else if (code == 106)          { Cin = 1; Cout = 1; in_elems = (size_t)N * yuv_h * W; out_elems = (size_t)N * H * W; }            // YUV2GRAY_420
  else if ((code >= 127 && code <= 134) || (code >= 140 && code <= 147)) { Cin = 3; Cout = 1; in_elems = (size_t)N * H * W * 3; out_elems = (size_t)N * yuv_h * W; }  // BGR->YUV420
  else if (code >= 107 && code <= 122) { Cin = 1; Cout = 3; in_elems = (size_t)N * H * yuv_w; out_elems = (size_t)N * H * W * 3; } // YUV422->BGR
  else if (code == 123 || code == 124) { Cin = 1; Cout = 1; in_elems = (size_t)N * H * yuv_w; out_elems = (size_t)N * H * W; }      // YUV2GRAY_422
  else if (code == 4 || code == 5) bgr(3, 3);
  else if (code == 0 || code == 2) bgr(3, 4);
  else if (code == 1 || code == 3) bgr(4, 3);
  else if (code == 6 || code == 7) bgr(3, 1);
  else if (code == 10 || code == 11) bgr(4, 1);
  else if (code == 8) bgr(1, 3);
  else if (code == 9) bgr(1, 4);
  else bgr(3, 3);  // YUV/HSV packed
}

struct Case { int code; const char* name; DType dtype; };

int main() {
  setenv("VOX_PROFILE", "1", 1);
  auto op = cvcuda::cvt::create_cvt();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

  int N = 1, H = 720, W = 1280;
  std::cout << "image " << W << "x" << H << " N=" << N << ", " << 20 << " runs (best)\n";

  std::vector<Case> cases = {
    {0,  "BGR2BGRA   (A)", DType::U8},
    {4,  "BGR2RGB    (A)", DType::U8},
    {6,  "BGR2GRAY   (B)", DType::U8},
    {8,  "GRAY2BGR   (B)", DType::U8},
    {82, "BGR2YUV    (C)", DType::U8},
    {84, "YUV2BGR    (C)", DType::U8},
    {40, "BGR2HSV    (D)", DType::U8},
    {40, "BGR2HSV    (D)", DType::F32},
    {54, "HSV2BGR    (D)", DType::U8},
    {91, "YUV2BGR_NV12(E)", DType::U8},
    {141,"BGR2YUV_NV12(E)", DType::U8},
    {106,"YUV2GRAY_420(E)", DType::U8},
    {116,"YUV2BGR_YUY2(F)", DType::U8},
    {124,"YUV2GRAY_YUY2(F)", DType::U8},
  };

  printf("%-22s %10s %10s %12s\n", "code", "kernel_ms", "wall_ms", "Mpix/s");
  printf("%-22s %10s %10s %12s\n", "----", "---------", "-------", "------");
  for (auto& c : cases) {
    int Cin, Cout;
    size_t in_elems, out_elems;
    sizes(c.code, N, H, W, Cin, Cout, in_elems, out_elems);
    size_t in_bytes = in_elems * sizeof(uint8_t);  // U8; F32 uses 4 bytes/elem
    if (c.dtype == DType::F32) in_bytes = in_elems * 4;
    size_t out_bytes = out_elems * ((c.dtype == DType::F32) ? 4 : 1);

    std::vector<uint8_t> host_in(in_bytes);
    std::mt19937 rng(7);
    for (auto& v : host_in) v = (uint8_t)(rng() & 0xff);
    void* d_in = (c.dtype == DType::F32) ? (void*)malloc_device<float>(in_elems, q)
                                         : (void*)malloc_device<uint8_t>(in_bytes, q);
    void* d_out = (c.dtype == DType::F32) ? (void*)malloc_device<float>(out_elems, q)
                                          : (void*)malloc_device<uint8_t>(out_bytes, q);
    q.memcpy(d_in, host_in.data(), in_bytes).wait();

    auto run = [&](double& wall_ms, uint64_t& kernel_ns) {
      auto t0 = std::chrono::steady_clock::now();
      op->forward(d_in, d_out, N, H, W, Cin, Cout, c.dtype, c.code, &q);
      q.wait();
      auto t1 = std::chrono::steady_clock::now();
      wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      op->last_profile(kernel_ns);
    };
    double w; uint64_t k;
    for (int i = 0; i < 3; ++i) run(w, k);  // warmup
    double best_wall = 1e9;
    uint64_t best_kernel = UINT64_MAX;
    for (int i = 0; i < 20; ++i) {
      run(w, k);
      best_wall = std::min(best_wall, w);
      best_kernel = std::min(best_kernel, k);
    }
    double k_ms = best_kernel / 1e6;
    double mpix = (double)H * W / (k_ms * 1e-3) / 1e6;
    printf("%-22s %10.4f %10.4f %12.1f\n", c.name, k_ms, best_wall, mpix);
    sycl::free(d_in, q);
    sycl::free(d_out, q);
  }
  std::cout << "\n=== profile done ===\n";
  return 0;
}
