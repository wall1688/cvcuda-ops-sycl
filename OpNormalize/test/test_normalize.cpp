// Correctness test for the SYCL Normalize port. Hand-picked exact values so the
// float arithmetic lands on exactly-representable results (powers-of-2 scale,
// integer base/shift, stddev eps chosen so 1/sqrt is an exact power of 2) ->
// bit-exact vs CPU gold (tolerance 0). Exercises plain + stddev modes, all 6
// dtypes, C in {1,3,4}, and scalar / per-channel / N/H/W broadcast base+scale.
// Negative tests: C==2 and C>4 rejected, bad broadcast rejected.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_normalize.cpp \
//   -o test/test_normalize
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_normalize

#include "../normalize.hpp"
#include "normalize_gold.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <string>
#include <type_traits>
#include <vector>

using namespace sycl;
using cvcuda::normalize::Normalize;
using cvcuda::normalize::DType;
using cvcuda::normalize::Shape;

template <typename T>
static int count_mismatch_tol(const std::vector<T>& gpu, const std::vector<T>& cpu, double tol) {
  int m = 0;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs((double)gpu[i] - (double)cpu[i]) > tol) ++m;
  return m;
}

// One case. tol=0 (bit-exact) by construction — see file header.
template <typename T>
static bool run_case(Normalize& op, queue& q, int N, int H, int W, int C, Shape bS, Shape sS,
                     float gscale, float shift, float eps, bool stddev, DType dt,
                     const std::string& tag) {
  size_t nin = (size_t)N * H * W * C;
  std::vector<T> src(nin), gpu(nin), cpu(nin);

  std::mt19937 rng(12345 + C * 7 + (stddev ? 1000 : 0) + (int)gscale * 13 + (int)eps);
  if constexpr (std::is_signed_v<T>) {
    std::uniform_int_distribution<int> d(-8, 8);
    for (auto& v : src) v = (T)d(rng);
  } else if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, 200);
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_int_distribution<int> d(0, 200);  // integer-valued floats -> exact
    for (auto& v : src) v = (T)d(rng);
  }

  size_t nb = (size_t)bS.N * bS.H * bS.W * bS.C;
  size_t ns = (size_t)sS.N * sS.H * sS.W * sS.C;
  std::vector<float> base(nb), scale(ns);
  for (size_t i = 0; i < nb; ++i) base[i] = (float)(i % 7);  // 0..6 (exact)
  if (stddev) {
    for (size_t i = 0; i < ns; ++i) scale[i] = 0.0f;  // scale=0 -> 1/sqrt(eps) exact
  } else {
    for (size_t i = 0; i < ns; ++i) scale[i] = (i % 2) ? 2.0f : 1.0f;  // 1 or 2 (exact)
  }

  T* d_in = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nin, q);
  float* d_base = malloc_device<float>(nb, q);
  float* d_scale = malloc_device<float>(ns, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();
  q.memcpy(d_base, base.data(), nb * sizeof(float)).wait();
  q.memcpy(d_scale, scale.data(), ns * sizeof(float)).wait();

  uint32_t flags = stddev ? cvcuda::normalize::SCALE_IS_STDDEV : 0;
  bool ok = op.forward(d_in, d_base, d_scale, d_out, N, H, W, C, bS, sS, gscale, shift, eps, dt, flags,
                       &q);
  if (!ok) {
    std::cout << "  [" << tag << "] forward returned false (unexpected)\n";
    free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);
    return false;
  }
  q.wait();
  q.memcpy(gpu.data(), d_out, nin * sizeof(T)).wait();

  normalize_gold::gold_normalize<T>(cpu.data(), src.data(), base.data(), scale.data(), N, H, W, C, bS,
                                    sS, gscale, shift, eps, stddev);
  int m = count_mismatch_tol<T>(gpu, cpu, 0.0);
  std::cout << "  [" << tag << "] mismatch=" << m << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);
  return m == 0;
}

template <typename T>
static bool run_dtype(Normalize& op, queue& q, const std::string& name, DType dt) {
  bool all = true;
  int N = 2, H = 3, W = 4;
  // plain: scalar base+scale, C=1
  all &= run_case<T>(op, q, N, H, W, 1, {1,1,1,1}, {1,1,1,1}, 2.f, 5.f, 0.f, false, dt, name+"/plain C1 scalar");
  // plain: per-channel base, per-pixel scalar scale, C=3
  all &= run_case<T>(op, q, N, H, W, 3, {1,1,1,3}, {N,H,W,1}, 1.f, 0.f, 0.f, false, dt, name+"/plain C3 bCh sPix");
  // plain: full base, N-broadcast scale, C=4
  all &= run_case<T>(op, q, N, H, W, 4, {N,H,W,4}, {1,H,W,4}, 2.f, 5.f, 0.f, false, dt, name+"/plain C4 bFull sN");
  // plain: H-broadcast base, W-broadcast scale, C=3 (exercises .5 via scale=1 -> no; uses gscale=2)
  all &= run_case<T>(op, q, N, H, W, 3, {N,1,W,3}, {N,H,1,3}, 1.f, 5.f, 0.f, false, dt, name+"/plain C3 bH sW");
  // stddev: scalar base, per-channel scale(=0), eps=1 -> mul=1
  all &= run_case<T>(op, q, N, H, W, 4, {1,1,1,1}, {1,1,1,4}, 2.f, 5.f, 1.f, true, dt, name+"/std C4 eps1");
  // stddev: per-channel base, per-pixel scale(=0), eps=4 -> mul=0.5 (round-to-even)
  all &= run_case<T>(op, q, N, H, W, 3, {1,1,1,3}, {N,H,W,1}, 1.f, 0.f, 4.f, true, dt, name+"/std C3 eps4");
  // stddev: scalar, eps=16 -> mul=0.25
  all &= run_case<T>(op, q, N, H, W, 1, {1,1,1,1}, {1,1,1,1}, 2.f, 5.f, 16.f, true, dt, name+"/std C1 eps16");
  return all;
}

int main() {
  auto op = cvcuda::normalize::create_normalize();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

  bool all = true;
  std::cout << "--- U8 ---\n";   all &= run_dtype<uint8_t>(*op, q, "U8",  DType::U8);
  std::cout << "--- S8 ---\n";   all &= run_dtype<int8_t>(*op, q,  "S8",  DType::S8);
  std::cout << "--- U16 ---\n";  all &= run_dtype<uint16_t>(*op, q, "U16", DType::U16);
  std::cout << "--- S16 ---\n";  all &= run_dtype<int16_t>(*op, q, "S16", DType::S16);
  std::cout << "--- S32 ---\n";  all &= run_dtype<int32_t>(*op, q, "S32", DType::S32);
  std::cout << "--- F32 ---\n";  all &= run_dtype<float>(*op, q,     "F32", DType::F32);

  // ---- negative tests ----
  std::cout << "--- negative ---\n";
  uint8_t* dummy = malloc_device<uint8_t>(16, q);
  float* dummyf = malloc_device<float>(16, q);
  q.wait();
  // C==2 must be rejected (CV-CUDA dispatch table null slot).
  bool r1 = op->forward(dummy, dummyf, dummyf, dummy, 1, 2, 2, 2, {1,1,1,1}, {1,1,1,1}, 1.f, 0.f, 0.f, DType::U8, 0, &q);
  std::cout << "  [C==2] rejected=" << (!r1) << (r1 ? " FAIL" : " PASS") << "\n";
  all &= !r1;
  // C>4 must be rejected.
  bool r2 = op->forward(dummy, dummyf, dummyf, dummy, 1, 2, 2, 5, {1,1,1,1}, {1,1,1,1}, 1.f, 0.f, 0.f, DType::U8, 0, &q);
  std::cout << "  [C==5] rejected=" << (!r2) << (r2 ? " FAIL" : " PASS") << "\n";
  all &= !r2;
  // bad broadcast: baseShape.H=3 when H=2 (not 1, not H) must be rejected.
  bool r3 = op->forward(dummy, dummyf, dummyf, dummy, 1, 2, 2, 3, {1,3,1,3}, {1,1,1,1}, 1.f, 0.f, 0.f, DType::U8, 0, &q);
  std::cout << "  [bad broadcast] rejected=" << (!r3) << (r3 ? " FAIL" : " PASS") << "\n";
  all &= !r3;
  free(dummy, q); free(dummyf, q);

  std::cout << (all ? "=== ALL TESTS PASSED ===" : "=== TESTS FAILED ===") << "\n";
  return all ? 0 : 1;
}
