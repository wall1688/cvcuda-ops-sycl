// Real random-data test for the SYCL Normalize port. Random src/base/scale
// (typical per-channel mean/stddev normalization: scalar base, per-channel
// scale). Compares GPU vs CPU gold with tolerance and, for stddev mode on
// integral types, classifies divergences as FP-boundary (the 1/sqrt float
// division uses Intel GPU reciprocal approximation — see intel-gpu-fp-quirk).
// Also reports wall-clock throughput.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_normalize_real.cpp \
//   -o test/test_normalize_real
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_normalize_real [N] [H] [W] [C] [runs]

#include "../normalize.hpp"
#include "normalize_gold.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

using namespace sycl;
using cvcuda::normalize::Normalize;
using cvcuda::normalize::DType;
using cvcuda::normalize::Shape;

static constexpr float BOUND_TOL = 1e-3f;

// Is the pre-saturate float result on a rounding/clamp boundary? (integral only)
template <typename T>
static bool is_boundary(float res) {
  float fl = std::floor(res);
  float frac = res - fl;
  if (std::fabs(frac - 0.5f) < BOUND_TOL) return true;        // rint flip point
  float mn = (float)cvcuda::normalize::TypeMin<T>();
  float mx = (float)cvcuda::normalize::TypeMax<T>();
  if (std::fabs(res - mn) < BOUND_TOL) return true;            // clamp lower
  if (std::fabs(res - mx) < BOUND_TOL) return true;            // clamp upper
  return false;
}

template <typename T>
static bool run_real(Normalize& op, queue& q, int N, int H, int W, int C, bool stddev, DType dt,
                     const std::string& tag) {
  size_t nin = (size_t)N * H * W * C;
  std::vector<T> src(nin), gpu(nin), cpu(nin);
  std::mt19937 rng(98765 + (stddev ? 1 : 0) + (int)sizeof(T));

  // Typical Normalize usage: scalar base (mean), per-channel scale (stddev).
  Shape bS{1, 1, 1, 1}, sS{1, 1, 1, C};
  std::vector<float> base(1, 127.5f);
  std::vector<float> scale(C);
  std::uniform_real_distribution<float> sd(0.01f, 1.0f);
  for (int c = 0; c < C; ++c) scale[c] = sd(rng);

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::normalize::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<float> d(0.f, 1.f);
    for (auto& v : src) v = (T)d(rng);
  }

  T* d_in = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nin, q);
  float* d_base = malloc_device<float>(1, q);
  float* d_scale = malloc_device<float>(C, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();
  q.memcpy(d_base, base.data(), sizeof(float)).wait();
  q.memcpy(d_scale, scale.data(), C * sizeof(float)).wait();

  float eps = stddev ? 1e-5f : 0.f;
  float gscale = 1.0f, shift = 0.0f;
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

  int mismatch = 0, boundary = 0;
  double maxdiff = 0.0;
  double tol = std::is_integral_v<T> ? 1.0 : 1e-4;
  if constexpr (std::is_integral_v<T>) {
    std::vector<float> res(nin);
    normalize_gold::gold_normalize_res<T>(src.data(), base.data(), scale.data(), N, H, W, C, bS, sS,
                                          gscale, shift, eps, stddev, res.data());
    for (size_t i = 0; i < nin; ++i) {
      double d = std::fabs((double)gpu[i] - (double)cpu[i]);
      if (d > maxdiff) maxdiff = d;
      if (d > tol) {
        ++mismatch;
        if (is_boundary<T>(res[i])) ++boundary;
      }
    }
  } else {
    for (size_t i = 0; i < nin; ++i) {
      double d = std::fabs((double)gpu[i] - (double)cpu[i]);
      if (d > maxdiff) maxdiff = d;
      if (d > tol) ++mismatch;
    }
  }

  if constexpr (std::is_integral_v<T>) {
    bool pass = (mismatch == 0 || boundary == mismatch);
    std::cout << "  [" << tag << "] mismatch=" << mismatch << " boundary=" << boundary
              << " maxdiff=" << maxdiff << " (tol=" << tol << ")"
              << (pass ? " PASS" : " FAIL") << "\n";
    free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);
    return pass;
  } else {
    bool pass = (mismatch == 0);
    std::cout << "  [" << tag << "] mismatch=" << mismatch << " maxdiff=" << maxdiff
              << " (tol=" << tol << ")"
              << (pass ? " PASS" : " FAIL") << "\n";
    free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);
    return pass;
  }
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 64;
  int H = argc > 2 ? atoi(argv[2]) : 128;
  int W = argc > 3 ? atoi(argv[3]) : 128;
  int C = argc > 4 ? atoi(argv[4]) : 4;
  int runs = argc > 5 ? atoi(argv[5]) : 10;

  auto op = cvcuda::normalize::create_normalize();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "N=" << N << " H=" << H << " W=" << W << " C=" << C << " runs=" << runs << "\n";

  bool all = true;
  std::cout << "--- U8 plain (tol 1, expect exact) ---\n";
  all &= run_real<uint8_t>(*op, q, N, H, W, C, false, DType::U8, "U8 plain");
  std::cout << "--- U8 stddev (tol 1, boundary-classified) ---\n";
  all &= run_real<uint8_t>(*op, q, N, H, W, C, true, DType::U8, "U8 stddev");
  std::cout << "--- F32 plain (tol 1e-4, expect exact) ---\n";
  all &= run_real<float>(*op, q, N, H, W, C, false, DType::F32, "F32 plain");
  std::cout << "--- F32 stddev (tol 1e-4, ULP noise) ---\n";
  all &= run_real<float>(*op, q, N, H, W, C, true, DType::F32, "F32 stddev");

  // ---- wall-clock throughput (F32 plain, the heaviest exact path) ----
  size_t nin = (size_t)N * H * W * C;
  std::vector<float> src(nin);
  std::mt19937 rng(11111);
  std::uniform_real_distribution<float> d(0.f, 1.f);
  for (auto& v : src) v = d(rng);
  Shape bS{1, 1, 1, 1}, sS{1, 1, 1, C};
  std::vector<float> base(1, 0.5f), scale(C, 0.5f);
  float* d_in = malloc_device<float>(nin, q);
  float* d_out = malloc_device<float>(nin, q);
  float* d_base = malloc_device<float>(1, q);
  float* d_scale = malloc_device<float>(C, q);
  q.memcpy(d_in, src.data(), nin * sizeof(float)).wait();
  q.memcpy(d_base, base.data(), sizeof(float)).wait();
  q.memcpy(d_scale, scale.data(), C * sizeof(float)).wait();

  for (int i = 0; i < 3; ++i)
    op->forward(d_in, d_base, d_scale, d_out, N, H, W, C, bS, sS, 1.f, 0.f, 0.f, DType::F32, 0, &q);
  q.wait();

  double best = 1e9, sum = 0;
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_in, d_base, d_scale, d_out, N, H, W, C, bS, sS, 1.f, 0.f, 0.f, DType::F32, 0, &q);
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms); sum += ms;
  }
  double avg = sum / runs;
  double pixels = (double)N * H * W;
  std::cout << "[perf] F32 plain wall best=" << best << "ms avg=" << avg << "ms"
            << " throughput=" << (pixels / best / 1e3) << " Mpix/s"
            << " (" << (nin / best / 1e3) << " Melem/s)\n";
  free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);

  std::cout << (all ? "=== REAL TESTS PASSED ===" : "=== REAL TESTS FAILED ===") << "\n";
  return all ? 0 : 1;
}
