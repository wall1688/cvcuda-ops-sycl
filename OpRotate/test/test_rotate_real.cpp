// Real random-data test for the SYCL Rotate port. Random src + random angle,
// all 3 interpolations (NEAREST/LINEAR/CUBIC), U8 + F32. Compares GPU vs CPU
// gold:
//   - NEAREST is bit-exact for any angle (integer indexing from shared coeffs).
//   - LINEAR/CUBIC use only mul/add (no div/sqrt), so the Intel GPU FP-division
//     quirk does NOT apply; residual divergence is FMA-contraction ULP noise.
//     Integral types: tol=1, mismatches classified as rint/clamp boundary cases
//     (pre-saturate float within BOUND_TOL of a half-integer or dtype bound).
//     F32: tol=1e-4 (LINEAR) / 1e-3 (CUBIC). Also reports wall-clock throughput.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_rotate_real.cpp \
//   -o test/test_rotate_real
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu \
//        ./test/test_rotate_real [N] [S] [C] [runs]

#include "../rotate.hpp"
#include "rotate_gold.hpp"

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
using cvcuda::rotate::Rotate;
using cvcuda::rotate::DType;
using cvcuda::rotate::Interp;

static constexpr float BOUND_TOL = 1e-3f;

// Is the pre-saturate float result on a rounding/clamp boundary? (integral only)
template <typename T>
static bool is_boundary(float res) {
  float fl = std::floor(res);
  float frac = res - fl;
  if (std::fabs(frac - 0.5f) < BOUND_TOL) return true;        // rint flip point
  float mn = (float)cvcuda::rotate::TypeMin<T>();
  float mx = (float)cvcuda::rotate::TypeMax<T>();
  if (std::fabs(res - mn) < BOUND_TOL) return true;            // clamp lower
  if (std::fabs(res - mx) < BOUND_TOL) return true;            // clamp upper
  return false;
}

template <typename T>
static bool run_real(Rotate& op, queue& q, int N, int S, int C, double angle, Interp interp,
                     DType dt, double tol, const std::string& tag) {
  size_t nin  = (size_t)N * S * S * C;
  size_t nout = nin;  // square in/out
  std::vector<T> src(nin), gpu(nout), cpu(nout);
  std::mt19937 rng(424242 + (int)angle + C + (int)sizeof(T) + (int)interp);

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(0, (int)cvcuda::rotate::TypeMax<T>());
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_real_distribution<float> d(0.f, 1.f);
    for (auto& v : src) v = (T)d(rng);
  }

  T* d_in  = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nout, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();

  double sx = (S - 1) / 2.0, sy = (S - 1) / 2.0;
  bool ok = op.forward(d_in, d_out, N, S, S, S, S, C, angle, sx, sy, interp, dt, &q);
  if (!ok) {
    std::cout << "  [" << tag << "] forward returned false (unexpected)\n";
    free(d_in, q); free(d_out, q);
    return false;
  }
  q.wait();
  q.memcpy(gpu.data(), d_out, nout * sizeof(T)).wait();

  rotate_gold::gold_rotate<T>(cpu.data(), src.data(), N, S, S, S, S, C, angle, sx, sy, interp);

  int mismatch = 0, boundary = 0;
  double maxdiff = 0.0;
  if constexpr (std::is_integral_v<T>) {
    std::vector<float> res(nout);
    rotate_gold::gold_rotate_res<T>(src.data(), N, S, S, S, S, C, angle, sx, sy, interp, res.data());
    for (size_t i = 0; i < nout; ++i) {
      double d = std::fabs((double)gpu[i] - (double)cpu[i]);
      if (d > maxdiff) maxdiff = d;
      if (d > tol) {
        ++mismatch;
        if (is_boundary<T>(res[i])) ++boundary;
      }
    }
  } else {
    for (size_t i = 0; i < nout; ++i) {
      double d = std::fabs((double)gpu[i] - (double)cpu[i]);
      if (d > maxdiff) maxdiff = d;
      if (d > tol) ++mismatch;
    }
  }

  bool pass;
  if constexpr (std::is_integral_v<T>) {
    pass = (mismatch == 0 || boundary == mismatch);
    std::cout << "  [" << tag << "] mismatch=" << mismatch << " boundary=" << boundary
              << " maxdiff=" << maxdiff << " (tol=" << tol << ")"
              << (pass ? " PASS" : " FAIL") << "\n";
  } else {
    pass = (mismatch == 0);
    std::cout << "  [" << tag << "] mismatch=" << mismatch << " maxdiff=" << maxdiff
              << " (tol=" << tol << ")"
              << (pass ? " PASS" : " FAIL") << "\n";
  }
  free(d_in, q); free(d_out, q);
  return pass;
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 4;
  int S = argc > 2 ? atoi(argv[2]) : 128;
  int C = argc > 3 ? atoi(argv[3]) : 3;
  int runs = argc > 4 ? atoi(argv[4]) : 10;

  auto op = cvcuda::rotate::create_rotate();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "N=" << N << " S=" << S << " C=" << C << " runs=" << runs << "\n";

  // Fixed non-axis-aligned angle so LINEAR/CUBIC exercise real interpolation.
  double angle = 23.5;
  bool all = true;
  std::cout << "--- U8 NEAREST (tol 0, expect exact) ---\n";
  all &= run_real<uint8_t>(*op, q, N, S, C, angle, Interp::NEAREST, DType::U8, 0.0, "U8 NEAREST");
  std::cout << "--- U8 LINEAR (tol 1, boundary-classified) ---\n";
  all &= run_real<uint8_t>(*op, q, N, S, C, angle, Interp::LINEAR, DType::U8, 1.0, "U8 LINEAR");
  std::cout << "--- U8 CUBIC (tol 1, boundary-classified) ---\n";
  all &= run_real<uint8_t>(*op, q, N, S, C, angle, Interp::CUBIC, DType::U8, 1.0, "U8 CUBIC");
  std::cout << "--- F32 NEAREST (tol 0, expect exact) ---\n";
  all &= run_real<float>(*op, q, N, S, C, angle, Interp::NEAREST, DType::F32, 0.0, "F32 NEAREST");
  std::cout << "--- F32 LINEAR (tol 1e-4) ---\n";
  all &= run_real<float>(*op, q, N, S, C, angle, Interp::LINEAR, DType::F32, 1e-4, "F32 LINEAR");
  std::cout << "--- F32 CUBIC (tol 1e-3) ---\n";
  all &= run_real<float>(*op, q, N, S, C, angle, Interp::CUBIC, DType::F32, 1e-3, "F32 CUBIC");

  // ---- wall-clock throughput (F32 LINEAR, the heaviest exact-ish path) ----
  size_t nin = (size_t)N * S * S * C;
  std::vector<float> src(nin);
  std::mt19937 rng(11111);
  std::uniform_real_distribution<float> d(0.f, 1.f);
  for (auto& v : src) v = d(rng);
  float* d_in  = malloc_device<float>(nin, q);
  float* d_out = malloc_device<float>(nin, q);
  q.memcpy(d_in, src.data(), nin * sizeof(float)).wait();
  double sx = (S - 1) / 2.0, sy = (S - 1) / 2.0;

  for (int i = 0; i < 3; ++i)
    op->forward(d_in, d_out, N, S, S, S, S, C, angle, sx, sy, Interp::LINEAR, DType::F32, &q);
  q.wait();

  double best = 1e9, sum = 0;
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_in, d_out, N, S, S, S, S, C, angle, sx, sy, Interp::LINEAR, DType::F32, &q);
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms); sum += ms;
  }
  double avg = sum / runs;
  double pixels = (double)N * S * S;
  std::cout << "[perf] F32 LINEAR wall best=" << best << "ms avg=" << avg << "ms"
            << " throughput=" << (pixels / best / 1e3) << " Mpix/s\n";
  free(d_in, q); free(d_out, q);

  std::cout << (all ? "=== REAL TESTS PASSED ===" : "=== REAL TESTS FAILED ===") << "\n";
  return all ? 0 : 1;
}
