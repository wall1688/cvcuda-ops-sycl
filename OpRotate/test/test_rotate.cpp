// Correctness test for the SYCL Rotate port. Hand-picked cases so the float
// arithmetic lands on exactly-representable results -> bit-exact vs CPU gold
// (tolerance 0):
//   - NEAREST is bit-exact for ANY angle: srcX/srcY are computed in double from
//     host-computed coeffs (shared bit-identically by kernel and gold), cast to
//     float, then floor(idx+0.5) -> integer; the pixel read is exact.
//   - LINEAR at 0 deg (identity) and 180 deg (c=-1,s=0 -> integer srcX/srcY):
//     weights collapse to an exact 1.0 on one tap -> exact pixel copy.
//   - CUBIC at 0 deg (identity): cubic_coeffs(0) = [0,1,0,0] -> exact center tap.
// Exercises all 4 dtypes (U8/U16/S16/F32), C in {1,3,4}, Hin!=Hout. Negative
// tests: C==2 and C>4 rejected.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_rotate.cpp \
//   -o test/test_rotate
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_rotate

#include "../rotate.hpp"
#include "rotate_gold.hpp"

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
using cvcuda::rotate::Rotate;
using cvcuda::rotate::DType;
using cvcuda::rotate::Interp;

template <typename T>
static int count_mismatch_tol(const std::vector<T>& gpu, const std::vector<T>& cpu, double tol) {
  int m = 0;
  for (size_t i = 0; i < gpu.size(); ++i)
    if (std::fabs((double)gpu[i] - (double)cpu[i]) > tol) ++m;
  return m;
}

// One case. tol=0 (bit-exact) by construction — see file header.
template <typename T>
static bool run_case(Rotate& op, queue& q, int N, int Hin, int Win, int Hout, int Wout, int C,
                     double angle, double sx, double sy, Interp interp, DType dt,
                     const std::string& tag) {
  size_t nin  = (size_t)N * Hin * Win * C;
  size_t nout = (size_t)N * Hout * Wout * C;
  std::vector<T> src(nin), gpu(nout), cpu(nout);

  // Integer-valued data so axis-aligned (0/180 deg) cases are exact copies.
  std::mt19937 rng(20240630 + C * 7 + (int)angle + (int)sx + (int)sy + (int)dt);
  if constexpr (std::is_signed_v<T>) {
    std::uniform_int_distribution<int> d(-8, 8);
    for (auto& v : src) v = (T)d(rng);
  } else if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<int> d(10, 200);
    for (auto& v : src) v = (T)d(rng);
  } else {
    std::uniform_int_distribution<int> d(10, 200);  // integer-valued floats -> exact
    for (auto& v : src) v = (T)d(rng);
  }

  T* d_in  = malloc_device<T>(nin, q);
  T* d_out = malloc_device<T>(nout, q);
  q.memcpy(d_in, src.data(), nin * sizeof(T)).wait();

  bool ok = op.forward(d_in, d_out, N, Hin, Win, Hout, Wout, C, angle, sx, sy, interp, dt, &q);
  if (!ok) {
    std::cout << "  [" << tag << "] forward returned false (unexpected)\n";
    free(d_in, q); free(d_out, q);
    return false;
  }
  q.wait();
  q.memcpy(gpu.data(), d_out, nout * sizeof(T)).wait();

  rotate_gold::gold_rotate<T>(cpu.data(), src.data(), N, Hin, Win, Hout, Wout, C, angle, sx, sy,
                              interp);
  int m = count_mismatch_tol<T>(gpu, cpu, 0.0);
  std::cout << "  [" << tag << "] mismatch=" << m << (m ? " FAIL" : " PASS") << "\n";
  free(d_in, q); free(d_out, q);
  return m == 0;
}

template <typename T>
static bool run_dtype(Rotate& op, queue& q, const std::string& name, DType dt) {
  bool all = true;
  // Small image, Hin!=Hout to exercise size mismatch. Shift = canvas center.
  int N = 2, Hin = 6, Win = 7, Hout = 8, Wout = 9;
  double sx = (Wout - 1) / 2.0, sy = (Hout - 1) / 2.0;

  // NEAREST: bit-exact for any angle (3 angles + 3 channel counts).
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 0.0,   sx, sy, Interp::NEAREST, dt, name+"/NEAREST 0deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 90.0,  sx, sy, Interp::NEAREST, dt, name+"/NEAREST 90deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 45.0,  sx, sy, Interp::NEAREST, dt, name+"/NEAREST 45deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 1, 30.0,  sx, sy, Interp::NEAREST, dt, name+"/NEAREST 30deg C1");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 4, 200.0, sx, sy, Interp::NEAREST, dt, name+"/NEAREST 200deg C4");

  // LINEAR: bit-exact at 0 deg (identity) and 180 deg (c=-1,s=0 integer map).
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 0.0,   0.0, 0.0, Interp::LINEAR, dt, name+"/LINEAR 0deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 180.0, (double)(Win-1), (double)(Hin-1), Interp::LINEAR, dt, name+"/LINEAR 180deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 4, 0.0,   0.0, 0.0, Interp::LINEAR, dt, name+"/LINEAR 0deg C4");

  // CUBIC: bit-exact at 0 deg (identity, coeffs [0,1,0,0]).
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 3, 0.0,   0.0, 0.0, Interp::CUBIC, dt, name+"/CUBIC 0deg C3");
  all &= run_case<T>(op, q, N, Hin, Win, Hout, Wout, 1, 0.0,   0.0, 0.0, Interp::CUBIC, dt, name+"/CUBIC 0deg C1");
  return all;
}

int main() {
  auto op = cvcuda::rotate::create_rotate();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

  bool all = true;
  std::cout << "--- U8 ---\n";  all &= run_dtype<uint8_t>(*op, q,  "U8",  DType::U8);
  std::cout << "--- U16 ---\n"; all &= run_dtype<uint16_t>(*op, q, "U16", DType::U16);
  std::cout << "--- S16 ---\n"; all &= run_dtype<int16_t>(*op, q,  "S16", DType::S16);
  std::cout << "--- F32 ---\n"; all &= run_dtype<float>(*op, q,    "F32", DType::F32);

  // ---- negative tests ----
  std::cout << "--- negative ---\n";
  uint8_t* dummy = malloc_device<uint8_t>(64, q);
  q.wait();
  // C==2 must be rejected (CV-CUDA dispatch table null slot).
  bool r1 = op->forward(dummy, dummy, 1, 4, 4, 4, 4, 2, 0.0, 0.0, 0.0, Interp::NEAREST, DType::U8, &q);
  std::cout << "  [C==2] rejected=" << (!r1) << (r1 ? " FAIL" : " PASS") << "\n";
  all &= !r1;
  // C>4 must be rejected.
  bool r2 = op->forward(dummy, dummy, 1, 4, 4, 4, 4, 5, 0.0, 0.0, 0.0, Interp::NEAREST, DType::U8, &q);
  std::cout << "  [C==5] rejected=" << (!r2) << (r2 ? " FAIL" : " PASS") << "\n";
  all &= !r2;
  free(dummy, q);

  std::cout << (all ? "=== ALL TESTS PASSED ===" : "=== TESTS FAILED ===") << "\n";
  return all ? 0 : 1;
}
