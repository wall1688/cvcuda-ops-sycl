// Per-kernel event-profiling test for the SYCL Rotate port (VOX_PROFILE=1 at
// runtime). Reports device kernel time (best/avg), wall time, launch+wait
// overhead, and throughput. Single kernel per forward(). Default F32 LINEAR.
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_rotate_profile.cpp \
//   -o test/test_rotate_profile
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu VOX_PROFILE=1 \
//        ./test/test_rotate_profile [N] [S] [C] [runs]

#include "../rotate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;
using cvcuda::rotate::Rotate;
using cvcuda::rotate::DType;
using cvcuda::rotate::Interp;

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 4;
  int S = argc > 2 ? atoi(argv[2]) : 128;
  int C = argc > 3 ? atoi(argv[3]) : 3;
  int runs = argc > 4 ? atoi(argv[4]) : 20;

  auto op = cvcuda::rotate::create_rotate();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "N=" << N << " S=" << S << " C=" << C << " runs=" << runs << "\n";

  uint64_t probe = 0;
  if (!op->last_profile(probe)) {
    std::cout << "Profiling NOT enabled — rerun with VOX_PROFILE=1\n";
    return 1;
  }

  size_t nin = (size_t)N * S * S * C;
  std::vector<float> src(nin);
  std::mt19937 rng(11111);
  std::uniform_real_distribution<float> d(0.f, 1.f);
  for (auto& v : src) v = d(rng);

  float* d_in  = malloc_device<float>(nin, q);
  float* d_out = malloc_device<float>(nin, q);
  q.memcpy(d_in, src.data(), nin * sizeof(float)).wait();
  double sx = (S - 1) / 2.0, sy = (S - 1) / 2.0, angle = 23.5;

  auto run = [&](double& wall_ms, uint64_t& kernel_ns) {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_in, d_out, N, S, S, S, S, C, angle, sx, sy, Interp::LINEAR, DType::F32, &q);
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    op->last_profile(kernel_ns);
  };

  // warmup
  for (int i = 0; i < 3; ++i) {
    double w; uint64_t k;
    run(w, k);
  }

  double best_wall = 1e9, sum_wall = 0;
  uint64_t best_kernel = UINT64_MAX, sum_kernel = 0;
  for (int i = 0; i < runs; ++i) {
    double w; uint64_t k;
    run(w, k);
    best_wall = std::min(best_wall, w); sum_wall += w;
    best_kernel = std::min(best_kernel, k); sum_kernel += k;
  }
  double avg_wall = sum_wall / runs;
  double avg_kernel = (double)sum_kernel / runs / 1e6;       // ns -> ms
  double best_kernel_ms = (double)best_kernel / 1e6;
  double overhead = best_wall - best_kernel_ms;
  double pixels = (double)N * S * S;
  double elems = (double)nin;

  std::cout << "----- Rotate per-kernel (best) -----\n";
  std::cout << "rotate_kernel      best=" << best_kernel_ms << "ms avg=" << avg_kernel
            << "ms (100%)\n";
  std::cout << "kernel sum         =" << best_kernel_ms << "ms\n";
  std::cout << "wall               best=" << best_wall << "ms avg=" << avg_wall << "ms\n";
  std::cout << "overhead           =" << overhead << "ms (launch + wait + coeff memcpy)\n";
  std::cout << "throughput         kernel=" << (pixels / best_kernel_ms / 1e3) << " Mpix/s"
            << "  wall=" << (pixels / best_wall / 1e3) << " Mpix/s"
            << "  (" << (elems / best_kernel_ms / 1e3) << " Melem/s kernel)\n";

  free(d_in, q); free(d_out, q);
  return 0;
}
