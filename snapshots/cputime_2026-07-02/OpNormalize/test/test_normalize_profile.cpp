// Per-kernel event-profiling test for the SYCL Normalize port (VOX_PROFILE=1 at
// runtime). Reports device kernel time (best/avg), wall time, launch+wait
// overhead, and throughput. Single kernel per forward().
//
// Build: icpx -fsycl -std=c++17 -O2 -g -I<root> \
//   test/test_normalize_profile.cpp \
//   -o test/test_normalize_profile
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu VOX_PROFILE=1 \
//        ./test/test_normalize_profile [N] [H] [W] [C] [runs]

#include "../normalize.hpp"
#include "normalize_gold.hpp"

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
using cvcuda::normalize::Normalize;
using cvcuda::normalize::DType;
using cvcuda::normalize::Shape;

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 64;
  int H = argc > 2 ? atoi(argv[2]) : 128;
  int W = argc > 3 ? atoi(argv[3]) : 128;
  int C = argc > 4 ? atoi(argv[4]) : 4;
  int runs = argc > 5 ? atoi(argv[5]) : 20;

  auto op = cvcuda::normalize::create_normalize();
  queue& q = *static_cast<queue*>(op->internal_stream());
  std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "N=" << N << " H=" << H << " W=" << W << " C=" << C << " runs=" << runs << "\n";

  uint64_t probe = 0;
  if (!op->last_profile(probe)) {
    std::cout << "Profiling NOT enabled — rerun with VOX_PROFILE=1\n";
    return 1;
  }

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

  auto run = [&](double& wall_ms, uint64_t& kernel_ns) {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_in, d_base, d_scale, d_out, N, H, W, C, bS, sS, 1.f, 0.f, 0.f, DType::F32, 0, &q);
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
  double pixels = (double)N * H * W;
  double elems = (double)nin;

  std::cout << "----- Normalize per-kernel (best) -----\n";
  std::cout << "normalize_kernel  best=" << best_kernel_ms << "ms avg=" << avg_kernel
            << "ms (100%)\n";
  std::cout << "kernel sum        =" << best_kernel_ms << "ms\n";
  std::cout << "wall              best=" << best_wall << "ms avg=" << avg_wall << "ms\n";
  std::cout << "overhead          =" << overhead << "ms (launch + wait)\n";
  std::cout << "throughput        kernel=" << (pixels / best_kernel_ms / 1e3) << " Mpix/s"
            << "  wall=" << (pixels / best_wall / 1e3) << " Mpix/s"
            << "  (" << (elems / best_kernel_ms / 1e3) << " Melem/s kernel)\n";

  // Optional CPU gold reference timing for GPU-vs-CPU speedup (env-gated so the
  // default profile run stays fast). Mirrors the voxelizer [CPU ref time] line.
  const char* cputime_env = std::getenv("VOX_CPUTIME");
  bool want_cputime = cputime_env && cputime_env[0] == '1';
  if (want_cputime) {
    std::vector<float> cpu(nin);
    // plain mode (stddev=false), same params as the GPU run above.
    auto cpu_run = [&]() {
      normalize_gold::gold_normalize<float>(cpu.data(), src.data(), base.data(), scale.data(), N, H,
                                            W, C, bS, sS, 1.f, 0.f, 0.f, false);
    };
    cpu_run();  // warmup (also fills cpu)
    int cruns = 3;
    double best_cpu = 1e9;
    for (int i = 0; i < cruns; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      cpu_run();
      auto t1 = std::chrono::steady_clock::now();
      best_cpu = std::min(best_cpu, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double speedup = best_cpu / best_kernel_ms;
    std::cout << "[CPU ref time: " << best_cpu << " ms, single-thread, best-of-" << cruns
              << "]  speedup = " << speedup << "x  (gpu best_kernel=" << best_kernel_ms
              << "ms)\n";
  } else {
    std::cout << "(set VOX_CPUTIME=1 for CPU-vs-GPU speedup)\n";
  }

  free(d_in, q); free(d_out, q); free(d_base, q); free(d_scale, q);
  return 0;
}
