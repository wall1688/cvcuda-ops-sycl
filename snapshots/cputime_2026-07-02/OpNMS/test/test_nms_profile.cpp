// Per-kernel profiling test for the SYCL NMS port. Uses SYCL event profiling
// (VOX_PROFILE=1 at construction) to get the single NMS kernel's device time,
// plus wall time, overhead and throughput. Mirrors the voxelizer profile test.
//
// Build: bash build.sh   (header-only kernel — only this test TU is compiled)
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu VOX_PROFILE=1 \
//          ./test/test_nms_profile [N=5000] [S=4] [runs=20]

#include "../nms.hpp"
#include "nms_gold.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

using cvcuda::nms::NonMaximumSuppression;

int main(int argc, char** argv) {
  int N    = (argc > 1) ? std::atoi(argv[1]) : 5000;
  int S    = (argc > 2) ? std::atoi(argv[2]) : 4;
  int runs = (argc > 3) ? std::atoi(argv[3]) : 20;
  const float SC = 0.3f, IOU = 0.45f;

  auto op = cvcuda::nms::create_nms();
  sycl::queue* q = static_cast<sycl::queue*>(op->internal_stream());
  std::cout << "Running on: " << q->get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "S=" << S << " N=" << N << " runs=" << runs << "\n";

  uint64_t nms_ns_dummy = 0;
  if (!op->last_profile(nms_ns_dummy)) {
    std::cerr << "Profiling NOT enabled — rerun with VOX_PROFILE=1\n";
    return 1;
  }

  size_t n_box = (size_t)S * N;
  std::vector<short> bb(n_box * 4);
  std::vector<float> sc(n_box);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> pos(0, 128), wh(8, 64);
  std::uniform_real_distribution<float> scr(0.0f, 1.0f);
  for (size_t i = 0; i < n_box; ++i) {
    bb[i * 4 + 0] = (short)pos(rng);
    bb[i * 4 + 1] = (short)pos(rng);
    bb[i * 4 + 2] = (short)wh(rng);
    bb[i * 4 + 3] = (short)wh(rng);
    sc[i] = scr(rng);
  }

  short*   d_bb  = sycl::malloc_device<short>(n_box * 4, *q);
  float*   d_sc  = sycl::malloc_device<float>(n_box, *q);
  uint8_t* d_out = sycl::malloc_device<uint8_t>(n_box, *q);
  q->memcpy(d_bb, bb.data(), n_box * 4 * sizeof(short));
  q->memcpy(d_sc, sc.data(), n_box * sizeof(float));
  q->wait();

  auto run = [&](double& wall_ms, uint64_t& kernel_ns) {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_bb, d_sc, d_out, S, N, SC, IOU);
    q->wait();
    auto t1 = std::chrono::steady_clock::now();
    wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    op->last_profile(kernel_ns);
  };

  for (int w = 0; w < 3; ++w) { double wm; uint64_t kn; run(wm, kn); }  // warmup

  double best_wall = 1e9, sum_wall = 0;
  uint64_t best_kernel = UINT64_MAX, sum_kernel = 0;
  for (int r = 0; r < runs; ++r) {
    double wm; uint64_t kn;
    run(wm, kn);
    best_wall   = std::min(best_wall, wm);
    sum_wall   += wm;
    best_kernel = std::min(best_kernel, kn);
    sum_kernel += kn;
  }

  double best_kernel_ms = best_kernel / 1e6;
  double avg_kernel_ms  = (sum_kernel / runs) / 1e6;
  double overhead_ms    = best_wall - best_kernel_ms;
  double mbox_kern      = (double)n_box / (best_kernel_ms) / 1e3;       // Mbox/s kernel
  double mbox_wall      = (double)n_box / (best_wall) / 1e3;            // Mbox/s wall

  std::cout << "----- NMS per-kernel (best) -----\n";
  std::cout << "nms_kernel   best=" << best_kernel_ms << "ms avg=" << avg_kernel_ms << "ms (100%)\n";
  std::cout << "kernel sum   =" << best_kernel_ms << "ms\n";
  std::cout << "wall         best=" << best_wall << "ms avg=" << (sum_wall / runs) << "ms\n";
  std::cout << "overhead     =" << overhead_ms << "ms (launch + wait)\n";
  std::cout << "throughput   kernel=" << mbox_kern << " Mbox/s  wall=" << mbox_wall << " Mbox/s\n";

  // Optional CPU gold reference timing for GPU-vs-CPU speedup (env-gated).
  // gold_nms is O(S*N^2); at the default N=5000 a single run can take seconds,
  // so we time exactly 1 run (no best-of-N). Lower N via CLI for a faster turn.
  const char* cputime_env = std::getenv("VOX_CPUTIME");
  bool want_cputime = cputime_env && cputime_env[0] == '1';
  if (want_cputime) {
    std::vector<uint8_t> cpu_out(n_box);
    auto t0 = std::chrono::steady_clock::now();
    nms_gold::gold_nms(bb.data(), sc.data(), cpu_out.data(), S, N, SC, IOU);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double speedup = cpu_ms / best_kernel_ms;
    std::cout << "[CPU ref time: " << cpu_ms << " ms, single-thread, 1 run]  speedup = "
              << speedup << "x  (gpu best_kernel=" << best_kernel_ms << "ms)\n";
  } else {
    std::cout << "(set VOX_CPUTIME=1 for CPU-vs-GPU speedup)\n";
  }

  sycl::free(d_bb, *q);
  sycl::free(d_sc, *q);
  sycl::free(d_out, *q);
  return 0;
}
