// Real-data correctness + performance test for the SYCL NMS port.
// Random integer boxes/scores (mimicking detector output), CPU gold compare with
// FP-boundary classification: any GPU/CPU mask divergence must be explainable by
// a box pair whose exact IoU sits on the threshold boundary (Intel GPU float div
// is reciprocal-approx, not IEEE-rounded — see intel-gpu-fp-quirk). Also reports
// best/avg wall ms.
//
// Build: bash build.sh   (header-only kernel — only this test TU is compiled)
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_nms_real [N=5000] [S=4] [runs=10]

#include "../nms.hpp"
#include "nms_gold.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

using cvcuda::nms::NonMaximumSuppression;
using nms_gold::Box;

// Exact rational IoU distance to threshold for a box vs all others in its batch.
// Returns the smallest |iou_exact(j,y) - iou_threshold| over y != j (1.0 if none).
static float min_iou_dist_to_thresh(const std::vector<short>& bboxes, int b, int j, int N,
                                    float iou_threshold) {
  Box X = nms_gold::load_box(bboxes.data(), b, j, N);
  float best = 1.0f;
  for (int y = 0; y < N; ++y) {
    if (y == j) continue;
    Box Y = nms_gold::load_box(bboxes.data(), b, y, N);
    // replicate exact integer-then-divide IoU
    int xl = std::max(X.x, Y.x), yt = std::max(X.y, Y.y);
    int xr = std::min(X.x + X.w, Y.x + Y.w), yb = std::min(X.y + X.h, Y.y + Y.h);
    int wi = xr - xl, hi = yb - yt;
    if (wi > 0 && hi > 0) {
      float inter = (float)wi * (float)hi;
      float uni   = nms_gold::area(X) + nms_gold::area(Y) - inter;
      if (uni > 0.f) {
        float v = inter / uni;
        best = std::min(best, std::fabs(v - iou_threshold));
      }
    }
  }
  return best;
}

int main(int argc, char** argv) {
  int N    = (argc > 1) ? std::atoi(argv[1]) : 5000;
  int S    = (argc > 2) ? std::atoi(argv[2]) : 4;
  int runs = (argc > 3) ? std::atoi(argv[3]) : 10;
  const float SC = 0.3f, IOU = 0.45f;  // 0.45 avoids clean 1/2 rational collisions

  auto op = cvcuda::nms::create_nms();
  sycl::queue* q = static_cast<sycl::queue*>(op->internal_stream());
  std::cout << "Running on: " << q->get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "S=" << S << " N=" << N << " runs=" << runs << " score_thresh=" << SC
            << " iou_thresh=" << IOU << "\n";

  size_t n_box = (size_t)S * N;
  std::vector<short> bb(n_box * 4);
  std::vector<float> sc(n_box);

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> pos(0, 128);
  std::uniform_int_distribution<int> wh(8, 64);
  std::uniform_real_distribution<float> scr(0.0f, 1.0f);
  for (size_t i = 0; i < n_box; ++i) {
    bb[i * 4 + 0] = (short)pos(rng);
    bb[i * 4 + 1] = (short)pos(rng);
    bb[i * 4 + 2] = (short)wh(rng);
    bb[i * 4 + 3] = (short)wh(rng);
    sc[i] = scr(rng);
  }

  // device buffers
  short*   d_bb  = sycl::malloc_device<short>(n_box * 4, *q);
  float*   d_sc  = sycl::malloc_device<float>(n_box, *q);
  uint8_t* d_out = sycl::malloc_device<uint8_t>(n_box, *q);
  q->memcpy(d_bb, bb.data(), n_box * 4 * sizeof(short));
  q->memcpy(d_sc, sc.data(), n_box * sizeof(float));
  q->wait();

  // ---- correctness ----
  std::vector<uint8_t> gpu_out(n_box), cpu_out(n_box);
  op->forward(d_bb, d_sc, d_out, S, N, SC, IOU);
  q->wait();
  q->memcpy(gpu_out.data(), d_out, n_box * sizeof(uint8_t));
  q->wait();
  nms_gold::gold_nms(bb.data(), sc.data(), cpu_out.data(), S, N, SC, IOU);

  int mismatch = 0, boundary = 0;
  const float BOUND_TOL = 1e-3f;
  for (int b = 0; b < S; ++b) {
    for (int j = 0; j < N; ++j) {
      size_t i = (size_t)b * N + j;
      if (gpu_out[i] != cpu_out[i]) {
        ++mismatch;
        float d = min_iou_dist_to_thresh(bb, b, j, N, IOU);
        if (d < BOUND_TOL) ++boundary;
      }
    }
  }
  int kept = 0;
  for (uint8_t v : cpu_out) kept += (v != 0);
  double match = 100.0 * (1.0 - (double)mismatch / (double)n_box);
  std::cout << "CPU kept=" << kept << " / " << n_box << "\n";
  std::cout << "[XYZ] mismatch=" << mismatch << " match=" << match << "%";
  if (mismatch) std::cout << " of which IoU-boundary(±" << BOUND_TOL << ")=" << boundary
                          << " (" << (100.0 * boundary / mismatch) << "%)";
  std::cout << "\n";

  // ---- perf (wall) ----
  auto bench = [&]() {
    auto t0 = std::chrono::steady_clock::now();
    op->forward(d_bb, d_sc, d_out, S, N, SC, IOU);
    q->wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  for (int w = 0; w < 3; ++w) bench();  // warmup
  double best = 1e9, sum = 0;
  for (int r = 0; r < runs; ++r) {
    double ms = bench();
    best = std::min(best, ms);
    sum += ms;
  }
  std::cout << "[perf] wall best=" << best << "ms avg=" << (sum / runs) << "ms"
            << " throughput=" << ((double)n_box / best / 1e3) << " Mbox/s\n";

  sycl::free(d_bb, *q);
  sycl::free(d_sc, *q);
  sycl::free(d_out, *q);

  bool pass = (mismatch == 0) || (boundary == mismatch);
  std::cout << (pass ? "=== REAL-DATA TEST PASSED ==="
                     : "=== REAL-DATA TEST FAILED (non-boundary divergence) ===")
            << "\n";
  return pass ? 0 : 1;
}
