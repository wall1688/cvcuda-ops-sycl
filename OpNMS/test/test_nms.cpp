// Correctness test for the SYCL NMS port.
// Hand-picked integer boxes whose pairwise IoUs are clearly separated from the
// threshold (0.5) — so the Intel GPU float-division quirk cannot flip any
// comparison, and the GPU mask matches the CPU gold bit-for-bit.
//
// Build: bash build.sh   (header-only kernel — only this test TU is compiled)
// Run:   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_nms

#include "../nms.hpp"
#include "nms_gold.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using cvcuda::nms::NonMaximumSuppression;

static void run_case(NonMaximumSuppression& op, sycl::queue& q, const std::vector<short>& bboxes,
                     const std::vector<float>& scores, int num_samples, int num_bboxes,
                     float score_thresh, float iou_thresh, const char* name) {
  size_t n_box = (size_t)num_samples * num_bboxes;
  std::vector<uint8_t> gpu_out(n_box, 0xff), cpu_out(n_box, 0xff);

  short*   d_bb  = sycl::malloc_device<short>(n_box * 4, q);
  float*   d_sc  = sycl::malloc_device<float>(n_box, q);
  uint8_t* d_out = sycl::malloc_device<uint8_t>(n_box, q);
  q.memcpy(d_bb, bboxes.data(), n_box * 4 * sizeof(short));
  q.memcpy(d_sc, scores.data(), n_box * sizeof(float));
  q.wait();

  op.forward(d_bb, d_sc, d_out, num_samples, num_bboxes, score_thresh, iou_thresh);
  q.wait();
  q.memcpy(gpu_out.data(), d_out, n_box * sizeof(uint8_t));
  q.wait();

  nms_gold::gold_nms(bboxes.data(), scores.data(), cpu_out.data(), num_samples, num_bboxes,
                     score_thresh, iou_thresh);

  int mismatch = 0, first = -1;
  for (size_t i = 0; i < n_box; ++i) {
    if (gpu_out[i] != cpu_out[i]) {
      ++mismatch;
      if (first < 0) first = (int)i;
    }
  }

  int kept = 0;
  for (uint8_t v : cpu_out) kept += (v != 0);

  std::cout << "[" << name << "] S=" << num_samples << " N=" << num_bboxes << " kept=" << kept
            << " mismatch=" << mismatch;
  if (mismatch) std::cout << " (first at idx " << first << ")";
  std::cout << (mismatch ? "  FAIL" : "  PASS") << "\n";

  sycl::free(d_bb, q);
  sycl::free(d_sc, q);
  sycl::free(d_out, q);

  if (mismatch) std::exit(1);
}

int main() {
  auto op = cvcuda::nms::create_nms();
  sycl::queue* q = static_cast<sycl::queue*>(op->internal_stream());
  std::cout << "Running on: "
            << q->get_device().get_info<sycl::info::device::name>() << "\n";

  const float SC = 0.5f, IOU = 0.5f;

  // ---- Case 1: mixed scenarios, single batch ----
  // Layout per box: x, y, w, h (short).
  //  b0: score below threshold          -> discard (mask 0)
  //  b1,b2: IoU~0.818 (>0.5), b1 higher score -> b2 discarded
  //  b3,b4: disjoint (IoU 0)            -> both kept
  //  b5,b6: IoU~0.818, equal score + equal area -> tie: BOTH kept
  //  b7,b8: IoU~0.694 (>0.5), equal score, b8 bigger area -> b7 discarded
  //  b9,b10: IoU~0.333 (<0.5)           -> both kept
  {
    int S = 1, N = 11;
    std::vector<short> bb(S * N * 4);
    std::vector<float> sc(S * N);
    auto put = [&](int j, short x, short y, short w, short h, float s) {
      bb[(j) * 4 + 0] = x; bb[(j) * 4 + 1] = y; bb[(j) * 4 + 2] = w; bb[(j) * 4 + 3] = h;
      sc[j] = s;
    };
    put(0,  0, 0, 10, 10, 0.20f);  // below score thresh
    put(1,  0, 0, 10, 10, 0.90f);  // overlaps b2, higher score -> kept
    put(2,  1, 0, 10, 10, 0.80f);  // overlaps b1, lower score  -> discarded
    put(3,  0, 0, 10, 10, 0.70f);  // disjoint from b4
    put(4, 50, 50, 10, 10, 0.70f); // disjoint from b3
    put(5,  0, 0, 10, 10, 0.60f);  // overlaps b6, equal score+area -> kept (tie)
    put(6,  1, 0, 10, 10, 0.60f);  // overlaps b5, equal score+area -> kept (tie)
    put(7,  0, 0, 10, 10, 0.55f);  // overlaps b8, equal score, smaller area -> discarded
    put(8,  0, 0, 12, 12, 0.55f);  // overlaps b7, bigger area -> kept
    put(9,  0, 0, 10, 10, 0.50f);  // IoU 1/3 with b10 (<0.5) -> kept
    put(10, 5, 0, 10, 10, 0.50f);  // IoU 1/3 with b9 (<0.5) -> kept
    run_case(*op, *q, bb, sc, S, N, SC, IOU, "mixed");
  }

  // ---- Case 2: two batches, identical content (batch independence) ----
  {
    int S = 2, N = 4;
    std::vector<short> bb(S * N * 4);
    std::vector<float> sc(S * N);
    for (int b = 0; b < S; ++b) {
      auto put = [&](int j, short x, short y, short w, short h, float s) {
        int i = b * N + j;
        bb[i * 4 + 0] = x; bb[i * 4 + 1] = y; bb[i * 4 + 2] = w; bb[i * 4 + 3] = h;
        sc[i] = s;
      };
      put(0, 0, 0, 10, 10, 0.9f);
      put(1, 1, 0, 10, 10, 0.8f);  // overlaps 0, lower score -> discarded
      put(2, 60, 60, 10, 10, 0.7f);
      put(3, 0, 0, 10, 10, 0.1f);  // below threshold
    }
    run_case(*op, *q, bb, sc, S, N, SC, IOU, "batch2");
  }

  // ---- Case 3: score threshold edge — all below -> all 0 ----
  {
    int S = 1, N = 3;
    std::vector<short> bb(S * N * 4, 0);
    std::vector<float> sc(S * N, 0.1f);
    run_case(*op, *q, bb, sc, S, N, 0.5f, 0.5f, "all_below");
  }

  // ---- Case 4: single box per batch -> always kept ----
  {
    int S = 3, N = 1;
    std::vector<short> bb(S * N * 4, 1);
    std::vector<float> sc(S * N, 0.9f);
    run_case(*op, *q, bb, sc, S, N, SC, IOU, "single");
  }

  std::cout << "=== ALL TESTS PASSED ===\n";
  return 0;
}
