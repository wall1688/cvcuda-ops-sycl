// CPU gold reference for NMS, ported from CV-CUDA's GoldIoU/GoldArea/GoldNMS
// (tests/cvcuda/system/TestOpNonMaximumSuppression.cpp). Shared by the
// correctness / real / profile tests. Operates on the same flat row-major
// layout as the SYCL kernel: bboxes [S*N*4] short, scores [S*N] float,
// out [S*N] uint8, box (b,j) at b*N + j.
//
// NOTE: the kernel and this gold are bit-identical EXCEPT the final IoU
// division: CPU float div is IEEE-correctly-rounded, Intel GPU div is
// reciprocal-approximation (see intel-gpu-fp-quirk). They agree whenever no
// box pair has IoU exactly on the threshold boundary.

#ifndef __NMS_GOLD_HPP__
#define __NMS_GOLD_HPP__

#include <algorithm>
#include <cstdint>

namespace nms_gold {

struct Box {
  short x, y, w, h;  // top-left + width/height (w=source .z, h=source .w)
};

inline float area(const Box& b) { return (float)b.w * (float)b.h; }

inline float iou(const Box& a, const Box& b) {
  int x_inter_left   = std::max(a.x, b.x);
  int y_inter_top    = std::max(a.y, b.y);
  int x_inter_right  = std::min(a.x + a.w, b.x + b.w);
  int y_inter_bottom = std::min(a.y + a.h, b.y + b.h);
  int width_inter    = x_inter_right - x_inter_left;
  int height_inter   = y_inter_bottom - y_inter_top;
  float inter_area   = (float)width_inter * (float)height_inter;
  if (width_inter > 0 && height_inter > 0) {
    float union_area = area(a) + area(b) - inter_area;
    if (union_area > 0.f) return inter_area / union_area;
  }
  return 0.f;
}

inline Box load_box(const short* bboxes, int b, int j, int num_bboxes) {
  const short* p = bboxes + (b * num_bboxes + j) * 4;
  return Box{p[0], p[1], p[2], p[3]};
}

inline void gold_nms(const short* bboxes, const float* scores, uint8_t* out, int num_samples,
                     int num_bboxes, float score_threshold, float iou_threshold) {
  for (int b = 0; b < num_samples; ++b) {
    for (int j = 0; j < num_bboxes; ++j) {
      float score_x   = scores[b * num_bboxes + j];
      uint8_t& dst    = out[b * num_bboxes + j];
      if (score_x < score_threshold) {
        dst = 0;
        continue;
      }
      Box src_x     = load_box(bboxes, b, j, num_bboxes);
      bool discard  = false;
      for (int y = 0; y < num_bboxes; ++y) {
        if (j == y) continue;
        Box src_y = load_box(bboxes, b, y, num_bboxes);
        if (iou(src_x, src_y) > iou_threshold) {
          float score_y = scores[b * num_bboxes + y];
          if (score_x < score_y || (score_x == score_y && area(src_x) < area(src_y))) {
            discard = true;
            break;
          }
        }
      }
      dst = discard ? 0 : 1;
    }
  }
}

}  // namespace nms_gold

#endif  // __NMS_GOLD_HPP__
