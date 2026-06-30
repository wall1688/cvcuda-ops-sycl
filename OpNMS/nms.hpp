// Ported from CV-CUDA 0.16.0 src/cvcuda/priv/OpNonMaximumSuppression.cu
// SYCL port of the Non-Maximum-Suppression operator. Algorithm/semantics unchanged:
// all-pairs IoU mask NMS, short4 bboxes (x,y,w,h), float scores, uint8 keep-mask.
//
// Self-contained header-only kernel (matches the cvcuda-ops-sycl convention:
// the test TUs include this header directly; build.sh only compiles the tests).
// The original CV-CUDA code is Apache-2.0 (NVIDIA); this port preserves the
// algorithm 1:1 and is also Apache-2.0.

#ifndef CVCUDA_OPS_NMS_HPP
#define CVCUDA_OPS_NMS_HPP

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace cvcuda {
namespace nms {

// Single-kernel, embarrassingly-parallel NMS producing a per-box keep-mask
// (1 = keep, 0 = discard). Faithful to CV-CUDA's mask-output semantics — NOT a
// sorted index list like torchvision NMS.
//
// Layout (row-major, contiguous — matches CV-CUDA Tensor2DWrap {bbox,batch} ->
// (dim0=batch, dim1=bbox) mapping):
//   bboxes : short*  [num_samples * num_bboxes * 4]  (x, y, w, h) per box
//   scores : float*  [num_samples * num_bboxes]
//   out    : uint8_t*[num_samples * num_bboxes]
// Box (b, j) lives at b*num_bboxes + j (bboxes stride 4 shorts).

// --------------------------------------------------------------------------- //
// device-side helpers (compiled for device when called from parallel_for)
// --------------------------------------------------------------------------- //

// Matches CUDA short4 layout: .x/.y = top-left, .z = width, .w = height.
struct short4 {
  short x, y, z, w;
};

static inline float compute_area(const short4& bbox) { return bbox.z * bbox.w; }

// IoU over integer box coords (only the final division is float). Ported verbatim
// from OpNonMaximumSuppression.cu:38-63 (ComputeArea / ComputeIoU).
static inline float compute_iou(const short4& box1, const short4& box2) {
  int x_inter_left   = sycl::max((int)box1.x, (int)box2.x);
  int y_inter_top    = sycl::max((int)box1.y, (int)box2.y);
  int x_inter_right  = sycl::min((int)box1.x + box1.z, (int)box2.x + box2.z);
  int y_inter_bottom = sycl::min((int)box1.y + box1.w, (int)box2.y + box2.w);
  int width_inter    = x_inter_right - x_inter_left;
  int height_inter   = y_inter_bottom - y_inter_top;
  float inter_area   = width_inter * height_inter;
  float iou          = 0.f;
  if (width_inter > 0.f && height_inter > 0.f) {
    float union_area = compute_area(box1) + compute_area(box2) - inter_area;
    if (union_area > 0.f) {
      iou = inter_area / union_area;
    }
  }
  return iou;
}

static inline void load_box(const short* bboxes, int b, int j, int num_bboxes, short4& out) {
  const short* p = bboxes + (b * num_bboxes + j) * 4;
  out.x = p[0];
  out.y = p[1];
  out.z = p[2];
  out.w = p[3];
}

// One thread per (batch, box). Faithful 1:1 port of the CUDA kernel
// (OpNonMaximumSuppression.cu:66-115): the CUDA grid used blockIdx.z = batch and
// threadIdx.x = bbox; here we flatten to a 1D linear index and decode.
static inline void nms_kernel(size_t idx, size_t total, const short* bboxes, const float* scores,
                              uint8_t* out, int num_samples, int num_bboxes, float score_threshold,
                              float iou_threshold) {
  if (idx >= total) return;

  int b = (int)(idx / (size_t)num_bboxes);
  int j = (int)(idx % (size_t)num_bboxes);

  float score_x = scores[b * num_bboxes + j];
  uint8_t& dst  = out[b * num_bboxes + j];

  if (score_x < score_threshold) {
    dst = 0;
    return;
  }

  short4 src_x;
  load_box(bboxes, b, j, num_bboxes, src_x);
  bool discard = false;

  for (int y = 0; y < num_bboxes; ++y) {
    if (j == y) continue;

    short4 src_y;
    load_box(bboxes, b, y, num_bboxes, src_y);

    if (compute_iou(src_x, src_y) > iou_threshold) {
      float score_y = scores[b * num_bboxes + y];
      if (score_x < score_y || (score_x == score_y && compute_area(src_x) < compute_area(src_y))) {
        discard = true;
        break;
      }
    }
  }

  dst = discard ? 0 : 1;
}

// --------------------------------------------------------------------------- //
// NonMaximumSuppression interface + implementation
// --------------------------------------------------------------------------- //

class NonMaximumSuppression {
 public:
  virtual ~NonMaximumSuppression() = default;

  // All pointers must be device USM allocated in the same context as the
  // implementation's internal queue (use internal_stream() to get it).
  // stream is a sycl::queue* (nullptr => use the implementation's internal queue).
  // iou_threshold must be in (0, 1].
  virtual void forward(const short* bboxes, const float* scores, uint8_t* out,
                       int num_samples, int num_bboxes, float score_threshold,
                       float iou_threshold, void* stream = nullptr) = 0;

  // Per-kernel device profiling from the most recent forward() call.
  // Returns true if profiling was enabled (VOX_PROFILE env var set at
  // construction) and a run has completed; nms_ns is filled with the single
  // kernel's device execution time in nanoseconds. Returns false otherwise.
  // The caller must ensure the queue is idle (q.wait()) before calling.
  virtual bool last_profile(uint64_t& nms_ns) = 0;

  // Returns the implementation's internal sycl::queue* so callers can allocate
  // USM in the SAME context (cast the void* to sycl::queue*).
  virtual void* internal_stream() = 0;
};

// Build the internal GPU queue. Profiling is opt-in via the VOX_PROFILE env var
// (same convention as the voxelizer): when set, enable_profiling is added so
// last_profile() can report the kernel's device time. When unset the queue is
// plain (no profiling overhead).
static inline sycl::queue make_internal_queue() {
  sycl::device dev{sycl::gpu_selector_v};
  if (std::getenv("VOX_PROFILE")) {
    return sycl::queue{dev, sycl::property_list{sycl::property::queue::in_order{},
                                                 sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{dev, sycl::property::queue::in_order{}};
}

class NMSImplement : public NonMaximumSuppression {
 public:
  NMSImplement() {}
  virtual ~NMSImplement() {}

  virtual void forward(const short* bboxes, const float* scores, uint8_t* out, int num_samples,
                       int num_bboxes, float score_threshold, float iou_threshold,
                       void* stream) override {
    if (!(iou_threshold > 0.f && iou_threshold <= 1.f))
      throw std::invalid_argument("iou_threshold must be in (0,1]");
    if (num_samples <= 0 || num_bboxes <= 0)
      throw std::invalid_argument("num_samples/num_bboxes must be > 0");

    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : queue_;

    size_t total = (size_t)num_samples * (size_t)num_bboxes;

    // SYCL device lambdas cannot implicitly capture `this`; copy args by value.
    const short* _bboxes = bboxes;
    const float* _scores = scores;
    uint8_t* _out         = out;
    int _num_samples      = num_samples;
    int _num_bboxes       = num_bboxes;
    float _score_thresh   = score_threshold;
    float _iou_thresh     = iou_threshold;

    ev_nms_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      nms_kernel(i.get(0), total, _bboxes, _scores, _out, _num_samples, _num_bboxes, _score_thresh,
                 _iou_thresh);
    });
  }

  virtual void* internal_stream() override { return &queue_; }

  virtual bool last_profile(uint64_t& nms_ns) override {
    if (!profiling_enabled_) return false;
    try {
      uint64_t s = ev_nms_.get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t en = ev_nms_.get_profiling_info<sycl::info::event_profiling::command_end>();
      nms_ns = en - s;
    } catch (...) {
      nms_ns = 0;
    }
    return true;
  }

 private:
  sycl::queue queue_{make_internal_queue()};
  bool profiling_enabled_{std::getenv("VOX_PROFILE") != nullptr};
  sycl::event ev_nms_;
};

inline std::shared_ptr<NonMaximumSuppression> create_nms() {
  return std::make_shared<NMSImplement>();
}

};  // namespace nms
};  // namespace cvcuda

#endif  // CVCUDA_OPS_NMS_HPP
