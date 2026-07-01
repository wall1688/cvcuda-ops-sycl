// Ported from CV-CUDA 0.16.0 src/cvcuda/priv/legacy/normalize.cu (tensor path).
// SYCL port of the Normalize operator. Algorithm/semantics unchanged:
//   out = SaturateCast<outT>((src - base) * mul * global_scale + shift)
//   mul = scale                          (plain mode)
//   mul = 1 / sqrt(scale^2 + epsilon)    (SCALE_IS_STDDEV mode)
// base/scale broadcast per-axis (each of N/H/W/C independently 1 or matching).
//
// Self-contained header-only kernel (matches the cvcuda-ops-sycl convention:
// the test TUs include this header directly; build.sh only compiles the tests).
// The original CV-CUDA code is Apache-2.0 (NVIDIA); this port preserves the
// algorithm 1:1 and is also Apache-2.0.

#ifndef CVCUDA_OPS_NORMALIZE_HPP
#define CVCUDA_OPS_NORMALIZE_HPP

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "normalize_helpers.hpp"

namespace cvcuda {
namespace normalize {



// Flag matching CV-CUDA's CVCUDA_NORMALIZE_SCALE_IS_STDDEV (bit 0).
constexpr uint32_t SCALE_IS_STDDEV = 1u;

// Data types CV-CUDA Normalize supports (U8/S8/U16/S16/S32/F32). F64/U32/F16
// are NOT supported by CV-CUDA Normalize -> rejected in forward().
enum class DType : int {
  U8 = 0,
  S8,
  U16,
  S16,
  S32,
  F32,
};

// 4-D shape for the per-axis broadcast check (each axis is 1 or matches input).
struct Shape {
  int N, H, W, C;
};

// Layout: NHWC contiguous. in/out are device USM in the same context as the
// internal queue (use internal_stream() to get it). Normalize requires
// in_dtype == out_dtype (CV-CUDA rejects mismatched dtypes), so a single dtype
// is passed. base/scale are always float* (CV-CUDA uses work_type =
// ConvertBaseTypeTo<float,input_type>). Channels C must be 1, 3, or 4 (CV-CUDA
// rejects C==2 and C>4). Each base/scale axis must be 1 or equal to the input's
// corresponding axis.
class Normalize {
 public:
  virtual ~Normalize() = default;

  // Returns true on success, false on invalid dtype/channels/shape/broadcast
  // (matching CV-CUDA INVALID_PARAMETER semantics). stream is sycl::queue*
  // (nullptr => internal queue). epsilon is ignored in plain mode.
  virtual bool forward(const void* in, const float* base, const float* scale, void* out,
                       int N, int H, int W, int C, Shape baseShape, Shape scaleShape,
                       float global_scale, float shift, float epsilon, DType dtype,
                       uint32_t flags = 0, void* stream = nullptr) = 0;

  // Per-kernel device profiling from the most recent forward() call.
  // Returns true if profiling enabled (VOX_PROFILE env at construction) and a
  // run completed; ns filled with the single kernel's device time. Caller must
  // q.wait() first.
  virtual bool last_profile(uint64_t& ns) = 0;

  virtual void* internal_stream() = 0;
};

inline std::shared_ptr<Normalize> create_normalize();

}  // namespace normalize
}  // namespace cvcuda

#endif  // __CVCUDA_OPS_NORMALIZE_HPP__



// --------------------------------------------------------------------------- //
// SaturateCast (device): float -> integral = round-to-nearest-even (sycl::rint,
// matches PTX cvt.rni) + clamp [TypeMin, TypeMax]; float -> identity (no round,
// no clamp — matches CV-CUDA's F32 SaturateCast path). Mirrors colorcvt's
// sat_cast_f but generalized to signed types.
// --------------------------------------------------------------------------- //
template <typename T>
static inline T sat_cast(float u) {
  if constexpr (std::is_integral_v<T>) {
    float r = sycl::rint(u);
    if (r < (float)TypeMin<T>()) r = (float)TypeMin<T>();
    if (r > (float)TypeMax<T>()) r = (float)TypeMax<T>();
    return (T)r;
  } else {
    return (T)u;
  }
}

// --------------------------------------------------------------------------- //
// device kernel: normalize. Faithful 1:1 port of normalizeKernel /
// normalizeInvStdDevKernel (normalize.cu:34-93). The CUDA grid used blockIdx.z
// = batch and (threadIdx.x, threadIdx.y) = (x,y); here we flatten to 1D and
// decode. base/scale broadcast per-axis (param_idx = shape==1 ? 0 : data_idx),
// matching CV-CUDA's base_size/scale_size int3 + channel==1 logic.
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void normalize_kernel(size_t idx, size_t total, const EltT* in, const float* base,
                                    const float* scale, EltT* out, int N, int H, int W, int C,
                                    Shape bS, Shape sS, float gscale, float shift, float eps,
                                    bool stddev) {
  if (idx >= total) return;
  int n   = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y   = rem / W;
  int x   = rem % W;

  // per-axis broadcast indices (1 => broadcast, use 0)
  int bn = bS.N == 1 ? 0 : n, by = bS.H == 1 ? 0 : y, bx = bS.W == 1 ? 0 : x;
  int sn = sS.N == 1 ? 0 : n, sy = sS.H == 1 ? 0 : y, sx = sS.W == 1 ? 0 : x;

  const EltT* sp = in + ((static_cast<size_t>(n) * H + y) * W + x) * C;
  EltT* dp        = out + ((static_cast<size_t>(n) * H + y) * W + x) * C;

  size_t base_off  = ((static_cast<size_t>(bn) * bS.H + by) * bS.W + bx) * bS.C;
  size_t scale_off = ((static_cast<size_t>(sn) * sS.H + sy) * sS.W + sx) * sS.C;

  for (int c = 0; c < C; ++c) {
    float sv = (float)sp[c];
    float bv = base[base_off + (bS.C == 1 ? 0 : c)];
    float sc = scale[scale_off + (sS.C == 1 ? 0 : c)];
    float mul = stddev ? (1.0f / sycl::sqrt(sc * sc + eps)) : sc;
    float res = (sv - bv) * mul * gscale + shift;
    dp[c]     = sat_cast<EltT>(res);
  }
}

// --------------------------------------------------------------------------- //
// internal queue (VOX_PROFILE opt-in) — same pattern as voxelizer/nms/cvt.
// --------------------------------------------------------------------------- //
static sycl::queue make_internal_queue() {
  sycl::device dev{sycl::gpu_selector_v};
  if (std::getenv("VOX_PROFILE")) {
    return sycl::queue{dev, sycl::property_list{sycl::property::queue::in_order{},
                                                 sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{dev, sycl::property_list{sycl::property::queue::in_order{}}};
}

// --------------------------------------------------------------------------- //
// NormalizeImplement
// --------------------------------------------------------------------------- //
class NormalizeImplement : public Normalize {
 public:
  NormalizeImplement() : profiling_enabled_(std::getenv("VOX_PROFILE") != nullptr) {}
  virtual ~NormalizeImplement() {}

  virtual bool forward(const void* in, const float* base, const float* scale, void* out, int N, int H,
                       int W, int C, Shape baseShape, Shape scaleShape, float global_scale,
                       float shift, float epsilon, DType dtype, uint32_t flags,
                       void* stream) override {
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) return false;
    // CV-CUDA rejects C>4 and C==2 (the [6][4] dispatch table has a null slot
    // for channels==2). Supported: 1, 3, 4.
    if (C != 1 && C != 3 && C != 4) return false;
    // per-axis broadcast: each base/scale axis must be 1 or match the input.
    if (!check_broadcast(baseShape, N, H, W, C)) return false;
    if (!check_broadcast(scaleShape, N, H, W, C)) return false;

    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : queue_;
    const char* inb = static_cast<const char*>(in);
    char* outb       = static_cast<char*>(out);
    bool stddev      = (flags & SCALE_IS_STDDEV) != 0;

    switch (dtype) {
      case DType::U8:  return do_normalize<uint8_t>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      case DType::S8:  return do_normalize<int8_t>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      case DType::U16: return do_normalize<uint16_t>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      case DType::S16: return do_normalize<int16_t>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      case DType::S32: return do_normalize<int32_t>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      case DType::F32: return do_normalize<float>(q, inb, outb, base, scale, N, H, W, C, baseShape, scaleShape, global_scale, shift, epsilon, stddev);
      default: return false;
    }
  }

  virtual void* internal_stream() override { return &queue_; }

  virtual bool last_profile(uint64_t& ns) override {
    if (!profiling_enabled_) return false;
    try {
      uint64_t s = ev_.get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t e = ev_.get_profiling_info<sycl::info::event_profiling::command_end>();
      ns = e - s;
    } catch (...) {
      ns = 0;
    }
    return true;
  }

 private:
  static bool check_broadcast(Shape s, int N, int H, int W, int C) {
    auto ok = [](int dim, int ref) { return dim == 1 || dim == ref; };
    return ok(s.N, N) && ok(s.H, H) && ok(s.W, W) && ok(s.C, C);
  }

  template <typename EltT>
  bool do_normalize(sycl::queue& q, const char* in, char* out, const float* base, const float* scale,
                    int N, int H, int W, int C, Shape bS, Shape sS, float gscale, float shift,
                    float eps, bool stddev) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT      = reinterpret_cast<EltT*>(out);
    size_t total    = (size_t)N * H * W;

    // SYCL device lambdas cannot implicitly capture `this`; copy args by value.
    const float* _base = base;
    const float* _scale = scale;
    int _N = N, _H = H, _W = W, _C = C;
    Shape _bS = bS, _sS = sS;
    float _gscale = gscale, _shift = shift, _eps = eps;
    bool _stddev = stddev;

    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      normalize_kernel<EltT>(i.get(0), total, inT, _base, _scale, outT, _N, _H, _W, _C, _bS, _sS, _gscale,
                             _shift, _eps, _stddev);
    });
    return true;
  }

  sycl::queue queue_{make_internal_queue()};
  bool profiling_enabled_;
  sycl::event ev_;
};

inline std::shared_ptr<Normalize> create_normalize() { return std::make_shared<NormalizeImplement>(); }

}  // namespace normalize
}  // namespace cvcuda

#endif  // CVCUDA_OPS_NORMALIZE_HPP
