// Ported from CV-CUDA 0.16.0 src/cvcuda/priv/legacy/rotate.cu (tensor path).
// SYCL port of the Rotate operator. Algorithm/semantics unchanged:
//   Backward (inverse-map) warp. For each output pixel (n,y,x):
//     dx = dstX - xShift,  dy = dstY - yShift           (double)
//     srcX = (float)(dx*cos - dy*sin)
//     srcY = (float)(dx*sin + dy*cos)
//     if srcX > -0.5 && srcX < W_in && srcY > -0.5 && srcY < H_in:
//         out = sample_src(n, srcY, srcX) with BORDER_REPLICATE + interp
//     else:
//         out = 0   (exposed corners filled black)
//   cos/sin computed on the HOST in double (CV-CUDA computes them in a 1-thread
//   device kernel; computing on host lets kernel and gold share identical
//   coeffs, isolating divergence to per-pixel float math). Interpolation types
//   NEAREST / LINEAR / CUBIC, matching CV-CUDA's InterpolationWrap.
//
// Self-contained header-only kernel (matches the cvcuda-ops-sycl convention:
// the test TUs include this header directly; build.sh only compiles the tests).
// The original CV-CUDA code is Apache-2.0 (NVIDIA); this port preserves the
// algorithm 1:1 and is also Apache-2.0.

#ifndef CVCUDA_OPS_ROTATE_HPP
#define CVCUDA_OPS_ROTATE_HPP

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "rotate_helpers.hpp"

namespace cvcuda {
namespace rotate {



// Interpolation type (subset of NVCVInterpolationType that rotate supports).
enum class Interp : int {
  NEAREST = 0,
  LINEAR = 1,
  CUBIC = 2,
};

// Data types CV-CUDA Rotate supports (U8/U16/S16/F32). S8/S32/F64/F16 are NOT
// supported by CV-CUDA Rotate -> rejected in forward().
enum class DType : int {
  U8 = 0,
  U16,
  S16,
  F32,
};

// Layout: NHWC contiguous. in/out are device USM in the same context as the
// internal queue (use internal_stream() to get it). Input and output may have
// different H/W (rotation of a rectangle into a differently-sized canvas);
// batch N must match. Channels C must be 1, 3, or 4 (CV-CUDA's funcs[6][4]
// dispatch table has a null slot for channels==2; C>4 rejected). angleDeg and
// shift are double (matches CV-CUDA's double angleDeg / double2 shift).
class Rotate {
 public:
  virtual ~Rotate() = default;

  // Returns true on success, false on invalid dtype/channels/interp/shape
  // (matching CV-CUDA INVALID_PARAMETER semantics). stream is sycl::queue*
  // (nullptr => internal queue). shiftX/shiftY map a source anchor to a dest
  // anchor (the center of rotation in dest coords).
  virtual bool forward(const void* in, void* out, int N, int Hin, int Win,
                       int Hout, int Wout, int C, double angleDeg, double shiftX,
                       double shiftY, Interp interp, DType dtype,
                       void* stream = nullptr) = 0;

  // Per-kernel device profiling from the most recent forward() call.
  // Returns true if profiling enabled (VOX_PROFILE env at construction) and a
  // run completed; ns filled with the single kernel's device time. Caller must
  // q.wait() first.
  virtual bool last_profile(uint64_t& ns) = 0;

  virtual void* internal_stream() = 0;
};

inline std::shared_ptr<Rotate> create_rotate();

}  // namespace rotate
}  // namespace cvcuda

#endif  // __CVCUDA_OPS_ROTATE_HPP__



// --------------------------------------------------------------------------- //
// SaturateCast (device): float -> integral = round-to-nearest-even (sycl::rint,
// matches PTX cvt.rni) + clamp [TypeMin, TypeMax]; float -> identity. Mirrors
// normalize/colorcvt's sat_cast_f.
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
// device kernel: rotate. Faithful 1:1 port of rotate.cu:41-63 + the
// InterpolationWrap NEAREST/LINEAR/CUBIC operator[] (InterpolationWrap.hpp).
// Backward inverse-map with BORDER_REPLICATE sampling. aCoeffs is the 6-double
// device buffer [c, s, xShift, -s, c, yShift] (compute_warpAffine, computed on
// host here). The CUDA grid used blockIdx.z = batch; here we flatten to 1D.
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void rotate_kernel(size_t idx, size_t total, const EltT* in, EltT* out,
                                 int N, int Hin, int Win, int Hout, int Wout, int C,
                                 const double* aCoeffs, Interp interp) {
  if (idx >= total) return;
  int n   = (int)(idx / ((size_t)Hout * Wout));
  int rem = (int)(idx % ((size_t)Hout * Wout));
  int y   = rem / Wout;   // dst row
  int x   = rem % Wout;   // dst col

  // Inverse map (double intermediate, cast to float — matches rotate.cu:51-55).
  const double c = aCoeffs[0], s = aCoeffs[1];
  const double dx = (double)x - aCoeffs[2];
  const double dy = (double)y - aCoeffs[5];
  const float srcX = (float)(dx * c + dy * (-aCoeffs[1]));   // dx*c - dy*s
  const float srcY = (float)(dx * (-aCoeffs[3]) + dy * c);   // dx*s + dy*c

  EltT* dp = nhwc_ptr(out, n, y, x, Hout, Wout, C);

  // In-range test (rotate.cu:59). Outside -> fill 0 (see file header note).
  if (!(srcX > -0.5f && srcX < (float)Win && srcY > -0.5f && srcY < (float)Hin)) {
    for (int cch = 0; cch < C; ++cch) dp[cch] = (EltT)0;
    return;
  }

  if (interp == Interp::NEAREST) {
    // InterpolationWrap<NEAREST>: idx = floor(coord + 0.5), BORDER_REPLICATE.
    int ix = (int)sycl::floor(srcX + 0.5f);
    int iy = (int)sycl::floor(srcY + 0.5f);
    ix = border_replicate(ix, Win);
    iy = border_replicate(iy, Hin);
    const EltT* sp = nhwc_ptr(in, n, iy, ix, Hin, Win, C);
    for (int cch = 0; cch < C; ++cch) dp[cch] = sp[cch];
    return;
  }

  if (interp == Interp::LINEAR) {
    // InterpolationWrap<LINEAR>: x1=floor(x), x2=x1+1; bilinear 4-tap.
    int x1 = (int)sycl::floor(srcX);
    int x2 = x1 + 1;
    int y1 = (int)sycl::floor(srcY);
    int y2 = y1 + 1;
    float fx = srcX - (float)x1;   // = x - x1
    float fy = srcY - (float)y1;
    float w00 = ((float)x2 - srcX) * ((float)y2 - srcY);
    float w10 = (fx) * ((float)y2 - srcY);
    float w01 = ((float)x2 - srcX) * (fy);
    float w11 = (fx) * (fy);
    const EltT* s00 = nhwc_ptr(in, n, border_replicate(y1, Hin), border_replicate(x1, Win), Hin, Win, C);
    const EltT* s10 = nhwc_ptr(in, n, border_replicate(y1, Hin), border_replicate(x2, Win), Hin, Win, C);
    const EltT* s01 = nhwc_ptr(in, n, border_replicate(y2, Hin), border_replicate(x1, Win), Hin, Win, C);
    const EltT* s11 = nhwc_ptr(in, n, border_replicate(y2, Hin), border_replicate(x2, Win), Hin, Win, C);
    for (int cch = 0; cch < C; ++cch) {
      float v = (float)s00[cch] * w00 + (float)s10[cch] * w10 + (float)s01[cch] * w01 +
                (float)s11[cch] * w11;
      dp[cch] = sat_cast<EltT>(v);
    }
    return;
  }

  // CUBIC: InterpolationWrap<CUBIC> + GetCubicCoeffs, 4x4 tap.
  int ix = (int)sycl::floor(srcX);
  int iy = (int)sycl::floor(srcY);
  float wx[4], wy[4];
  cubic_coeffs(srcX - (float)ix, wx[0], wx[1], wx[2], wx[3]);
  cubic_coeffs(srcY - (float)iy, wy[0], wy[1], wy[2], wy[3]);
  for (int cch = 0; cch < C; ++cch) {
    float sum = 0.f;
    for (int cy = -1; cy <= 2; ++cy) {
      int syi = border_replicate(iy + cy, Hin);
      for (int cx = -1; cx <= 2; ++cx) {
        int sxi = border_replicate(ix + cx, Win);
        float w = wx[cx + 1] * wy[cy + 1];
        sum += (float)nhwc_ptr(in, n, syi, sxi, Hin, Win, C)[cch] * w;
      }
    }
    dp[cch] = sat_cast<EltT>(sum);
  }
}

// --------------------------------------------------------------------------- //
// internal queue (VOX_PROFILE opt-in) — same pattern as voxelizer/nms/cvt/norm.
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
// RotateImplement
// --------------------------------------------------------------------------- //
class RotateImplement : public Rotate {
 public:
  RotateImplement()
      : profiling_enabled_(std::getenv("VOX_PROFILE") != nullptr),
        coeffs_(sycl::malloc_device<double>(6, queue_)) {}

  virtual ~RotateImplement() {
    if (coeffs_) sycl::free(coeffs_, queue_);
  }

  virtual bool forward(const void* in, void* out, int N, int Hin, int Win, int Hout,
                       int Wout, int C, double angleDeg, double shiftX, double shiftY,
                       Interp interp, DType dtype, void* stream) override {
    if (N <= 0 || Hin <= 0 || Win <= 0 || Hout <= 0 || Wout <= 0 || C <= 0) return false;
    // CV-CUDA rejects C>4 and C==2 (the [6][4] dispatch table has a null slot
    // for channels==2). Supported: 1, 3, 4.
    if (C != 1 && C != 3 && C != 4) return false;

    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : queue_;
    const char* inb = static_cast<const char*>(in);
    char* outb       = static_cast<char*>(out);

    // Compute the 6 affine coeffs on the host in double (compute_warpAffine).
    static const double PI = 3.1415926535897932384626433832795;
    double rad = angleDeg * PI / 180.0;
    double c = std::cos(rad), s = std::sin(rad);
    double h_coeffs[6] = {c, s, shiftX, -s, c, shiftY};
    q.memcpy(coeffs_, h_coeffs, 6 * sizeof(double)).wait();

    switch (dtype) {
      case DType::U8:  return do_rotate<uint8_t>(q, inb, outb, N, Hin, Win, Hout, Wout, C, interp);
      case DType::U16: return do_rotate<uint16_t>(q, inb, outb, N, Hin, Win, Hout, Wout, C, interp);
      case DType::S16: return do_rotate<int16_t>(q, inb, outb, N, Hin, Win, Hout, Wout, C, interp);
      case DType::F32: return do_rotate<float>(q, inb, outb, N, Hin, Win, Hout, Wout, C, interp);
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
  template <typename EltT>
  bool do_rotate(sycl::queue& q, const char* in, char* out, int N, int Hin, int Win,
                 int Hout, int Wout, int C, Interp interp) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT      = reinterpret_cast<EltT*>(out);
    size_t total    = (size_t)N * Hout * Wout;

    const double* _coeffs = coeffs_;
    int _N = N, _Hin = Hin, _Win = Win, _Hout = Hout, _Wout = Wout, _C = C;
    Interp _interp = interp;

    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      rotate_kernel<EltT>(i.get(0), total, inT, outT, _N, _Hin, _Win, _Hout, _Wout, _C, _coeffs,
                          _interp);
    });
    return true;
  }

  sycl::queue queue_{make_internal_queue()};
  bool profiling_enabled_;
  sycl::event ev_;
  double* coeffs_;   // 6-double device buffer (matches CV-CUDA's d_aCoeffs)
};

inline std::shared_ptr<Rotate> create_rotate() { return std::make_shared<RotateImplement>(); }

}  // namespace rotate
}  // namespace cvcuda

#endif  // CVCUDA_OPS_ROTATE_HPP
