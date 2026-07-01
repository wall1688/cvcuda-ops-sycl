// Ported from CV-CUDA 0.16.0 src/cvcuda/priv/legacy/cvt_color.cu (tensor path).
// SYCL port of the CvtColor operator (6 color families / 71 codes). Algorithm
// and semantics are unchanged from the NVIDIA original.
//
// Self-contained header-only kernel (matches the cvcuda-ops-sycl convention:
// the test TUs include this header directly; build.sh only compiles the tests).
// The original CV-CUDA code is Apache-2.0 (NVIDIA); this port preserves the
// algorithm 1:1 and is also Apache-2.0.

#ifndef CVCUDA_OPS_CVTCOLOR_HPP
#define CVCUDA_OPS_CVTCOLOR_HPP

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "cvt_helpers.hpp"

namespace cvcuda {
namespace cvt {



// Data types (mirrors CV-CUDA legacy DataType for the subset we support).
enum class DType : int {
  U8 = 0,
  S8,
  U16,
  S16,
  F16,  // treated as 16-bit storage (integer-style path), matching CV-CUDA
  S32,
  F32,
  F64,
};

// Color conversion codes — use the NVCVColorConversionCode numeric values
// directly (0..148). Only the codes CV-CUDA actually implements are supported;
// unsupported codes return false from forward() (matching CV-CUDA's
// INVALID_PARAMETER). Family A (0-5) is implemented in Stage 0; later stages
// add gray (6-11), YUV (82-85), HSV (40,41,54,55,66,67,70,71),
// YUV420 (90-106,127-134,140-147), YUV422 (107,108,111,112,115-124).
using Code = int;

// Layout: NHWC contiguous. in/out are device USM in the same context as the
// internal queue (use internal_stream() to get it). CvtColor requires
// in_dtype == out_dtype (CV-CUDA rejects mismatched dtypes), so a single dtype
// is passed. Channels (Cin/Cout) must match the code's expected src/dst channel
// counts; N/H/W must match between in and out (except YUV subsampled layouts,
// handled per-code in later stages).
class CvtColor {
 public:
  virtual ~CvtColor() = default;

  // Returns true on success, false on invalid code/dtype/channels/shape
  // (matching CV-CUDA error semantics). stream is sycl::queue* (nullptr =>
  // internal queue).
  virtual bool forward(const void* in, void* out, int N, int H, int W, int Cin, int Cout,
                       DType dtype, Code code, void* stream = nullptr) = 0;

  // Per-kernel device profiling from the most recent forward() call.
  // Returns true if profiling enabled (VOX_PROFILE env at construction) and a
  // run completed; ns filled with the kernel's device time. Caller must q.wait()
  // first. For multi-kernel codes (none in Stage 0), ns is the sum.
  virtual bool last_profile(uint64_t& ns) = 0;

  virtual void* internal_stream() = 0;
};

inline std::shared_ptr<CvtColor> create_cvt();

}  // namespace cvt
}  // namespace cvcuda

#endif  // __CVCUDA_OPS_CVTCOLOR_HPP__



// --------------------------------------------------------------------------- //
// device kernel: Family A — channel swap / alpha add-drop.
// Faithful 1:1 port of rgb_to_bgr_nhwc + load_bgra/store_bgra (cvt_color.cu:
// 183-232). One thread per output pixel. bidx routes R/B at load; store always
// writes BGR order. Net effect: codes 0,1 = copy (+/-alpha); codes 2,3,4,5 =
// reverse [2,1,0] (+ alpha add/drop/keep).
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void family_a_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                   int W, int Cin, int Cout, int bidx) {
  if (idx >= total) return;
  size_t px = idx;
  int n = (int)(px / ((size_t)H * W));
  int rem = (int)(px % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;

  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, Cin);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, Cout);

  // load_bgra: B,G,R per bidx; A = src alpha if Cin==4 else Alpha.
  EltT B, G, R, A;
  if (bidx == 0) {
    B = sp[0];
    G = sp[1];
    R = sp[2];
  } else {
    B = sp[2];
    G = sp[1];
    R = sp[0];
  }
  A = (Cin == 4) ? sp[3] : cvt::AlphaVal<EltT>();

  // store_bgra (bidx=0): dst = [B,G,R] (+A if Cout==4).
  dp[0] = B;
  dp[1] = G;
  dp[2] = R;
  if (Cout == 4) dp[3] = A;
}

// --------------------------------------------------------------------------- //
// device kernel: Family B — BGR/RGB/RGBA/BGRA -> GRAY (codes 6,7,10,11).
// Faithful port of bgr_to_gray_nhwc (cvt_color.cu:251-273). int path = Q15
// fixed-point CV_DESCALE (rounds); float path = B2YF/G2YF/R2YF dot. Note the
// GPU int path rounds while the CPU gold truncates -> U8 differs by <=1, U16
// by <=2 (matches CV-CUDA test tolerances).
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void bgr_to_gray_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                      int W, int Cin, int bidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, Cin);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, 1);

  EltT B, G, R;
  if (bidx == 0) {
    B = sp[0];
    G = sp[1];
    R = sp[2];
  } else {
    B = sp[2];
    G = sp[1];
    R = sp[0];
  }

  EltT gray;
  if constexpr (std::is_integral_v<EltT>) {
    int yv = cvt::cv_descale((int)B * cvt::BY15 + (int)G * cvt::GY15 + (int)R * cvt::RY15, cvt::gray_shift);
    gray = (EltT)yv;
  } else {
    gray = (EltT)(B * cvt::B2YF + G * cvt::G2YF + R * cvt::R2YF);
  }
  dp[0] = gray;
}

// --------------------------------------------------------------------------- //
// device kernel: Family B — GRAY -> BGR/BGRA (codes 8,9).
// Faithful port of gray_to_bgr_nhwc (cvt_color.cu:234-249): replicate gray to
// all output channels (alpha = gray, matching the GPU code). Pure replicate.
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void gray_to_bgr_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                      int W, int Cout) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, 1);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, Cout);
  EltT g = sp[0];
  for (int c = 0; c < Cout; ++c) dp[c] = g;
}

// SaturateCast for the int path: int -> unsigned integral (U8/U16), clamp to
// [0, max] (matches CV-CUDA BaseSaturateCastImpl signed->unsigned big->small).
// Only used for integral EltT; the float path needs no saturate cast.
template <typename T>
static inline T sat_int(int u) {
  if (u < 0) return (T)0;
  int mx = (int)cvt::TypeMax<T>();
  if (u > mx) return (T)mx;
  return (T)u;
}

// --------------------------------------------------------------------------- //
// device kernel: Family C — BGR/RGB -> YUV (codes 82,83).
// Faithful port of bgr_to_yuv_nhwc + bgr_to_yuv_int/_float (cvt_color.cu:
// 275-323). int path = Q14 fixed-point CV_DESCALE + SaturateCast; float path =
// R2YF/G2YF/B2YF + delta 0.5. Output [Y, Cb, Cr].
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void bgr_to_yuv_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                     int W, int bidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, 3);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, 3);

  EltT B, G, R;
  if (bidx == 0) {
    B = sp[0];
    G = sp[1];
    R = sp[2];
  } else {
    B = sp[2];
    G = sp[1];
    R = sp[0];
  }

  EltT Y, Cb, Cr;
  if constexpr (std::is_integral_v<EltT>) {
    const int delta = ((int)(EltT)(cvt::TypeMax<EltT>() / 2 + 1)) << cvt::yuv_shift;
    int Bv = B, Gv = G, Rv = R;
    int Yv  = cvt::cv_descale(Rv * cvt::R2Y + Gv * cvt::G2Y + Bv * cvt::B2Y, cvt::yuv_shift);
    int Crv = cvt::cv_descale((Rv - Yv) * cvt::R2VI + delta, cvt::yuv_shift);
    int Cbv = cvt::cv_descale((Bv - Yv) * cvt::B2UI + delta, cvt::yuv_shift);
    Y = sat_int<EltT>(Yv);
    Cb = sat_int<EltT>(Cbv);
    Cr = sat_int<EltT>(Crv);
  } else {
    float Bf = B, Gf = G, Rf = R;
    float Yf = Rf * cvt::R2YF + Gf * cvt::G2YF + Bf * cvt::B2YF;
    Cr = (EltT)((Rf - Yf) * cvt::R2VF + 0.5f);
    Cb = (EltT)((Bf - Yf) * cvt::B2UF + 0.5f);
    Y = (EltT)Yf;
  }
  dp[0] = Y;
  dp[1] = Cb;
  dp[2] = Cr;
}

// --------------------------------------------------------------------------- //
// device kernel: Family C — YUV -> BGR/RGB (codes 84,85).
// Faithful port of yuv_to_bgr_nhwc + yuv_to_bgr_int/_flt (cvt_color.cu:
// 325-369). int path = Q14 CV_DESCALE + SaturateCast; float path = V2RF etc +
// delta 0.5. Store routes B/R by bidx (bidx==0 -> BGR, bidx==2 -> RGB).
// --------------------------------------------------------------------------- //
template <typename EltT>
static inline void yuv_to_bgr_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                     int W, int bidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, 3);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, 3);

  EltT Yv = sp[0], Cbv = sp[1], Crv = sp[2];
  EltT B, G, R;
  if constexpr (std::is_integral_v<EltT>) {
    const int delta = (int)(EltT)(cvt::TypeMax<EltT>() / 2 + 1);
    int Y = Yv, Cb = Cbv, Cr = Crv;
    int Bv = Y + cvt::cv_descale((Cb - delta) * cvt::U2BI, cvt::yuv_shift);
    int Gv = Y + cvt::cv_descale((Cb - delta) * cvt::U2GI + (Cr - delta) * cvt::V2GI, cvt::yuv_shift);
    int Rv = Y + cvt::cv_descale((Cr - delta) * cvt::V2RI, cvt::yuv_shift);
    B = sat_int<EltT>(Bv);
    G = sat_int<EltT>(Gv);
    R = sat_int<EltT>(Rv);
  } else {
    float Y = Yv, Cb = Cbv, Cr = Crv;
    B = (EltT)(Y + (Cb - 0.5f) * cvt::U2BF);
    G = (EltT)(Y + (Cb - 0.5f) * cvt::U2GF + (Cr - 0.5f) * cvt::V2GF);
    R = (EltT)(Y + (Cr - 0.5f) * cvt::V2RF);
  }
  // store_bgra: bidx==0 -> [B,G,R]; bidx==2 -> [R,G,B].
  if (bidx == 0) {
    dp[0] = B;
    dp[1] = G;
    dp[2] = R;
  } else {
    dp[0] = R;
    dp[1] = G;
    dp[2] = B;
  }
}

// SaturateCast for the float path: float -> integral, round-to-nearest-even
// (sycl::rint, matches PTX cvt.rni) + clamp to [TypeMin, TypeMax].
template <typename T>
static inline T sat_cast_f(float u) {
  float r = sycl::rint(u);
  if (r < (float)cvt::TypeMin<T>()) r = (float)cvt::TypeMin<T>();
  if (r > (float)cvt::TypeMax<T>()) r = (float)cvt::TypeMax<T>();
  return (T)r;
}

// --------------------------------------------------------------------------- //
// Family D: HSV math helpers (ported verbatim from cvt_color.cu:371-479).
// --------------------------------------------------------------------------- //
static inline void bgr_to_hsv_uchar(uint8_t b8, uint8_t g8, uint8_t r8, uint8_t& h8, uint8_t& s8,
                                    uint8_t& v8, bool isFullRange) {
  const int hrange    = isFullRange ? 256 : 180;
  const int hsv_shift = 12;
  const int b = (int)b8, g = (int)g8, r = (int)r8;
  const int vmin = sycl::min(b, sycl::min(g, r));
  const int v    = sycl::max(b, sycl::max(g, r));
  const int diff = v - vmin;
  const int vr   = v == r ? -1 : 0;
  const int vg   = v == g ? -1 : 0;
  const int hdiv_table = diff == 0 ? 0 : sat_cast_f<int>((hrange << hsv_shift) / (6.f * diff));
  const int sdiv_table = v == 0 ? 0 : sat_cast_f<int>((255 << hsv_shift) / (float)v);
  const int s = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
  int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
  h = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
  h += h < 0 ? hrange : 0;
  h8 = sat_int<uint8_t>(h);
  s8 = (uint8_t)s;
  v8 = (uint8_t)v;
}

static inline void bgr_to_hsv_float(float b, float g, float r, float& h, float& s, float& v) {
  float vmin = sycl::min(r, sycl::min(g, b));
  v          = sycl::max(r, sycl::max(g, b));
  float diff = v - vmin;
  s          = diff / (sycl::fabs(v) + FLT_EPSILON);
  diff       = 60.f / (diff + FLT_EPSILON);
  if (v == r) h = (g - b) * diff;
  else if (v == g) h = (b - r) * diff + 120.f;
  else h = (r - g) * diff + 240.f;
  if (h < 0.f) h += 360.f;
}

template <typename T>
static inline T select4_reg(const T (&tab)[4], int idx) {
  T out = idx == 1 ? tab[1] : tab[0];
  out = idx == 2 ? tab[2] : out;
  out = idx == 3 ? tab[3] : out;
  return out;
}

static inline void hsv_to_bgr_float(float h, float s, float v, float& b, float& g, float& r) {
  if (s == 0) {
    b = g = r = v;
  } else {
    h += 6 * (h < 0);
    int idx = static_cast<int>(h);
    h -= idx;
    idx = (idx % 6) << 2;
    const float tab[4] = {v, v * (1 - s), v * (1 - s * h), v * (1 - s * (1 - h))};
    constexpr int32_t idx_lutb = 0x00200311;
    constexpr int32_t idx_lutg = 0x00112003;
    constexpr int32_t idx_lutr = 0x00031120;
    b = select4_reg(tab, (idx_lutb >> idx) & 0xf);
    g = select4_reg(tab, (idx_lutg >> idx) & 0xf);
    r = select4_reg(tab, (idx_lutr >> idx) & 0xf);
  }
}

// device kernel: BGR/RGB -> HSV (codes 40,41,66,67). store [H,S,V].
template <typename EltT>
static inline void bgr_to_hsv_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                    int W, int bidx, bool isFullRange) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, 3);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, 3);
  EltT B, G, R;
  if (bidx == 0) {
    B = sp[0];
    G = sp[1];
    R = sp[2];
  } else {
    B = sp[2];
    G = sp[1];
    R = sp[0];
  }
  EltT Hv, Sv, Vv;
  if constexpr (std::is_same_v<EltT, uint8_t>)
    bgr_to_hsv_uchar(B, G, R, Hv, Sv, Vv, isFullRange);
  else
    bgr_to_hsv_float((float)B, (float)G, (float)R, Hv, Sv, Vv);
  dp[0] = Hv;
  dp[1] = Sv;
  dp[2] = Vv;
}

// device kernel: HSV -> BGR/RGB (codes 54,55,70,71). store routes by bidx.
template <typename EltT>
static inline void hsv_to_bgr_kernel(size_t idx, size_t total, const EltT* in, EltT* out, int N, int H,
                                    int W, int bidx, bool isFullRange, int Cout) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  const EltT* sp = cvt::nhwc_ptr(in, n, y, x, H, W, 3);
  EltT* dp = cvt::nhwc_ptr(out, n, y, x, H, W, Cout);
  EltT Hin = sp[0], Sin = sp[1], Vin = sp[2];
  EltT B, G, R;
  if constexpr (std::is_same_v<EltT, uint8_t>) {
    const float scaleH = isFullRange ? (6.0f / 256.0f) : (6.0f / 180.0f);
    constexpr float scaleSV = 1.0f / 255.0f;
    float Bf, Gf, Rf;
    hsv_to_bgr_float((float)Hin * scaleH, (float)Sin * scaleSV, (float)Vin * scaleSV, Bf, Gf, Rf);
    B = sat_cast_f<uint8_t>(Bf * 255.0f);
    G = sat_cast_f<uint8_t>(Gf * 255.0f);
    R = sat_cast_f<uint8_t>(Rf * 255.0f);
  } else {
    constexpr float scaleH = 6.0f / 360.0f;
    float Bf, Gf, Rf;
    hsv_to_bgr_float((float)Hin * scaleH, (float)Sin, (float)Vin, Bf, Gf, Rf);
    B = (EltT)Bf;
    G = (EltT)Gf;
    R = (EltT)Rf;
  }
  if (bidx == 0) {
    dp[0] = B;
    dp[1] = G;
    dp[2] = R;
  } else {
    dp[0] = R;
    dp[1] = G;
    dp[2] = B;
  }
  if (Cout == 4) dp[3] = cvt::AlphaVal<EltT>();
}

// --------------------------------------------------------------------------- //
// Family E: YUV420 math + layout (ported from cvt_color.cu:516-640). 8U only.
// ITUR_BT_601 fixed-point. YUV tensor is scalar [N, 1.5*H, W, 1] (C=1).
// --------------------------------------------------------------------------- //
static inline void bgr_to_yuv42xxp(uint8_t b, uint8_t g, uint8_t r, uint8_t& Y, uint8_t& U, uint8_t& V) {
  const int shifted16  = (16 << cvt::ITUR_BT_601_SHIFT);
  const int halfShift  = (1 << (cvt::ITUR_BT_601_SHIFT - 1));
  int yy = cvt::ITUR_BT_601_CRY * r + cvt::ITUR_BT_601_CGY * g + cvt::ITUR_BT_601_CBY * b + halfShift + shifted16;
  Y = sat_int<uint8_t>(yy >> cvt::ITUR_BT_601_SHIFT);
  const int shifted128 = (128 << cvt::ITUR_BT_601_SHIFT);
  int uu = cvt::ITUR_BT_601_CRU * r + cvt::ITUR_BT_601_CGU * g + cvt::ITUR_BT_601_CBU * b + halfShift + shifted128;
  int vv = cvt::ITUR_BT_601_CBU * r + cvt::ITUR_BT_601_CGV * g + cvt::ITUR_BT_601_CBV * b + halfShift + shifted128;
  U = sat_int<uint8_t>(uu >> cvt::ITUR_BT_601_SHIFT);
  V = sat_int<uint8_t>(vv >> cvt::ITUR_BT_601_SHIFT);
}

static inline void yuv42xxp_to_bgr(int Y, int U, int V, uint8_t& b, uint8_t& g, uint8_t& r) {
  const int C0 = cvt::ITUR_BT_601_CY, C1 = cvt::ITUR_BT_601_CVR, C2 = cvt::ITUR_BT_601_CVG,
            C3 = cvt::ITUR_BT_601_CUG, C4 = cvt::ITUR_BT_601_CUB;
  int yy = sycl::max(0, Y - 16) * C0;
  int uu = U - 128;
  int vv = V - 128;
  r = sat_int<uint8_t>(cvt::cv_descale(yy + C1 * vv, cvt::ITUR_BT_601_SHIFT));
  g = sat_int<uint8_t>(cvt::cv_descale(yy + C2 * vv + C3 * uu, cvt::ITUR_BT_601_SHIFT));
  b = sat_int<uint8_t>(cvt::cv_descale(yy + C4 * uu, cvt::ITUR_BT_601_SHIFT));
}

// load_yuv420: yuv[n][row][col] = yuv + n*yuv_h*W + row*W + col. H = Y-plane height.
template <bool IsSemiPlanar>
static inline void load_yuv420(const uint8_t* yuv, int W, int H, int yuv_h, int n, int x, int y, int uidx,
                               uint8_t& Y, uint8_t& U, uint8_t& V) {
  size_t base = (size_t)n * yuv_h * W;
  if constexpr (IsSemiPlanar) {
    int uv_y = H + y / 2;
    int uv_x = x & ~1;
    Y = yuv[base + (size_t)y * W + x];
    U = yuv[base + (size_t)uv_y * W + uv_x + uidx];
    V = yuv[base + (size_t)uv_y * W + uv_x + (uidx ^ 1)];
  } else {
    int by = H + y / 4;
    int h4 = H / 4;
    int uv_x = (x / 2) + ((W / 2) & -((y / 2) & 1));
    Y = yuv[base + (size_t)y * W + x];
    U = yuv[base + (size_t)(by + h4 * uidx) * W + uv_x];
    V = yuv[base + (size_t)(by + h4 * (uidx ^ 1)) * W + uv_x];
  }
}

template <bool IsSemiPlanar>
static inline void store_yuv420(uint8_t* yuv, int W, int H, int yuv_h, int n, int x, int y, int uidx,
                                uint8_t Y, uint8_t U, uint8_t V) {
  size_t base = (size_t)n * yuv_h * W;
  yuv[base + (size_t)y * W + x] = Y;
  if (y % 2 == 0 && x % 2 == 0) {
    if constexpr (IsSemiPlanar) {
      int uv_y = H + y / 2;
      int uv_x = x & ~1;
      yuv[base + (size_t)uv_y * W + uv_x + uidx] = U;
      yuv[base + (size_t)uv_y * W + uv_x + (uidx ^ 1)] = V;
    } else {
      int by = H + y / 4;
      int h4 = H / 4;
      int uv_x = (x / 2) + ((W / 2) & -((y / 2) & 1));
      yuv[base + (size_t)(by + h4 * uidx) * W + uv_x] = U;
      yuv[base + (size_t)(by + h4 * (uidx ^ 1)) * W + uv_x] = V;
    }
  }
}

// BGR -> YUV420 (codes 127-134,140-147). Per BGR pixel.
template <bool IsSemiPlanar>
static inline void bgr_to_yuv420_kernel(size_t idx, size_t total, const uint8_t* in, uint8_t* out, int N,
                                        int H, int W, int Cin, int bidx, int uidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  int yuv_h = H + H / 2;
  const uint8_t* sp = in + ((size_t)n * H + y) * W * Cin + (size_t)x * Cin;
  uint8_t B, G, R;
  if (bidx == 0) {
    B = sp[0]; G = sp[1]; R = sp[2];
  } else {
    B = sp[2]; G = sp[1]; R = sp[0];
  }
  uint8_t Y, U, V;
  bgr_to_yuv42xxp(B, G, R, Y, U, V);
  store_yuv420<IsSemiPlanar>(out, W, H, yuv_h, n, x, y, uidx, Y, U, V);
}

// YUV420 -> BGR (codes 90-105). Per BGR pixel.
template <bool IsSemiPlanar>
static inline void yuv420_to_bgr_kernel(size_t idx, size_t total, const uint8_t* in, uint8_t* out, int N,
                                        int H, int W, int Cout, int bidx, int uidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  int yuv_h = H + H / 2;
  uint8_t Y, U, V;
  load_yuv420<IsSemiPlanar>(in, W, H, yuv_h, n, x, y, uidx, Y, U, V);
  uint8_t b, g, r;
  yuv42xxp_to_bgr((int)Y, (int)U, (int)V, b, g, r);
  uint8_t* dp = out + ((size_t)n * H + y) * W * Cout + (size_t)x * Cout;
  if (bidx == 0) {
    dp[0] = b; dp[1] = g; dp[2] = r;
  } else {
    dp[0] = r; dp[1] = g; dp[2] = b;
  }
  if (Cout == 4) dp[3] = cvt::AlphaVal<uint8_t>();
}

// --------------------------------------------------------------------------- //
// Family F: YUV422 (codes 107-124, 8U only). Ported from cvt_color.cu:695-756.
// 4:2:2: chroma subsampled 2:1 horizontally. YUV tensor scalar [N,H,2W,1]
// (2 Y + 1 U + 1 V per 2 dst pixels). yidx routes Y position (YUY2/YVYU=0,
// UYVY=1); uidx routes U/V (YUY2/UYVY=0, YVYU=2). Reuses yuv42xxp_to_bgr.
// 1 thread per output pixel (kernel does 2/thread; equivalent).
// --------------------------------------------------------------------------- //
static inline void yuv422_to_bgr_kernel(size_t idx, size_t total, const uint8_t* in, uint8_t* out, int N,
                                        int H, int W, int Cout, int bidx, int yidx, int uidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  int yuv_w = 2 * W;
  size_t ybase = (size_t)n * H * yuv_w + (size_t)y * yuv_w + (size_t)(2 * (x & ~1));
  int Y = in[ybase + yidx + (x % 2) * 2];
  int U = in[ybase + (yidx ^ 1) + uidx];
  int V = in[ybase + (yidx ^ 1) + (uidx ^ 2)];
  uint8_t b, g, r;
  yuv42xxp_to_bgr(Y, U, V, b, g, r);
  uint8_t* dp = out + ((size_t)n * H + y) * W * Cout + (size_t)x * Cout;
  dp[bidx] = b;
  dp[1] = g;
  dp[bidx ^ 2] = r;
  if (Cout == 4) dp[3] = cvt::AlphaVal<uint8_t>();
}

static inline void yuv422_to_gray_kernel(size_t idx, size_t total, const uint8_t* in, uint8_t* out, int N,
                                         int H, int W, int yidx) {
  if (idx >= total) return;
  int n = (int)(idx / ((size_t)H * W));
  int rem = (int)(idx % ((size_t)H * W));
  int y = rem / W;
  int x = rem % W;
  int yuv_w = 2 * W;
  size_t ybase = (size_t)n * H * yuv_w + (size_t)y * yuv_w + (size_t)(2 * (x & ~1));
  out[(size_t)n * H * W + (size_t)y * W + x] = in[ybase + yidx + (x % 2) * 2];
}

// --------------------------------------------------------------------------- //
// internal queue (VOX_PROFILE opt-in) — same pattern as voxelizer/nms.
// --------------------------------------------------------------------------- //
static sycl::queue make_internal_queue() {
  sycl::device dev{sycl::gpu_selector_v};
  if (std::getenv("VOX_PROFILE")) {
    return sycl::queue{dev, sycl::property_list{sycl::property::queue::in_order{},
                                                 sycl::property::queue::enable_profiling{}}};
  }
  return sycl::queue{dev, sycl::property::queue::in_order{}};
}

// --------------------------------------------------------------------------- //
// CvtColorImplement
// --------------------------------------------------------------------------- //
class CvtColorImplement : public CvtColor {
 public:
  CvtColorImplement() : profiling_enabled_(std::getenv("VOX_PROFILE") != nullptr) {}
  virtual ~CvtColorImplement() {}

  virtual bool forward(const void* in, void* out, int N, int H, int W, int Cin, int Cout, DType dtype,
                       Code code, void* stream) override {
    if (N <= 0 || H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0) return false;
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : queue_;
    const char* inb = static_cast<const char*>(in);
    char* outb = static_cast<char*>(out);

    // ---- Family E: YUV420 (codes 90-106, 127-134, 140-147), 8U only ----
    if ((code >= 90 && code <= 106) || (code >= 127 && code <= 134) || (code >= 140 && code <= 147))
      return forward_yuv420(q, inb, outb, N, H, W, Cin, Cout, dtype, code);

    // ---- Family F: YUV422 (codes 107-124 except 109,110,113,114), 8U only ----
    if (code >= 107 && code <= 124 && code != 109 && code != 110 && code != 113 && code != 114)
      return forward_yuv422(q, inb, outb, N, H, W, Cin, Cout, dtype, code);

    switch (code) {
      // ---- Family A: channel swap / alpha (codes 0-5) ----
      case 0:  // BGR2BGRA
      case 1:  // BGRA2BGR
      case 2:  // BGR2RGBA
      case 3:  // RGBA2BGR
      case 4:  // BGR2RGB
      case 5: { // BGRA2RGBA
        int sch  = (code == 1 || code == 3 || code == 5) ? 4 : 3;
        int dch  = (code == 0 || code == 2 || code == 5) ? 4 : 3;
        int bidx = (code != 1 && code != 0) ? 2 : 0;
        if (Cin != sch || Cout != dch) return false;
        // 16F cannot add alpha to output (sch<4, dch==4) — CV-CUDA guard.
        if (dtype == DType::F16 && sch < 4 && dch == 4) return false;
        return launch_family_a(q, inb, outb, N, H, W, Cin, Cout, bidx, dtype);
      }
      // ---- Family B: BGR/RGB -> GRAY (codes 6,7,10,11) ----
      case 6:   // BGR2GRAY
      case 7:   // RGB2GRAY
      case 10:  // BGRA2GRAY
      case 11: { // RGBA2GRAY
        int bidx = (code == 7 || code == 11) ? 2 : 0;
        int sch  = (code == 10 || code == 11) ? 4 : 3;
        if (Cin != sch || Cout != 1) return false;
        // BGR->GRAY supports only 8U/16U/32F (CV-CUDA).
        if (dtype != DType::U8 && dtype != DType::U16 && dtype != DType::F32) return false;
        return launch_bgr_to_gray(q, inb, outb, N, H, W, Cin, bidx, dtype);
      }
      // ---- Family B: GRAY -> BGR/BGRA (codes 8,9) ----
      case 8:   // GRAY2BGR
      case 9: { // GRAY2BGRA / GRAY2RGBA
        int dch = (code == 9) ? 4 : 3;
        if (Cin != 1 || Cout != dch) return false;
        if (dtype == DType::F16 && dch == 4) return false;  // 16F + alpha guard
        return launch_gray_to_bgr(q, inb, outb, N, H, W, Cout, dtype);
      }
      // ---- Family C: packed YUV (codes 82-85) ----
      case 82:  // BGR2YUV
      case 83:  // RGB2YUV
      case 84:  // YUV2BGR
      case 85: { // YUV2RGB
        int bidx = (code == 82 || code == 84) ? 0 : 2;
        if (Cin != 3 || Cout != 3) return false;
        if (dtype != DType::U8 && dtype != DType::U16 && dtype != DType::F32) return false;
        if (code <= 83)
          return launch_bgr_to_yuv(q, inb, outb, N, H, W, bidx, dtype);
        else
          return launch_yuv_to_bgr(q, inb, outb, N, H, W, bidx, dtype);
      }
      // ---- Family D: HSV (codes 40,41,54,55,66,67,70,71) ----
      case 40:  // BGR2HSV
      case 41:  // RGB2HSV
      case 66:  // BGR2HSV_FULL
      case 67:  // RGB2HSV_FULL
      case 54:  // HSV2BGR
      case 55:  // HSV2RGB
      case 70:  // HSV2BGR_FULL
      case 71: { // HSV2RGB_FULL
        bool isFullRange = (code == 66 || code == 67 || code == 70 || code == 71);
        bool to_hsv = (code == 40 || code == 41 || code == 66 || code == 67);
        int bidx;
        if (to_hsv)
          bidx = (code == 40 || code == 66) ? 0 : 2;
        else
          bidx = (code == 54 || code == 70) ? 0 : 2;
        if (dtype != DType::U8 && dtype != DType::F32) return false;
        if (to_hsv) {
          if (Cin != 3 || Cout != 3) return false;
          return launch_bgr_to_hsv(q, inb, outb, N, H, W, bidx, isFullRange, dtype);
        } else {
          if (Cin != 3 || (Cout != 3 && Cout != 4)) return false;
          return launch_hsv_to_bgr(q, inb, outb, N, H, W, bidx, isFullRange, Cout, dtype);
        }
      }
      default:
        // Not yet implemented in this stage (or unsupported by CV-CUDA).
        return false;
    }
  }

  virtual void* internal_stream() override { return &queue_; }

  // ---- Family E: YUV420 (8U only) ----
  bool forward_yuv420(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin, int Cout,
                      DType dtype, Code code) {
    if (dtype != DType::U8) return false;
    if (H % 2 != 0 || W % 2 != 0) return false;
    const uint8_t* inU = reinterpret_cast<const uint8_t*>(in);
    uint8_t* outU = reinterpret_cast<uint8_t*>(out);
    int yuv_h = H + H / 2;
    size_t total = (size_t)N * H * W;

    if (code == 106) {  // YUV2GRAY_420: copy Y plane per sample.
      if (Cin != 1 || Cout != 1) return false;
      for (int n = 0; n < N; ++n)
        ev_ = q.memcpy(outU + (size_t)n * H * W, inU + (size_t)n * yuv_h * W, (size_t)H * W);
      return true;
    }

    if (code >= 90 && code <= 105) {  // YUV420 -> BGR
      if (Cin != 1 || (Cout != 3 && Cout != 4)) return false;
      bool isSemi = (code <= 97);
      if (!isSemi && H % 4 != 0) return false;  // planar needs H%4==0
      int bidx = (code % 2 == 0) ? 2 : 0;
      int uidx = isSemi ? ((code % 4 >= 2) ? 0 : 1) : ((code % 4 >= 2) ? 1 : 0);
      int _H = H, _W = W, _Cout = Cout, _bidx = bidx, _uidx = uidx, _N = N;
      if (isSemi)
        ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
          yuv420_to_bgr_kernel<true>(i.get(0), total, inU, outU, _N, _H, _W, _Cout, _bidx, _uidx);
        });
      else
        ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
          yuv420_to_bgr_kernel<false>(i.get(0), total, inU, outU, _N, _H, _W, _Cout, _bidx, _uidx);
        });
      return true;
    }

    // BGR -> YUV420 (codes 127-134, 140-147)
    if ((code >= 127 && code <= 134) || (code >= 140 && code <= 147)) {
      if ((Cin != 3 && Cin != 4) || Cout != 1) return false;
      bool isSemi = (code >= 140);
      if (!isSemi && H % 4 != 0) return false;
      int bidx = isSemi ? ((code % 2 == 0) ? 2 : 0) : ((code % 2 == 0) ? 0 : 2);
      int uidx = isSemi ? ((code % 4 < 2) ? 0 : 1) : ((code <= 130) ? 0 : 1);
      int _H = H, _W = W, _Cin = Cin, _bidx = bidx, _uidx = uidx, _N = N;
      if (isSemi)
        ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
          bgr_to_yuv420_kernel<true>(i.get(0), total, inU, outU, _N, _H, _W, _Cin, _bidx, _uidx);
        });
      else
        ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
          bgr_to_yuv420_kernel<false>(i.get(0), total, inU, outU, _N, _H, _W, _Cin, _bidx, _uidx);
        });
      return true;
    }
    return false;
  }

  // ---- Family F: YUV422 (8U only) ----
  bool forward_yuv422(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin, int Cout,
                      DType dtype, Code code) {
    if (dtype != DType::U8) return false;
    if (W % 2 != 0) return false;  // YUV422 needs W%2==0 (yuv width = 2W must be %4)
    if (Cin != 1) return false;
    bool is_gray = (code == 123 || code == 124);
    int bidx = is_gray ? 0 : ((code % 2 == 1) ? 2 : 0);
    int yidx = (code == 107 || code == 108 || code == 111 || code == 112 || code == 123) ? 1 : 0;  // UYVY
    int uidx = (code == 117 || code == 118 || code == 121 || code == 122) ? 2 : 0;                 // YVYU
    const uint8_t* inU = reinterpret_cast<const uint8_t*>(in);
    uint8_t* outU = reinterpret_cast<uint8_t*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W;

    if (is_gray) {
      if (Cout != 1) return false;
      int _yidx = yidx;
      ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
        yuv422_to_gray_kernel(i.get(0), total, inU, outU, _N, _H, _W, _yidx);
      });
      return true;
    }
    if (Cout != 3 && Cout != 4) return false;
    int _Cout = Cout, _bidx = bidx, _yidx = yidx, _uidx = uidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      yuv422_to_bgr_kernel(i.get(0), total, inU, outU, _N, _H, _W, _Cout, _bidx, _yidx, _uidx);
    });
    return true;
  }

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
  // dtype -> C++ type dispatch for Family A (pure copy/reorder, identical logic
  // for all types). 16F routes to uint16_t (16-bit storage, matching CV-CUDA).
  template <typename EltT>
  bool do_family_a(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin, int Cout,
                   int bidx) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _Cin = Cin, _Cout = Cout, _bidx = bidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      family_a_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _Cin, _Cout, _bidx);
    });
    return true;
  }

  bool launch_family_a(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin,
                       int Cout, int bidx, DType dtype) {
    switch (dtype) {
      case DType::U8: return do_family_a<uint8_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::S8: return do_family_a<int8_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::U16: return do_family_a<uint16_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::S16: return do_family_a<int16_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::F16: return do_family_a<uint16_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::S32: return do_family_a<int32_t>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::F32: return do_family_a<float>(q, in, out, N, H, W, Cin, Cout, bidx);
      case DType::F64: return do_family_a<double>(q, in, out, N, H, W, Cin, Cout, bidx);
      default: return false;
    }
  }

  // ---- Family B: BGR -> GRAY (8U/16U/32F) ----
  template <typename EltT>
  bool do_bgr_to_gray(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin, int bidx) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _Cin = Cin, _bidx = bidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      bgr_to_gray_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _Cin, _bidx);
    });
    return true;
  }
  bool launch_bgr_to_gray(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cin,
                          int bidx, DType dtype) {
    switch (dtype) {
      case DType::U8: return do_bgr_to_gray<uint8_t>(q, in, out, N, H, W, Cin, bidx);
      case DType::U16: return do_bgr_to_gray<uint16_t>(q, in, out, N, H, W, Cin, bidx);
      case DType::F32: return do_bgr_to_gray<float>(q, in, out, N, H, W, Cin, bidx);
      default: return false;
    }
  }

  // ---- Family B: GRAY -> BGR (all dtypes) ----
  template <typename EltT>
  bool do_gray_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cout) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _Cout = Cout;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      gray_to_bgr_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _Cout);
    });
    return true;
  }
  bool launch_gray_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int Cout,
                          DType dtype) {
    switch (dtype) {
      case DType::U8: return do_gray_to_bgr<uint8_t>(q, in, out, N, H, W, Cout);
      case DType::S8: return do_gray_to_bgr<int8_t>(q, in, out, N, H, W, Cout);
      case DType::U16: return do_gray_to_bgr<uint16_t>(q, in, out, N, H, W, Cout);
      case DType::S16: return do_gray_to_bgr<int16_t>(q, in, out, N, H, W, Cout);
      case DType::F16: return do_gray_to_bgr<uint16_t>(q, in, out, N, H, W, Cout);
      case DType::S32: return do_gray_to_bgr<int32_t>(q, in, out, N, H, W, Cout);
      case DType::F32: return do_gray_to_bgr<float>(q, in, out, N, H, W, Cout);
      case DType::F64: return do_gray_to_bgr<double>(q, in, out, N, H, W, Cout);
      default: return false;
    }
  }

  // ---- Family C: packed YUV (8U/16U/32F) ----
  template <typename EltT>
  bool do_bgr_to_yuv(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _bidx = bidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      bgr_to_yuv_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _bidx);
    });
    return true;
  }
  template <typename EltT>
  bool do_yuv_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _bidx = bidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      yuv_to_bgr_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _bidx);
    });
    return true;
  }
  bool launch_bgr_to_yuv(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                         DType dtype) {
    switch (dtype) {
      case DType::U8: return do_bgr_to_yuv<uint8_t>(q, in, out, N, H, W, bidx);
      case DType::U16: return do_bgr_to_yuv<uint16_t>(q, in, out, N, H, W, bidx);
      case DType::F32: return do_bgr_to_yuv<float>(q, in, out, N, H, W, bidx);
      default: return false;
    }
  }
  bool launch_yuv_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                         DType dtype) {
    switch (dtype) {
      case DType::U8: return do_yuv_to_bgr<uint8_t>(q, in, out, N, H, W, bidx);
      case DType::U16: return do_yuv_to_bgr<uint16_t>(q, in, out, N, H, W, bidx);
      case DType::F32: return do_yuv_to_bgr<float>(q, in, out, N, H, W, bidx);
      default: return false;
    }
  }

  // ---- Family D: HSV (8U/32F) ----
  template <typename EltT>
  bool do_bgr_to_hsv(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                     bool isFullRange) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _bidx = bidx;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      bgr_to_hsv_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _bidx, isFullRange);
    });
    return true;
  }
  template <typename EltT>
  bool do_hsv_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                     bool isFullRange, int Cout) {
    const EltT* inT = reinterpret_cast<const EltT*>(in);
    EltT* outT = reinterpret_cast<EltT*>(out);
    size_t total = (size_t)N * H * W;
    int _N = N, _H = H, _W = W, _bidx = bidx, _Cout = Cout;
    ev_ = q.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> i) {
      hsv_to_bgr_kernel<EltT>(i.get(0), total, inT, outT, _N, _H, _W, _bidx, isFullRange, _Cout);
    });
    return true;
  }
  bool launch_bgr_to_hsv(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                         bool isFullRange, DType dtype) {
    switch (dtype) {
      case DType::U8: return do_bgr_to_hsv<uint8_t>(q, in, out, N, H, W, bidx, isFullRange);
      case DType::F32: return do_bgr_to_hsv<float>(q, in, out, N, H, W, bidx, isFullRange);
      default: return false;
    }
  }
  bool launch_hsv_to_bgr(sycl::queue& q, const char* in, char* out, int N, int H, int W, int bidx,
                         bool isFullRange, int Cout, DType dtype) {
    switch (dtype) {
      case DType::U8: return do_hsv_to_bgr<uint8_t>(q, in, out, N, H, W, bidx, isFullRange, Cout);
      case DType::F32: return do_hsv_to_bgr<float>(q, in, out, N, H, W, bidx, isFullRange, Cout);
      default: return false;
    }
  }

  sycl::queue queue_{make_internal_queue()};
  bool profiling_enabled_;
  sycl::event ev_;
};

inline std::shared_ptr<CvtColor> create_cvt() { return std::make_shared<CvtColorImplement>(); }

}  // namespace cvt
}  // namespace cvcuda

#endif  // CVCUDA_OPS_CVTCOLOR_HPP
