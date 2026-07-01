// CPU gold reference for CvtColor, ported from CV-CUDA's
// tests/cvcuda/system/CvtColorUtils.cpp. Added per stage. Stage 0: Family A
// (convertRGBtoBGR / changeAlpha). Operates on flat NHWC arrays.
//
// Verification caveat (see operator-port-verification-caveat memory): this gold
// is ported from CV-CUDA's own independent CPU reference (a separate impl from
// the .cu kernel), so GPU-vs-gold agreement is evidence both faithfully track
// CV-CUDA's two originals — but not a direct comparison to NVIDIA's output.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __CVT_GOLD_HPP__
#define __CVT_GOLD_HPP__

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "../cvt_helpers.hpp"

namespace cvt_gold {

using cvcuda::cvt::AlphaVal;

// Double coefficients (CV-CUDA gold uses double Red2Y/Grn2Y/Blu2Y = 0.299/0.587/0.114).
constexpr double Red2Y_d = 0.299;
constexpr double Grn2Y_d = 0.587;
constexpr double Blu2Y_d = 0.114;

// PAL YUV chroma coeffs (CvtColorUtils.cpp L60-77).
constexpr double Blu2U_PAL = 0.492;
constexpr double Red2V_PAL = 0.877;
constexpr double U2Blu = 2.032;
constexpr double U2Grn = -0.395;
constexpr double V2Grn = -0.581;
constexpr double V2Red = 1.140;

// NV12 / YUV420 ITU-R BT.601 double coeffs (CvtColorUtils.cpp L80-108).
constexpr double R2Y_NV12 = 0.255785, G2Y_NV12 = 0.502160, B2Y_NV12 = 0.097523;
constexpr double R2U_NV12 = -0.147644, G2U_NV12 = -0.289856, B2U_NV12 = 0.4375;
constexpr double R2V_NV12 = 0.4375, G2V_NV12 = -0.366352, B2V_NV12 = -0.071148;
constexpr double Y2R_NV12 = 1.16895, U2R_NV12 = 0.0, V2R_NV12 = 1.60229;
constexpr double Y2G_NV12 = 1.16895, U2G_NV12 = -0.3933, V2G_NV12 = -0.81616;
constexpr double Y2B_NV12 = 1.16895, U2B_NV12 = 2.02514, V2B_NV12 = 0.0;
constexpr double Add2Y_NV12 = 16.0, Add2U_NV12 = 128.0, Add2V_NV12 = 128.0;

// SaturateCast<T>(double): round-to-nearest-even + clamp for integral T;
// identity cast for float T. Matches CV-CUDA gold's cuda::SaturateCast<T>.
template <typename T>
inline T gold_sat(double u) {
  if constexpr (std::is_integral_v<T>) {
    double r = std::nearbyint(u);
    double mn = 0.0;
    double mx = (double)cvcuda::cvt::TypeMax<T>();
    if (r < mn) r = mn;
    if (r > mx) r = mx;
    return (T)r;
  } else {
    return (T)u;
  }
}

template <typename T>
inline double yuv_delta() {
  // max/2 + (int?1:0): uchar 128, ushort 32768, float 0.5
  if constexpr (std::is_floating_point_v<T>)
    return 0.5;
  else
    return (double)(cvcuda::cvt::TypeMax<T>() / 2 + 1);
}

// Family A: channel swap / alpha add-drop. Ported from convertRGBtoBGR<T,AlphaOnly>.
// alpha_only=true (codes 0,1): copy [0,1,2] (+alpha). false (codes 2,3,4,5):
// reverse [2,1,0] (+alpha). Alpha added when dch==4: src alpha if sch==4 else Alpha.
template <typename T>
void gold_family_a(T* dst, const T* src, size_t numPixels, int sch, int dch, bool alpha_only) {
  for (size_t i = 0; i < numPixels; ++i) {
    if (alpha_only) {
      *dst++ = src[0];
      *dst++ = src[1];
      *dst++ = src[2];
    } else {
      *dst++ = src[2];
      *dst++ = src[1];
      *dst++ = src[0];
    }
    if (dch == 4) *dst++ = (sch == 4) ? src[3] : AlphaVal<T>();
    src += sch;
  }
}

// Family B: BGR/RGB -> GRAY. Ported from convertRGBtoGray (float coeffs +
// static_cast truncate). bidx routes B/R (bidx==0 => src is BGR). The GPU int
// path uses fixed-point CV_DESCALE (rounds), so U8/U16 differ from this gold by
// <=1 / <=2 — caller must use those tolerances.
template <typename T>
void gold_bgr_to_gray(T* dst, const T* src, size_t numPixels, int sch, int bidx) {
  for (size_t i = 0; i < numPixels; ++i, src += sch) {
    double B = (double)(bidx == 0 ? src[0] : src[2]);
    double G = (double)src[1];
    double R = (double)(bidx == 0 ? src[2] : src[0]);
    *dst++ = static_cast<T>(Blu2Y_d * B + Grn2Y_d * G + Red2Y_d * R);
  }
}

// Family B: GRAY -> BGR/BGRA. Ported from convertGrayToRGB: replicate gray to
// all output channels (alpha = gray). Pure replicate -> exact match.
template <typename T>
void gold_gray_to_bgr(T* dst, const T* src, size_t numPixels, int dch) {
  for (size_t i = 0; i < numPixels; ++i, ++src) {
    T v = *src;
    for (int c = 0; c < dch; ++c) *dst++ = v;
  }
}

// Family C: BGR/RGB -> YUV (PAL). Ported from convertRGBtoYUV_PAL (double math
// + SaturateCast round). bidx==0 => src is BGR. Output [Y, Cb, Cr].
template <typename T>
void gold_bgr_to_yuv(T* dst, const T* src, size_t numPixels, int bidx) {
  double delta = yuv_delta<T>();
  for (size_t i = 0; i < numPixels; ++i, src += 3) {
    double red = (double)(bidx == 0 ? src[2] : src[0]);
    double grn = (double)src[1];
    double blu = (double)(bidx == 0 ? src[0] : src[2]);
    double Y = Red2Y_d * red + Grn2Y_d * grn + Blu2Y_d * blu;
    *dst++ = gold_sat<T>(Y);
    *dst++ = gold_sat<T>(Blu2U_PAL * (blu - Y) + delta);
    *dst++ = gold_sat<T>(Red2V_PAL * (red - Y) + delta);
  }
}

// Family C: YUV -> BGR/RGB (PAL). Ported from convertYUVtoRGB_PAL (double math
// + SaturateCast round). bidx==0 => output BGR ([B,G,R]); bidx==2 => RGB.
template <typename T>
void gold_yuv_to_bgr(T* dst, const T* src, size_t numPixels, int bidx) {
  double delta = yuv_delta<T>();
  for (size_t i = 0; i < numPixels; ++i, src += 3) {
    double Y = (double)src[0];
    double U = (double)src[1] - delta;
    double V = (double)src[2] - delta;
    double red = Y + V * V2Red;
    double grn = Y + U * U2Grn + V * V2Grn;
    double blu = Y + U * U2Blu;
    if (bidx == 0) {  // BGR
      *dst++ = gold_sat<T>(blu);
      *dst++ = gold_sat<T>(grn);
      *dst++ = gold_sat<T>(red);
    } else {  // RGB
      *dst++ = gold_sat<T>(red);
      *dst++ = gold_sat<T>(grn);
      *dst++ = gold_sat<T>(blu);
    }
  }
}

// Family D: BGR/RGB -> HSV. Ported from convertRGBtoHSV<T,FullRange>. bidx==0
// => src is BGR. Output [H,S,V]. Hue is circular mod `range`.
template <typename T, bool FullRange>
void gold_bgr_to_hsv(T* dst, const T* src, size_t numPixels, int bidx) {
  constexpr double range = (sizeof(T) > 1) ? 360.0 : (FullRange ? 256.0 : 180.0);
  constexpr double scale = range / 360.0;
  constexpr double norm  = std::is_floating_point_v<T> ? 1.0 : (double)cvcuda::cvt::TypeMax<T>();
  constexpr double rnd   = std::is_floating_point_v<T> ? 0.0 : 0.5;
  for (size_t i = 0; i < numPixels; ++i, src += 3) {
    double R = (double)src[0] / norm, G = (double)src[1] / norm, B = (double)src[2] / norm;
    if (bidx == 0) std::swap(R, B);
    double Vmin = std::min(R, std::min(G, B));
    double V    = std::max(R, std::max(G, B));
    double diff = (double)(V - Vmin);
    double S = V > DBL_EPSILON ? diff / V : 0.0;
    double H = 0.0;
    if (diff > DBL_EPSILON) {
      diff = 60.0 / diff;
      if (V == R) H = (G - B) * diff;
      else if (V == G) H = (B - R) * diff + 120.0;
      else H = (R - G) * diff + 240.0;
    }
    H *= scale; S *= norm; V *= norm;
    H += rnd;
    if (H >= range) H -= range;
    else if (H < 0.0) H += range;
    H -= rnd;
    *dst++ = (T)(H + rnd);
    *dst++ = (T)(S + rnd);
    *dst++ = (T)(V + rnd);
  }
}

// Family D: HSV -> BGR/RGB. Ported from convertHSVtoRGB<T,FullRange>. bidx==0
// => output BGR. dch 3 or 4 (alpha = AlphaVal). Output is RGB (non-circular).
template <typename T, bool FullRange>
void gold_hsv_to_bgr(T* dst, const T* src, size_t numPixels, int bidx, int dch) {
  constexpr double range = (sizeof(T) > 1) ? 360.0 : (FullRange ? 256.0 : 180.0);
  constexpr double scale = 6.0 / range;
  constexpr double norm  = std::is_floating_point_v<T> ? 1.0 : (double)cvcuda::cvt::TypeMax<T>();
  constexpr double rnd   = std::is_floating_point_v<T> ? 0.0 : 0.5;
  constexpr unsigned int mapR[6] = {0, 2, 1, 1, 3, 0};
  constexpr unsigned int mapG[6] = {3, 0, 0, 2, 1, 1};
  constexpr unsigned int mapB[6] = {1, 1, 3, 0, 0, 2};
  for (size_t i = 0; i < numPixels; ++i, src += 3) {
    double H = (double)src[0] * scale, S = (double)src[1] / norm, V = (double)src[2] / norm;
    int idx = (int)std::floor(H);
    H -= idx;
    idx %= 6;
    if (idx < 0) idx += 6;
    double val[4] = {V, V * (1 - S), V * (1 - S * H), V * (1 - S * (1 - H))};
    unsigned int r = mapR[idx], g = mapG[idx], b = mapB[idx];
    if (bidx == 0) std::swap(r, b);
    *dst++ = (T)(val[r] * norm + rnd);
    *dst++ = (T)(val[g] * norm + rnd);
    *dst++ = (T)(val[b] * norm + rnd);
    if (dch == 4) *dst++ = cvcuda::cvt::AlphaVal<T>();
  }
}

// Family E: YUV420 (8U). Ported from CvtColorUtils convertRGBtoNV12 / convertRGBtoYUV_420
// (BGR->YUV420) and convertNV12toRGB / convertYUVtoRGB_420 (YUV420->BGR), plus
// convertYUVtoGray_420. bgr=(bidx==0); yvu=(uidx==1). Kernel uses ITUR int coeffs,
// gold uses NV12 double coeffs -> differ by <=2 (tol 2 YUV->BGR, 1 BGR->YUV, 0 gray).
// Layouts (semi-planar NV12/NV21 + planar I420/YV12, 4:2:0) match the kernel exactly.

template <typename T>
void gold_bgr_to_yuv420_nv12(T* dst, const T* src, int W, int H, int N, int Cin, int bidx, int uidx) {
  size_t imgPix = (size_t)H * W;
  bool bgr = (bidx == 0), yvu = (uidx == 1);
  size_t incrSrc = imgPix * Cin, incrDst = imgPix * 3 / 2;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += incrDst) {
    T* y = dst;
    T* uv = dst + imgPix;
    const T* rgb = src;
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w, rgb += Cin) {
        T R = rgb[0], G = rgb[1], B = rgb[2];
        if (bgr) std::swap(R, B);
        *y++ = gold_sat<T>(R2Y_NV12 * R + G2Y_NV12 * G + B2Y_NV12 * B + Add2Y_NV12);
        if ((w & 1) == 0 && (h & 1) == 0) {
          double U = R2U_NV12 * R + G2U_NV12 * G + B2U_NV12 * B + Add2U_NV12;
          double V = R2V_NV12 * R + G2V_NV12 * G + B2V_NV12 * B + Add2V_NV12;
          if (yvu) std::swap(U, V);
          *uv++ = gold_sat<T>(U);
          *uv++ = gold_sat<T>(V);
        }
      }
  }
}

template <typename T>
void gold_bgr_to_yuv420_planar(T* dst, const T* src, int W, int H, int N, int Cin, int bidx, int uidx) {
  size_t imgPix = (size_t)H * W;
  bool bgr = (bidx == 0), yvu = (uidx == 1);
  size_t incrSrc = imgPix * Cin, incrDst = imgPix * 3 / 2;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += incrDst) {
    T* y = dst;
    T* u = y + imgPix;
    T* v = u + imgPix / 4;
    if (yvu) std::swap(u, v);
    const T* rgb = src;
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w, rgb += Cin) {
        T R = rgb[0], G = rgb[1], B = rgb[2];
        if (bgr) std::swap(R, B);
        *y++ = gold_sat<T>(R2Y_NV12 * R + G2Y_NV12 * G + B2Y_NV12 * B + Add2Y_NV12);
        if ((w & 1) == 0 && (h & 1) == 0) {
          double U = R2U_NV12 * R + G2U_NV12 * G + B2U_NV12 * B + Add2U_NV12;
          double V = R2V_NV12 * R + G2V_NV12 * G + B2V_NV12 * B + Add2V_NV12;
          *u++ = gold_sat<T>(U);
          *v++ = gold_sat<T>(V);
        }
      }
  }
}

template <typename T>
void gold_yuv420_to_bgr_nv12(T* dst, const T* src, int W, int H, int N, int Cout, int bidx, int uidx) {
  size_t imgPix = (size_t)H * W;
  bool rgba = (Cout == 4), bgr = (bidx == 0), yvu = (uidx == 1);
  size_t incrSrc = imgPix * 3 / 2, incrDst = imgPix * Cout;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += incrDst) {
    T* rgb = dst;
    const T* y = src;
    for (int h = 0; h < H; ++h) {
      const T* uv = src + imgPix + (h >> 1) * W;
      for (int w = 0; w < W; ++w) {
        double Y = *y++, U = uv[0], V = uv[1];
        if (yvu) std::swap(U, V);
        Y -= Add2Y_NV12; U -= Add2U_NV12; V -= Add2V_NV12;
        if (Y < 0.0) Y = 0.0;
        T R = gold_sat<T>(Y2R_NV12 * Y + U2R_NV12 * U + V2R_NV12 * V);
        T G = gold_sat<T>(Y2G_NV12 * Y + U2G_NV12 * U + V2G_NV12 * V);
        T B = gold_sat<T>(Y2B_NV12 * Y + U2B_NV12 * U + V2B_NV12 * V);
        if (bgr) std::swap(R, B);
        *rgb++ = R; *rgb++ = G; *rgb++ = B;
        if (rgba) *rgb++ = cvcuda::cvt::AlphaVal<T>();
        if (w & 1) uv += 2;
      }
    }
  }
}

template <typename T>
void gold_yuv420_to_bgr_planar(T* dst, const T* src, int W, int H, int N, int Cout, int bidx, int uidx) {
  size_t imgPix = (size_t)H * W;
  bool rgba = (Cout == 4), bgr = (bidx == 0), yvu = (uidx == 1);
  size_t incrSrc = imgPix * 3 / 2, incrDst = imgPix * Cout;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += incrDst) {
    T* rgb = dst;
    const T* y = src;
    for (int h = 0; h < H; ++h) {
      const T* u = src + imgPix + (h / 4) * W + ((h / 2) & 1) * (W / 2);
      const T* v = u + imgPix / 4;
      if (yvu) std::swap(u, v);
      for (int w = 0; w < W; ++w) {
        double Y = *y++, U = *u, V = *v;
        Y -= Add2Y_NV12; U -= Add2U_NV12; V -= Add2V_NV12;
        if (Y < 0.0) Y = 0.0;
        T R = gold_sat<T>(Y2R_NV12 * Y + U2R_NV12 * U + V2R_NV12 * V);
        T G = gold_sat<T>(Y2G_NV12 * Y + U2G_NV12 * U + V2G_NV12 * V);
        T B = gold_sat<T>(Y2B_NV12 * Y + U2B_NV12 * U + V2B_NV12 * V);
        if (bgr) std::swap(R, B);
        *rgb++ = R; *rgb++ = G; *rgb++ = B;
        if (rgba) *rgb++ = cvcuda::cvt::AlphaVal<T>();
        u += (w & 1);
        v += (w & 1);
      }
    }
  }
}

template <typename T>
void gold_yuv420_to_gray(T* dst, const T* src, int W, int H, int N) {
  size_t imgPix = (size_t)H * W;
  size_t incrSrc = imgPix * 3 / 2;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += imgPix)
    std::memcpy(dst, src, imgPix * sizeof(T));
}

// Family F: YUV422 (8U). Ported from convertYUVtoRGB_422 / convertYUVtoGray_422.
// 4:2:2: yuv_w = 2W (4 elems per 2 dst pixels). yidx=0 YUY2/YVYU (Y-first),
// yidx=1 UYVY. uidx=2 YVYU (yvu swaps U,V). Gold uses NV12 double coeffs ->
// kernel ITUR int differs by <=2 (tol 2). Gray exact (tol 0).
template <typename T>
void gold_yuv422_to_bgr(T* dst, const T* src, int W, int H, int N, int Cout, int bidx, int yidx, int uidx) {
  unsigned idx0 = yidx, idx1 = yidx + 2, idxU = (yidx ^ 1), idxV = (yidx ^ 1) + 2;
  bool yvu = (uidx == 2), bgr = (bidx == 0), rgba = (Cout == 4);
  size_t imgPix = (size_t)H * W;
  size_t incrSrc = imgPix * 2, incrDst = imgPix * Cout;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += incrDst) {
    T* rgb = dst;
    const T* img = src;
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; w += 2, img += 4) {
        double U = img[idxU], V = img[idxV], Y0 = img[idx0], Y1 = img[idx1];
        if (yvu) std::swap(U, V);
        Y0 -= Add2Y_NV12; Y1 -= Add2Y_NV12; U -= Add2U_NV12; V -= Add2V_NV12;
        if (Y0 < 0.0) Y0 = 0.0;
        if (Y1 < 0.0) Y1 = 0.0;
        for (int which = 0; which < 2; ++which) {
          double Y = (which == 0) ? Y0 : Y1;
          T R = gold_sat<T>(Y2R_NV12 * Y + U2R_NV12 * U + V2R_NV12 * V);
          T G = gold_sat<T>(Y2G_NV12 * Y + U2G_NV12 * U + V2G_NV12 * V);
          T B = gold_sat<T>(Y2B_NV12 * Y + U2B_NV12 * U + V2B_NV12 * V);
          if (bgr) std::swap(R, B);
          *rgb++ = R; *rgb++ = G; *rgb++ = B;
          if (rgba) *rgb++ = cvcuda::cvt::AlphaVal<T>();
        }
      }
    }
  }
}

template <typename T>
void gold_yuv422_to_gray(T* dst, const T* src, int W, int H, int N, int yidx) {
  unsigned idx0 = yidx;
  size_t imgPix = (size_t)H * W;
  size_t incrSrc = imgPix * 2;
  for (int n = 0; n < N; ++n, src += incrSrc, dst += imgPix) {
    T* g = dst;
    const T* img = src;
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; w += 2, img += 4) {
        *g++ = img[idx0];
        *g++ = img[idx0 + 2];
      }
    }
  }
}

}  // namespace cvt_gold

#endif  // __CVT_GOLD_HPP__
