// CPU gold reference for Rotate, written to match the SYCL kernel's float math
// exactly (so GPU-vs-gold divergence isolates to Intel-GPU FMA-contraction ULP
// noise — LINEAR/CUBIC use only mul/add, no div/sqrt, so the FP-division quirk
// does NOT apply). Operates on flat NHWC arrays with the same layout as the
// kernel. Computes the affine coeffs the same way as rotate.cpp (host double
// cos/sin), so kernel and gold share bit-identical coeffs.
//
// Verification caveat (see operator-port-verification-caveat memory): CV-CUDA's
// rotate.cu ships no independent CPU test gold, so this gold is my own port of
// the .cu kernel + InterpolationWrap math — a self-consistency check, NOT a
// comparison to NVIDIA's output. The strongest independent check is the
// line-by-line diff of the kernel vs rotate.cu + InterpolationWrap.hpp.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __ROTATE_GOLD_HPP__
#define __ROTATE_GOLD_HPP__

#include <cmath>
#include <cstdint>
#include <cstddef>
#include "../rotate.hpp"
#include "../rotate_helpers.hpp"

namespace rotate_gold {

using cvcuda::rotate::Interp;
using cvcuda::rotate::TypeMax;
using cvcuda::rotate::TypeMin;
using cvcuda::rotate::border_replicate;
using cvcuda::rotate::cubic_coeffs;
using cvcuda::rotate::nhwc_ptr;

// SaturateCast<T>(float): round-to-nearest-even (std::nearbyint, matches PTX
// cvt.rni / sycl::rint) + clamp [TypeMin, TypeMax] for integral T; identity for
// float T. Matches the kernel's sat_cast.
template <typename T>
inline T gold_sat(float u) {
  if constexpr (std::is_integral_v<T>) {
    float r = std::nearbyint(u);
    float mn = (float)TypeMin<T>();
    float mx = (float)TypeMax<T>();
    if (r < mn) r = mn;
    if (r > mx) r = mx;
    return (T)r;
  } else {
    return (T)u;
  }
}

// Compute the 6 affine coeffs (host double) — identical to rotate.cpp.
inline void gold_coeffs(double angleDeg, double shiftX, double shiftY, double a[6]) {
  static const double PI = 3.1415926535897932384626433832795;
  double rad = angleDeg * PI / 180.0;
  double c = std::cos(rad), s = std::sin(rad);
  a[0] = c; a[1] = s; a[2] = shiftX; a[3] = -s; a[4] = c; a[5] = shiftY;
}

// Rotate gold. out is zero-filled for out-of-range dest pixels (exposed corners),
// matching the kernel's else-branch. Sampling uses BORDER_REPLICATE.
template <typename T>
void gold_rotate(T* dst, const T* src, int N, int Hin, int Win, int Hout, int Wout, int C,
                 double angleDeg, double shiftX, double shiftY, Interp interp) {
  double a[6];
  gold_coeffs(angleDeg, shiftX, shiftY, a);
  const double c = a[0], s = a[1];

  for (int n = 0; n < N; ++n)
    for (int y = 0; y < Hout; ++y)
      for (int x = 0; x < Wout; ++x) {
        const double dx = (double)x - a[2];
        const double dy = (double)y - a[5];
        const float srcX = (float)(dx * c + dy * (-a[1]));   // dx*c - dy*s
        const float srcY = (float)(dx * (-a[3]) + dy * c);   // dx*s + dy*c

        T* dp = nhwc_ptr(dst, n, y, x, Hout, Wout, C);

        if (!(srcX > -0.5f && srcX < (float)Win && srcY > -0.5f && srcY < (float)Hin)) {
          for (int cch = 0; cch < C; ++cch) dp[cch] = (T)0;
          continue;
        }

        if (interp == Interp::NEAREST) {
          int ix = (int)std::floor(srcX + 0.5f);
          int iy = (int)std::floor(srcY + 0.5f);
          ix = border_replicate(ix, Win);
          iy = border_replicate(iy, Hin);
          const T* sp = nhwc_ptr(src, n, iy, ix, Hin, Win, C);
          for (int cch = 0; cch < C; ++cch) dp[cch] = sp[cch];
        } else if (interp == Interp::LINEAR) {
          int x1 = (int)std::floor(srcX);
          int x2 = x1 + 1;
          int y1 = (int)std::floor(srcY);
          int y2 = y1 + 1;
          float fx = srcX - (float)x1;
          float fy = srcY - (float)y1;
          float w00 = ((float)x2 - srcX) * ((float)y2 - srcY);
          float w10 = (fx) * ((float)y2 - srcY);
          float w01 = ((float)x2 - srcX) * (fy);
          float w11 = (fx) * (fy);
          const T* s00 = nhwc_ptr(src, n, border_replicate(y1, Hin), border_replicate(x1, Win), Hin, Win, C);
          const T* s10 = nhwc_ptr(src, n, border_replicate(y1, Hin), border_replicate(x2, Win), Hin, Win, C);
          const T* s01 = nhwc_ptr(src, n, border_replicate(y2, Hin), border_replicate(x1, Win), Hin, Win, C);
          const T* s11 = nhwc_ptr(src, n, border_replicate(y2, Hin), border_replicate(x2, Win), Hin, Win, C);
          for (int cch = 0; cch < C; ++cch) {
            float v = (float)s00[cch] * w00 + (float)s10[cch] * w10 + (float)s01[cch] * w01 +
                      (float)s11[cch] * w11;
            dp[cch] = gold_sat<T>(v);
          }
        } else {  // CUBIC
          int ix = (int)std::floor(srcX);
          int iy = (int)std::floor(srcY);
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
                sum += (float)nhwc_ptr(src, n, syi, sxi, Hin, Win, C)[cch] * w;
              }
            }
            dp[cch] = gold_sat<T>(sum);
          }
        }
      }
}

// Pre-saturate float result for every output element (NHWC, same layout as dst).
// Used by the real test to classify LINEAR/CUBIC FP-boundary mismatches on
// integral types: a divergence is a boundary case if this pre-round value sits
// within BOUND_TOL of a half-integer (rint flip point) or a dtype clamp bound.
template <typename T>
void gold_rotate_res(const T* src, int N, int Hin, int Win, int Hout, int Wout, int C,
                     double angleDeg, double shiftX, double shiftY, Interp interp, float* res) {
  double a[6];
  gold_coeffs(angleDeg, shiftX, shiftY, a);
  const double c = a[0], s = a[1];
  for (int n = 0; n < N; ++n)
    for (int y = 0; y < Hout; ++y)
      for (int x = 0; x < Wout; ++x) {
        const double dx = (double)x - a[2];
        const double dy = (double)y - a[5];
        const float srcX = (float)(dx * c + dy * (-a[1]));
        const float srcY = (float)(dx * (-a[3]) + dy * c);
        float* rp = res + ((static_cast<size_t>(n) * Hout + y) * Wout + x) * C;
        if (!(srcX > -0.5f && srcX < (float)Win && srcY > -0.5f && srcY < (float)Hin)) {
          for (int cch = 0; cch < C; ++cch) rp[cch] = 0.f;
          continue;
        }
        if (interp == Interp::NEAREST) {
          // nearest picks an exact source pixel -> pre-saturate = that value.
          int ix = (int)std::floor(srcX + 0.5f);
          int iy = (int)std::floor(srcY + 0.5f);
          const T* sp = nhwc_ptr(src, n, border_replicate(iy, Hin), border_replicate(ix, Win), Hin, Win, C);
          for (int cch = 0; cch < C; ++cch) rp[cch] = (float)sp[cch];
        } else if (interp == Interp::LINEAR) {
          int x1 = (int)std::floor(srcX), x2 = x1 + 1;
          int y1 = (int)std::floor(srcY), y2 = y1 + 1;
          float fx = srcX - (float)x1, fy = srcY - (float)y1;
          float w00 = ((float)x2 - srcX) * ((float)y2 - srcY);
          float w10 = (fx) * ((float)y2 - srcY);
          float w01 = ((float)x2 - srcX) * (fy);
          float w11 = (fx) * (fy);
          const T* s00 = nhwc_ptr(src, n, border_replicate(y1, Hin), border_replicate(x1, Win), Hin, Win, C);
          const T* s10 = nhwc_ptr(src, n, border_replicate(y1, Hin), border_replicate(x2, Win), Hin, Win, C);
          const T* s01 = nhwc_ptr(src, n, border_replicate(y2, Hin), border_replicate(x1, Win), Hin, Win, C);
          const T* s11 = nhwc_ptr(src, n, border_replicate(y2, Hin), border_replicate(x2, Win), Hin, Win, C);
          for (int cch = 0; cch < C; ++cch)
            rp[cch] = (float)s00[cch] * w00 + (float)s10[cch] * w10 + (float)s01[cch] * w01 +
                      (float)s11[cch] * w11;
        } else {
          int ix = (int)std::floor(srcX), iy = (int)std::floor(srcY);
          float wx[4], wy[4];
          cubic_coeffs(srcX - (float)ix, wx[0], wx[1], wx[2], wx[3]);
          cubic_coeffs(srcY - (float)iy, wy[0], wy[1], wy[2], wy[3]);
          for (int cch = 0; cch < C; ++cch) {
            float sum = 0.f;
            for (int cy = -1; cy <= 2; ++cy) {
              int syi = border_replicate(iy + cy, Hin);
              for (int cx = -1; cx <= 2; ++cx) {
                int sxi = border_replicate(ix + cx, Win);
                sum += (float)nhwc_ptr(src, n, syi, sxi, Hin, Win, C)[cch] * (wx[cx + 1] * wy[cy + 1]);
              }
            }
            rp[cch] = sum;
          }
        }
      }
}

}  // namespace rotate_gold

#endif  // __ROTATE_GOLD_HPP__
