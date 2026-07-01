// CPU gold reference for Normalize, ported from CV-CUDA's
// tests/cvcuda/system/TestOpNormalize.cpp Normalize() helper (FT=float math +
// std::rint + manual clamp). Operates on flat NHWC arrays with the same layout
// as the kernel.
//
// Verification caveat (see operator-port-verification-caveat memory): this gold
// is ported from CV-CUDA's own independent CPU reference (a separate impl from
// the .cu kernel), so GPU-vs-gold agreement is evidence both faithfully track
// CV-CUDA's two originals — but not a direct comparison to NVIDIA's output.
// The stddev path has one float division (1/sqrt) — Intel GPU uses reciprocal
// approximation there (see intel-gpu-fp-quirk), so GPU vs this (IEEE) gold may
// diverge at ULP boundaries; the real test classifies such mismatches.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __NORMALIZE_GOLD_HPP__
#define __NORMALIZE_GOLD_HPP__

#include <cmath>
#include <cstdint>
#include <cstddef>
#include "../normalize.hpp"
#include "../normalize_helpers.hpp"

namespace normalize_gold {

using cvcuda::normalize::Shape;
using cvcuda::normalize::TypeMax;
using cvcuda::normalize::TypeMin;

// SaturateCast<T>(float): round-to-nearest-even (std::nearbyint, matches PTX
// cvt.rni / sycl::rint) + clamp [TypeMin, TypeMax] for integral T; identity for
// float T (CV-CUDA F32 path). Matches CV-CUDA gold's std::rint + clamp.
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

// Normalize gold. out = SaturateCast((src - base) * mul * global_scale + shift),
// mul = scale (plain) or 1/sqrt(scale^2 + epsilon) (stddev). base/scale
// broadcast per-axis exactly like the kernel.
template <typename T>
void gold_normalize(T* dst, const T* src, const float* base, const float* scale, int N, int H, int W,
                    int C, Shape bS, Shape sS, float gscale, float shift, float eps, bool stddev) {
  for (int n = 0; n < N; ++n)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int bn = bS.N == 1 ? 0 : n, by = bS.H == 1 ? 0 : y, bx = bS.W == 1 ? 0 : x;
        int sn = sS.N == 1 ? 0 : n, sy = sS.H == 1 ? 0 : y, sx = sS.W == 1 ? 0 : x;
        const T* sp = src + ((static_cast<size_t>(n) * H + y) * W + x) * C;
        T* dp        = dst + ((static_cast<size_t>(n) * H + y) * W + x) * C;
        size_t base_off  = ((static_cast<size_t>(bn) * bS.H + by) * bS.W + bx) * bS.C;
        size_t scale_off = ((static_cast<size_t>(sn) * sS.H + sy) * sS.W + sx) * sS.C;
        for (int c = 0; c < C; ++c) {
          float sv = (float)sp[c];
          float bv = base[base_off + (bS.C == 1 ? 0 : c)];
          float sc = scale[scale_off + (sS.C == 1 ? 0 : c)];
          float mul = stddev ? (1.0f / std::sqrt(sc * sc + eps)) : sc;
          float res = (sv - bv) * mul * gscale + shift;
          dp[c]     = gold_sat<T>(res);
        }
      }
}

// Pre-saturate float result for every output element (NHWC, same layout as
// dst). Used by the real test to classify stddev FP-boundary mismatches: a
// divergence is a boundary case if this pre-round value sits within BOUND_TOL
// of a half-integer (rint flip point) or a dtype clamp bound.
template <typename T>
void gold_normalize_res(const T* src, const float* base, const float* scale, int N, int H, int W,
                        int C, Shape bS, Shape sS, float gscale, float shift, float eps, bool stddev,
                        float* res) {
  for (int n = 0; n < N; ++n)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int bn = bS.N == 1 ? 0 : n, by = bS.H == 1 ? 0 : y, bx = bS.W == 1 ? 0 : x;
        int sn = sS.N == 1 ? 0 : n, sy = sS.H == 1 ? 0 : y, sx = sS.W == 1 ? 0 : x;
        const T* sp = src + ((static_cast<size_t>(n) * H + y) * W + x) * C;
        float* rp     = res + ((static_cast<size_t>(n) * H + y) * W + x) * C;
        size_t base_off  = ((static_cast<size_t>(bn) * bS.H + by) * bS.W + bx) * bS.C;
        size_t scale_off = ((static_cast<size_t>(sn) * sS.H + sy) * sS.W + sx) * sS.C;
        for (int c = 0; c < C; ++c) {
          float sv = (float)sp[c];
          float bv = base[base_off + (bS.C == 1 ? 0 : c)];
          float sc = scale[scale_off + (sS.C == 1 ? 0 : c)];
          float mul = stddev ? (1.0f / std::sqrt(sc * sc + eps)) : sc;
          rp[c] = (sv - bv) * mul * gscale + shift;
        }
      }
}

}  // namespace normalize_gold

#endif  // __NORMALIZE_GOLD_HPP__
