// Border-aware interpolation sampler for the SYCL remap port.
// Ported concept from CV-CUDA's cuda_tools/InterpolationWrap.hpp — the
// Nearest/Linear/Cubic operator[] + doGetValue/border fetch, stripped of the
// nvcv tensor framework and unified into one runtime-dispatched sample().
//
// Remap samples both the map (float2, always REPLICATE border) and the source
// (T with C channels, user-chosen border) at arbitrary float coordinates. This
// header provides the reusable sampler both share.
//
// Design choice (vs. OpResize): interp type and border are RUNTIME parameters
// (switch inside sample), not compile-time template dims. The Remap
// combinatorial space (T x C x srcInterp x mapInterp x border x mapValueType x
// alignCorners) is too large to template fully. Since interp/border are uniform
// across a launch, the switch is warp-uniform -> no divergence.
//
// CUBIC here uses the OpenCV Keys kernel with A = -0.5 (GetCubicCoeffs from
// InterpolationWrap.hpp) — NOT the A = -0.75 cubic used by OpResize. Verbatim
// port; do not reuse OpResize's get_cubic_coeffs.
//
// Host+device portable: uses sycl::floor/round so the SAME sample() runs in the
// CPU reference (over a host pointer) and the device kernel (over USM) —
// guaranteeing identical math, isolating any diff to FP non-associativity in
// the weighted sums.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_SAMPLER_HPP
#define CVSACL_SAMPLER_HPP

#include <sycl/sycl.hpp>
#include "tensorwrap.hpp"
#include "saturate_cast.hpp"
#include "border.hpp"

namespace cvsycl {

enum class Interp {
    Nearest,
    Linear,
    Cubic,
    // Area is not used by Remap.
};

// Per-channel border value (up to 4 channels). Trivially copyable so it can be
// captured by value into device lambdas. CV-CUDA passes a float4; we broadcast
// component c to channel c.
struct BorderValue {
    float v[4];
};

// Cubic interpolation weights (OpenCV Keys kernel, A = -0.75? NO — A = -0.5).
// Verbatim port of InterpolationWrap.hpp::GetCubicCoeffs. delta in [0,1).
// NOTE: this differs from OpResize's A = -0.75 get_cubic_coeffs.
inline void getCubicCoeffs(float delta, float w[4]) {
    float w0 = -.5f;
    w0 = w0 * delta + 1.f;
    w0 = w0 * delta - .5f;
    w0 = w0 * delta;

    float w1 = 1.5f;
    w1 = w1 * delta - 2.5f;
    w1 = w1 * delta;
    w1 = w1 * delta + 1.f;

    float w2 = -1.5f;
    w2 = w2 * delta + 2.f;
    w2 = w2 * delta + .5f;
    w2 = w2 * delta;

    float w3 = 1.f - w0 - w1 - w2;

    w[0] = w0; w[1] = w1; w[2] = w2; w[3] = w3;
}

// Border-aware fetch of the integer pixel at (x, y, z) (x/y may be out of
// range). Mirrors InterpolationWrap::doGetValue + BorderWrap::operator[].
// For Constant: out-of-range -> borderValue component; in-range -> direct read.
// For others: remap x,y into range via getIndexWithBorder, then read.
template <typename T, int C>
inline void fetch(const TensorView<T, C>& v, int x, int y, int z, Border b,
                  const BorderValue& bv, T* out) {
    if (b == Border::Constant && (isOutside(x, v.W) || isOutside(y, v.H))) {
        #pragma unroll
        for (int c = 0; c < C; ++c) out[c] = SaturateCast<T>(bv.v[c]);
    } else {
        int bx = (b == Border::Constant) ? x : getIndexWithBorder(b, x, v.W);
        int by = (b == Border::Constant) ? y : getIndexWithBorder(b, y, v.H);
        v.read(bx, by, z, out);
    }
}

// Sample the tensor at float coordinate (fx, fy, z) with the given interpolation
// and border. Result written to out[C]. Mirrors InterpolationWrap::operator[]
// for NEAREST/LINEAR/CUBIC.
//   NEAREST: round half-up  -> floor(c + 0.5)
//   LINEAR : bilinear on floor(c), floor(c)+1 with weights (x2-x),(x-x1)
//   CUBIC  : 4x4 Keys (A=-0.5) on floor(c)..floor(c)+3 neighbors (-1..+2)
template <typename T, int C>
inline void sample(const TensorView<T, C>& v, float fx, float fy, int z,
                   Interp interp, Border b, const BorderValue& bv, T* out) {
    if (interp == Interp::Nearest) {
        // CV-CUDA: GetIndexForInterpolation<NEAREST>(c + .5f) = floor(c + .5)
        int ix = (int)sycl::floor(fx + 0.5f);
        int iy = (int)sycl::floor(fy + 0.5f);
        fetch<T, C>(v, ix, iy, z, b, bv, out);
    } else if (interp == Interp::Linear) {
        int x1 = (int)sycl::floor(fx);
        int x2 = x1 + 1;
        int y1 = (int)sycl::floor(fy);
        int y2 = y1 + 1;
        float wx1 = (float)x2 - fx;  // weight on x1
        float wx2 = fx - (float)x1;  // weight on x2
        float wy1 = (float)y2 - fy;
        float wy2 = fy - (float)y1;
        T p00[C], p10[C], p01[C], p11[C];
        fetch<T, C>(v, x1, y1, z, b, bv, p00);
        fetch<T, C>(v, x2, y1, z, b, bv, p10);
        fetch<T, C>(v, x1, y2, z, b, bv, p01);
        fetch<T, C>(v, x2, y2, z, b, bv, p11);
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            float acc = 0.f;
            acc += (float)p00[c] * wx1 * wy1;
            acc += (float)p10[c] * wx2 * wy1;
            acc += (float)p01[c] * wx1 * wy2;
            acc += (float)p11[c] * wx2 * wy2;
            out[c] = SaturateCast<T>(acc);
        }
    } else {  // Cubic
        int ix = (int)sycl::floor(fx);
        int iy = (int)sycl::floor(fy);
        float wx[4], wy[4];
        getCubicCoeffs(fx - (float)ix, wx);
        getCubicCoeffs(fy - (float)iy, wy);
        float acc[C];
        #pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
        T p[C];
        #pragma unroll
        for (int cy = -1; cy <= 2; ++cy) {
            #pragma unroll
            for (int cx = -1; cx <= 2; ++cx) {
                fetch<T, C>(v, ix + cx, iy + cy, z, b, bv, p);
                float w = wx[cx + 1] * wy[cy + 1];
                #pragma unroll
                for (int c = 0; c < C; ++c) acc[c] += (float)p[c] * w;
            }
        }
        #pragma unroll
        for (int c = 0; c < C; ++c) out[c] = SaturateCast<T>(acc[c]);
    }
}

}  // namespace cvsycl

#endif  // CVSACL_SAMPLER_HPP
