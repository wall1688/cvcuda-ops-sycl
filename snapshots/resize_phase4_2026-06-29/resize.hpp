// SYCL port of CV-CUDA's Resize operator (priv/OpResize.cu).
// Phase 2: templated on element type T (float / uint8_t) and channels C (1/3/4).
// NHWC layout. NEAREST + LINEAR. Header-only (templates).
//
// Faithful translation of the CUDA kernels with the phase-1/2 simplifications:
//   - NIX<T> == 1 (no vector pack write)  => one dst pixel per thread
//   - INTERSECT == false (no src-coord reuse optimization)  => simplest path
//   - per-channel interpolation done in float, then SaturateCast<T> back
//     (LINEAR in the CUDA original computes in T via vector math that promotes;
//      doing it explicitly in float is numerically equivalent and SYCL-clean).
//
// Coordinate convention (copied verbatim from OpResize.cu):
//   NEAREST: srcCoord = (dstCoord + 0.5) * scaleRatio
//   LINEAR : srcCoord = (dstCoord + 0.5) * scaleRatio - 0.5
//   scaleRatio = srcSize / dstSize   (per-axis)
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_RESIZE_HPP
#define CVSACL_RESIZE_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include "common/tensorwrap.hpp"
#include "common/math_ops.hpp"
#include "common/saturate_cast.hpp"

namespace cvsycl {

enum class Interp {
    Nearest,
    Linear,
    Cubic,
    Area,
};

// Cubic interpolation weights (Keys kernel, A = -0.75), matching
// OpResize.cu::GetCubicCoeffs exactly. delta in [0,1). Usable host + device.
inline void get_cubic_coeffs(float delta, float w[4]) {
    constexpr float A = -0.75f;
    w[0] = ((A * (delta + 1.f) - 5.f * A) * (delta + 1.f) + 8.f * A) * (delta + 1.f) - 4.f * A;
    w[1] = ((A + 2.f) * delta - (A + 3.f)) * delta * delta + 1.f;
    w[2] = ((A + 2.f) * (1.f - delta) - (A + 3.f)) * (1.f - delta) * (1.f - delta) + 1.f;
    w[3] = 1.f - w[0] - w[1] - w[2];
}

// ---- NEAREST ----------------------------------------------------------------
// One dst pixel per thread: map dst coord -> src, round down, clamp, copy.
// (OpResize.cu :: NearestResize, INTERSECT=false, NIX=1, generalized to C.)
template <typename T, int C>
inline sycl::event nearest_resize(sycl::queue& q, const TensorView<T, C>& src,
                                  const TensorView<T, C>& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;

    return q.submit([&](sycl::handler& h) {
        TensorView<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                int isx = (int)sycl::floor((x + 0.5f) * scaleX);
                int isy = (int)sycl::floor((y + 0.5f) * scaleY);
                isx = cvsycl::min(isx, src.W - 1);
                isy = cvsycl::min(isy, src.H - 1);

                T pix[C];
                s.read(isx, isy, z, pix);
                d.write(x, y, z, pix);
            });
    });
}

// ---- LINEAR -----------------------------------------------------------------
// One dst pixel per thread via 2x2 bilinear interpolation, per channel.
// (OpResize.cu :: LinearResize, INTERSECT=false, NIX=1, generalized to C.)
template <typename T, int C>
inline sycl::event linear_resize(sycl::queue& q, const TensorView<T, C>& src,
                                 const TensorView<T, C>& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;

    return q.submit([&](sycl::handler& h) {
        TensorView<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                float srcCoordX = (x + 0.5f) * scaleX - 0.5f;
                float srcCoordY = (y + 0.5f) * scaleY - 0.5f;

                // --- x axis ---
                int isx = (int)sycl::floor(srcCoordX);
                float wx = (isx < 0)          ? 0.f
                         : (isx > src.W - 2)  ? 1.f
                         :                      (srcCoordX - isx);
                isx = cvsycl::max(0, cvsycl::min(isx, src.W - 2));

                // --- y axis ---
                int isy = (int)sycl::floor(srcCoordY);
                float wy = (isy < 0)          ? 0.f
                         : (isy > src.H - 2)  ? 1.f
                         :                      (srcCoordY - isy);
                isy = cvsycl::max(0, cvsycl::min(isy, src.H - 2));

                T p00[C], p10[C], p01[C], p11[C];
                s.read(isx,     isy,     z, p00);
                s.read(isx + 1, isy,     z, p10);
                s.read(isx,     isy + 1, z, p01);
                s.read(isx + 1, isy + 1, z, p11);

                float w00 = (1.f - wx) * (1.f - wy);
                float w10 = wx * (1.f - wy);
                float w01 = (1.f - wx) * wy;
                float w11 = wx * wy;

                T out[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) {
                    float v = (float)p00[c] * w00 + (float)p10[c] * w10
                            + (float)p01[c] * w01 + (float)p11[c] * w11;
                    out[c] = SaturateCast<T>(v);
                }
                d.write(x, y, z, out);
            });
    });
}

// ---- CUBIC ------------------------------------------------------------------
// One dst pixel per thread via 4x4 bicubic interpolation (Keys, A=-0.75), per
// channel. (OpResize.cu :: CubicResize, generalized to C.)
// Note the original's asymmetric edge handling is replicated faithfully:
//   - fx is zeroed at the x-edges (isx<1 || isx>=W-3); fy is NOT zeroed.
//   - isx/isy are clamped to [1, size-3]  => requires src W,H >= 4.
// Result is abs(sum) then SaturateCast (cubic overshoots can go negative).
template <typename T, int C>
inline sycl::event cubic_resize(sycl::queue& q, const TensorView<T, C>& src,
                                const TensorView<T, C>& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;

    return q.submit([&](sycl::handler& h) {
        TensorView<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                float srcCoordX = (x + 0.5f) * scaleX - 0.5f;
                float srcCoordY = (y + 0.5f) * scaleY - 0.5f;

                int isx = (int)sycl::floor(srcCoordX);
                int isy = (int)sycl::floor(srcCoordY);

                float fx = srcCoordX - isx;
                // fy is left unchanged (matches the CUDA original).

                fx = (isx < 1 || isx >= src.W - 3) ? 0.f : fx;
                isy = cvsycl::max(1, cvsycl::min(isy, src.H - 3));
                isx = cvsycl::max(1, cvsycl::min(isx, src.W - 3));

                float wx[4], wy[4];
                get_cubic_coeffs(fx, wx);
                get_cubic_coeffs(srcCoordY - isy, wy);

                // Read the 4x4 src neighbourhood once.
                T p[4][4][C];
                #pragma unroll
                for (int cy = -1; cy <= 2; ++cy)
                    #pragma unroll
                    for (int cx = -1; cx <= 2; ++cx)
                        s.read(isx + cx, isy + cy, z, p[cy + 1][cx + 1]);

                T out[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int cy = -1; cy <= 2; ++cy)
                        #pragma unroll
                        for (int cx = -1; cx <= 2; ++cx)
                            sum += (float)p[cy + 1][cx + 1][c] * (wx[cx + 1] * wy[cy + 1]);
                    out[c] = SaturateCast<T>(sycl::fabs(sum));
                }
                d.write(x, y, z, out);
            });
    });
}

// ---- AREA -------------------------------------------------------------------
// One dst pixel per thread via area (box) averaging over the source footprint
// [x*scaleX, (x+1)*scaleX) x [y*scaleY, (y+1)*scaleY). Per channel.
// (OpResize.cu :: AreaResize + cuda_tools/InterpolationWrap AREA specialization.)
//
// The footprint bounds use ceil for the min edge and floor for the max edge
// (GetIndexForInterpolation AREA: position 1 = round UP, position 2 = round
// DOWN). Integer scales take a simple uniform average; non-integer scales add
// fractional-weight partial-edge terms (9 terms total). All scale math is done
// on the host and passed in, so the kernel does no division — it is unaffected
// by the Intel GPU's non-IEEE division. Out-of-range reads return 0 (border
// constant, matching NVCV_BORDER_CONSTANT with borderValue 0).
template <typename T, int C>
inline sycl::event area_resize(sycl::queue& q, const TensorView<T, C>& src,
                               const TensorView<T, C>& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;
    // Integer-area flag (both scales exact integers). Computed on host.
    bool isIntArea = ((float)(int)sycl::ceil(scaleX) == scaleX)
                     && ((float)(int)sycl::ceil(scaleY) == scaleY);

    return q.submit([&](sycl::handler& h) {
        TensorView<T, C> s = src, d = dst;
        float sX = scaleX, sY = scaleY;
        bool intArea = isIntArea;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                float fsx1 = x * sX;
                float fsy1 = y * sY;
                float fsx2 = fsx1 + sX;
                float fsy2 = fsy1 + sY;

                int xmin = (int)sycl::ceil(fsx1);   // round UP
                int xmax = (int)sycl::floor(fsx2);  // round DOWN
                int ymin = (int)sycl::ceil(fsy1);
                int ymax = (int)sycl::floor(fsy2);

                float out[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) out[c] = 0.f;

                // Read a source pixel at (cx,cy) with border-constant 0; add w
                // to every channel.
                auto add = [&](int cx, int cy, float w) {
                    T pix[C];
                    if (cx < 0 || cx >= s.W || cy < 0 || cy >= s.H) {
                        #pragma unroll
                        for (int c = 0; c < C; ++c) pix[c] = (T)0;
                    } else {
                        s.read(cx, cy, z, pix);
                    }
                    #pragma unroll
                    for (int c = 0; c < C; ++c) out[c] += (float)pix[c] * w;
                };

                if (intArea) {
                    float scale = 1.f / (sX * sY);
                    for (int cy = ymin; cy < ymax; ++cy)
                        for (int cx = xmin; cx < xmax; ++cx)
                            add(cx, cy, scale);
                } else {
                    const int W = s.W, H = s.H;
                    float scale = 1.f / (cvsycl::min(sX, (float)W - fsx1)
                                         * cvsycl::min(sY, (float)H - fsy1));

                    for (int cy = ymin; cy < ymax; ++cy) {
                        for (int cx = xmin; cx < xmax; ++cx)
                            add(cx, cy, scale);
                        if (xmin > fsx1) add(xmin - 1, cy, (xmin - fsx1) * scale);
                        if (xmax < fsx2) add(xmax,     cy, (fsx2 - xmax) * scale);
                    }
                    if (ymin > fsy1) {
                        for (int cx = xmin; cx < xmax; ++cx)
                            add(cx, ymin - 1, (ymin - fsy1) * scale);
                        if (xmin > fsx1)
                            add(xmin - 1, ymin - 1, (ymin - fsy1) * (xmin - fsx1) * scale);
                        if (xmax < fsx2)
                            add(xmax,     ymin - 1, (ymin - fsy1) * (fsx2 - xmax) * scale);
                    }
                    if (ymax < fsy2) {
                        for (int cx = xmin; cx < xmax; ++cx)
                            add(cx, ymax, (fsy2 - ymax) * scale);
                        if (xmax < fsx2)
                            add(xmax,     ymax, (fsy2 - ymax) * (fsx2 - xmax) * scale);
                        if (xmin > fsx1)
                            add(xmin - 1, ymax, (fsy2 - ymax) * (xmin - fsx1) * scale);
                    }
                }

                T outT[C];
                #pragma unroll
                for (int c = 0; c < C; ++c) outT[c] = SaturateCast<T>(out[c]);
                d.write(x, y, z, outT);
            });
    });
}

// ---- host dispatch ----------------------------------------------------------
template <typename T, int C>
inline sycl::event run_resize(sycl::queue& q, const TensorView<T, C>& src,
                              const TensorView<T, C>& dst, Interp interp) {
    if (src.N != dst.N) {
        throw std::runtime_error("run_resize: src/dst N mismatch");
    }
    if (interp == Interp::Linear && (src.W < 2 || src.H < 2)) {
        throw std::runtime_error("run_resize: LINEAR needs src W,H >= 2");
    }
    if (interp == Interp::Cubic && (src.W < 4 || src.H < 4)) {
        throw std::runtime_error("run_resize: CUBIC needs src W,H >= 4");
    }
    switch (interp) {
        case Interp::Nearest: return nearest_resize<T, C>(q, src, dst);
        case Interp::Linear:  return linear_resize<T, C>(q, src, dst);
        case Interp::Cubic:   return cubic_resize<T, C>(q, src, dst);
        case Interp::Area:    return area_resize<T, C>(q, src, dst);
    }
    throw std::runtime_error("run_resize: unsupported interpolation");
}

}  // namespace cvsycl

#endif  // CVSACL_RESIZE_HPP
