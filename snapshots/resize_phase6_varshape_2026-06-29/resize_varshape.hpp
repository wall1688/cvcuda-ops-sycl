// SYCL port of CV-CUDA's var-shape Resize (legacy/resize_var_shape.cu).
// Phase 6: each image in the batch has its own W/H. NN + LINEAR + CUBIC.
// (AREA var-shape uses a separate OpenCV-aligned implementation in the original
//  and is not ported here — see limitations note.)
//
// Layout: pitched VarShapeBatch<T,C> (see common/varshape.hpp). The kernel
// launches over the dst batch's MAX dst dims and each thread bounds-checks
// against its own image's dstWidth/dstHeight (exactly like the legacy kernels).
// Per-image scale = srcDim / dstDim. Interpolation math is the SAME as the
// fixed-shape port (resize.hpp), just with per-image dims/scale and pitched
// addressing. One dst pixel per thread (the var-shape path is not the perf
// hotspot; the fixed-shape NIX optimization is not applied here).
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_RESIZE_VARSHAPE_HPP
#define CVSACL_RESIZE_VARSHAPE_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include "common/varshape.hpp"
#include "common/math_ops.hpp"
#include "common/saturate_cast.hpp"
#include "resize.hpp"  // Interp, get_cubic_coeffs

namespace cvsycl {

// ---- NEAREST ----------------------------------------------------------------
template <typename T, int C>
inline sycl::event varshape_nearest(sycl::queue& q, const VarShapeBatch<T, C>& src,
                                    const VarShapeBatch<T, C>& dst) {
    return q.submit([&](sycl::handler& h) {
        VarShapeBatch<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.maxH, (size_t)dst.maxW},
            [=](sycl::item<3> it) {
                int n = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];
                int dw = d.imgW[n], dh = d.imgH[n];
                if (x >= dw || y >= dh) return;

                int sw = s.imgW[n], sh = s.imgH[n];
                float sx = (float)sw / dw, sy = (float)sh / dh;
                int isx = (int)sycl::floor((x + 0.5f) * sx);
                int isy = (int)sycl::floor((y + 0.5f) * sy);
                isx = cvsycl::min(isx, sw - 1);
                isy = cvsycl::min(isy, sh - 1);

                T pix[C];
                s.read(isx, isy, n, pix);
                d.write(x, y, n, pix);
            });
    });
}

// ---- LINEAR -----------------------------------------------------------------
template <typename T, int C>
inline sycl::event varshape_linear(sycl::queue& q, const VarShapeBatch<T, C>& src,
                                   const VarShapeBatch<T, C>& dst) {
    return q.submit([&](sycl::handler& h) {
        VarShapeBatch<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.maxH, (size_t)dst.maxW},
            [=](sycl::item<3> it) {
                int n = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];
                int dw = d.imgW[n], dh = d.imgH[n];
                if (x >= dw || y >= dh) return;

                int sw = s.imgW[n], sh = s.imgH[n];
                float scaleX = (float)sw / dw, scaleY = (float)sh / dh;

                float srcCoordX = (x + 0.5f) * scaleX - 0.5f;
                float srcCoordY = (y + 0.5f) * scaleY - 0.5f;
                int isx = (int)sycl::floor(srcCoordX);
                int isy = (int)sycl::floor(srcCoordY);
                float wx = (isx < 0)         ? 0.f
                         : (isx > sw - 2)    ? 1.f
                         :                     (srcCoordX - isx);
                float wy = (isy < 0)         ? 0.f
                         : (isy > sh - 2)    ? 1.f
                         :                     (srcCoordY - isy);
                isx = cvsycl::max(0, cvsycl::min(isx, sw - 2));
                isy = cvsycl::max(0, cvsycl::min(isy, sh - 2));

                T p00[C], p10[C], p01[C], p11[C];
                s.read(isx,     isy,     n, p00);
                s.read(isx + 1, isy,     n, p10);
                s.read(isx,     isy + 1, n, p01);
                s.read(isx + 1, isy + 1, n, p11);

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
                d.write(x, y, n, out);
            });
    });
}

// ---- CUBIC ------------------------------------------------------------------
template <typename T, int C>
inline sycl::event varshape_cubic(sycl::queue& q, const VarShapeBatch<T, C>& src,
                                  const VarShapeBatch<T, C>& dst) {
    return q.submit([&](sycl::handler& h) {
        VarShapeBatch<T, C> s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.maxH, (size_t)dst.maxW},
            [=](sycl::item<3> it) {
                int n = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];
                int dw = d.imgW[n], dh = d.imgH[n];
                if (x >= dw || y >= dh) return;

                int sw = s.imgW[n], sh = s.imgH[n];
                float scaleX = (float)sw / dw, scaleY = (float)sh / dh;

                float srcCoordX = (x + 0.5f) * scaleX - 0.5f;
                float srcCoordY = (y + 0.5f) * scaleY - 0.5f;
                int isx = (int)sycl::floor(srcCoordX);
                int isy = (int)sycl::floor(srcCoordY);
                float fx = srcCoordX - isx;
                fx = (isx < 1 || isx >= sw - 3) ? 0.f : fx;
                isy = cvsycl::max(1, cvsycl::min(isy, sh - 3));
                isx = cvsycl::max(1, cvsycl::min(isx, sw - 3));

                float wx[4], wy[4];
                get_cubic_coeffs(fx, wx);
                get_cubic_coeffs(srcCoordY - isy, wy);

                T p[4][4][C];
                #pragma unroll
                for (int cy = -1; cy <= 2; ++cy)
                    #pragma unroll
                    for (int cx = -1; cx <= 2; ++cx)
                        s.read(isx + cx, isy + cy, n, p[cy + 1][cx + 1]);

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
                d.write(x, y, n, out);
            });
    });
}

// ---- host dispatch ----------------------------------------------------------
template <typename T, int C>
inline sycl::event run_resize_varshape(sycl::queue& q, const VarShapeBatch<T, C>& src,
                                       const VarShapeBatch<T, C>& dst, Interp interp) {
    if (src.N != dst.N) {
        throw std::runtime_error("run_resize_varshape: src/dst N mismatch");
    }
    switch (interp) {
        case Interp::Nearest: return varshape_nearest<T, C>(q, src, dst);
        case Interp::Linear:  return varshape_linear<T, C>(q, src, dst);
        case Interp::Cubic:   return varshape_cubic<T, C>(q, src, dst);
        case Interp::Area:
            throw std::runtime_error("run_resize_varshape: AREA not ported (legacy uses "
                                     "a separate OpenCV-aligned implementation)");
    }
    throw std::runtime_error("run_resize_varshape: unsupported interpolation");
}

}  // namespace cvsycl

#endif  // CVSACL_RESIZE_VARSHAPE_HPP
