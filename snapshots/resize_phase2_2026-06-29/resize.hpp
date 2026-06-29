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
    // Cubic, Area — later phases
};

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
    switch (interp) {
        case Interp::Nearest: return nearest_resize<T, C>(q, src, dst);
        case Interp::Linear:  return linear_resize<T, C>(q, src, dst);
    }
    throw std::runtime_error("run_resize: unsupported interpolation");
}

}  // namespace cvsycl

#endif  // CVSACL_RESIZE_HPP
