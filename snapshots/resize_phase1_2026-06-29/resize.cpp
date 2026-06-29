// SYCL port of CV-CUDA's Resize operator (priv/OpResize.cu).
// Phase 1: F32, single-channel (C=1), NHWC. NEAREST + LINEAR.
//
// Faithful translation of the CUDA kernels with the phase-1 simplifications:
//   - NIX<T> == 1 (no vector pack write)  => one dst pixel per thread
//   - INTERSECT == false (no src-coord reuse optimization)  => simplest path
//   - T == float, C == 1
// These are exactly the cases the CUDA source falls into for a single-channel
// float tensor, so the numerics match bit-for-bit (modulo FP non-associativity
// in the LINEAR weighted sum, which the test tolerates).
//
// Coordinate convention (copied verbatim from OpResize.cu):
//   NEAREST: srcCoord = (dstCoord + 0.5) * scaleRatio
//   LINEAR : srcCoord = (dstCoord + 0.5) * scaleRatio - 0.5
//   scaleRatio = srcSize / dstSize   (per-axis)
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "resize.hpp"
#include "common/math_ops.hpp"
#include "common/saturate_cast.hpp"

#include <stdexcept>

namespace cvsycl {

// ---- NEAREST ----------------------------------------------------------------
// Each thread computes one dst pixel: map dst coord back to src, round down,
// clamp, copy. (OpResize.cu :: NearestResize, INTERSECT=false, NIX=1.)
static sycl::event nearest_resize(sycl::queue& q, const TensorView& src,
                                  const TensorView& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;

    return q.submit([&](sycl::handler& h) {
        TensorView s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                float srcCoordX = (x + 0.5f) * scaleX;
                float srcCoordY = (y + 0.5f) * scaleY;

                int isx = (int)sycl::floor(srcCoordX);
                int isy = (int)sycl::floor(srcCoordY);

                isx = cvsycl::min(isx, src.W - 1);
                isy = cvsycl::min(isy, src.H - 1);

                d.write1(x, y, z, s.read1(isx, isy, z));
            });
    });
}

// ---- LINEAR -----------------------------------------------------------------
// Each thread computes one dst pixel via 2x2 bilinear interpolation.
// (OpResize.cu :: LinearResize, INTERSECT=false, NIX=1.)
static sycl::event linear_resize(sycl::queue& q, const TensorView& src,
                                 const TensorView& dst) {
    float scaleX = (float)src.W / (float)dst.W;
    float scaleY = (float)src.H / (float)dst.H;

    return q.submit([&](sycl::handler& h) {
        TensorView s = src, d = dst;
        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int y = (int)it[1];
                int x = (int)it[2];

                float srcCoordX = (x + 0.5f) * scaleX - 0.5f;
                float srcCoordY = (y + 0.5f) * scaleY - 0.5f;

                // --- x axis ---
                int isx = (int)sycl::floor(srcCoordX);
                float wx = (isx < 0)              ? 0.f
                         : (isx > src.W - 2)      ? 1.f
                         :                          (srcCoordX - isx);
                isx = cvsycl::max(0, cvsycl::min(isx, src.W - 2));

                // --- y axis ---
                int isy = (int)sycl::floor(srcCoordY);
                float wy = (isy < 0)              ? 0.f
                         : (isy > src.H - 2)      ? 1.f
                         :                          (srcCoordY - isy);
                isy = cvsycl::max(0, cvsycl::min(isy, src.H - 2));

                float p00 = s.read1(isx,     isy,     z);
                float p10 = s.read1(isx + 1, isy,     z);
                float p01 = s.read1(isx,     isy + 1, z);
                float p11 = s.read1(isx + 1, isy + 1, z);

                float v = p00 * ((1.f - wx) * (1.f - wy))
                        + p10 * (wx * (1.f - wy))
                        + p01 * ((1.f - wx) * wy)
                        + p11 * (wx * wy);

                d.write1(x, y, z, SaturateCast<float>(v));
            });
    });
}

// ---- host dispatch ----------------------------------------------------------
sycl::event run_resize(sycl::queue& q, const TensorView& src, const TensorView& dst,
                       Interp interp) {
    // Minimal validation (the CUDA op does much more via nvcv::TensorDataAccess;
    // we only need the bits that affect kernel correctness).
    if (src.C != dst.C || src.N != dst.N) {
        throw std::runtime_error("run_resize: src/dst N or C mismatch");
    }
    if (src.C != 1) {
        throw std::runtime_error("run_resize: phase 1 supports C==1 only");
    }
    if (interp == Interp::Linear && (src.W < 2 || src.H < 2)) {
        throw std::runtime_error("run_resize: LINEAR needs src W,H >= 2");
    }

    switch (interp) {
        case Interp::Nearest: return nearest_resize(q, src, dst);
        case Interp::Linear:  return linear_resize(q, src, dst);
    }
    throw std::runtime_error("run_resize: unsupported interpolation");
}

}  // namespace cvsycl
