// SYCL port of CV-CUDA's Remap operator (priv/OpRemap.cu).
//
// Remap writes to each output pixel the source value at a position determined
// by a map. For each dst coord (dx,dy,z):
//   1. mapCoord = ((dx+dstOffset)*mapScale, (dy+dstOffset)*mapScale, mz)
//   2. mapValue = sample_map(mapCoord, mapInterp)        // map border = REPLICATE
//   3. srcCoord = (dx*srcScale + mapValue.x*valScale + srcOffset,
//                  dy*srcScale + mapValue.y*valScale + srcOffset)
//   4. dst      = sample_src(srcCoord, srcInterp, user border)
//
// All {srcScale,mapScale,valScale,srcOffset,dstOffset} are computed on the HOST
// by getRemapParams (depends on mapValueType x alignCorners) and passed in —
// the kernel does only multiply-add, NO division. This sidesteps the Intel Arc
// GPU FP-division non-IEEE issue (same win as OpResize's AREA path).
//
// Faithful translation of DoRemap + GetRemapParams + RunRemap dispatch from
// priv/OpRemap.cu, with the nvcv/cvcuda framework (IOperator /
// TensorDataStridedCuda / InterpolationWrapNHW) stripped and replaced by the
// self-contained TensorView + sampler (see common/sampler.hpp). The map is a
// TensorView<float,2> (float2 per pixel). mapNumSamples may be 1 (broadcast
// same map to all N) or N.
//
// Phase 1 scope: F32 C1, ABSOLUTE, NEAREST x NEAREST, REPLICATE. But the kernel
// and sampler are written general (any T/C/interp/border/mapValueType) — later
// phases only expand TEST coverage, not kernel code.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_REMAP_HPP
#define CVSACL_REMAP_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include "common/sampler.hpp"

namespace cvsycl {

enum class MapValueType {
    Absolute,             // NVCV_REMAP_ABSOLUTE
    AbsoluteNormalized,   // NVCV_REMAP_ABSOLUTE_NORMALIZED
    RelativeNormalized,   // NVCV_REMAP_RELATIVE_NORMALIZED
};

// Per-launch remap parameters (host-computed). Components of the float2
// srcScale/mapScale/valScale/srcOffset from NVCVRemapParams, plus scalar
// dstOffset. Verbatim port of OpRemap.cu::GetRemapParams.
struct RemapParams {
    float srcScaleX, srcScaleY;   // float2 srcScale
    float mapScaleX, mapScaleY;   // float2 mapScale
    float valScaleX, valScaleY;   // float2 valScale
    float srcOffsetX, srcOffsetY; // float2 srcOffset
    float dstOffset;              // float  dstOffset (same for x,y)
};

// Compute remap params from sizes + mapValueType + alignCorners.
// Verbatim port of OpRemap.cu::GetRemapParams (the scale/offset formulas).
inline RemapParams getRemapParams(int srcW, int srcH, int dstW, int dstH,
                                  int mapW, int mapH, bool alignCorners,
                                  MapValueType mvt) {
    RemapParams p;
    float sfW = (float)srcW, sfH = (float)srcH;
    float dfW = (float)dstW, dfH = (float)dstH;
    float mfW = (float)mapW, mfH = (float)mapH;

    switch (mvt) {
        case MapValueType::Absolute:
            // srcScale=0, mapScale=mapSize/dstSize, valScale=1,
            // srcOffset=0, dstOffset=0  ->  srcCoord = mapValue (absolute)
            p.srcScaleX = 0.f; p.srcScaleY = 0.f;
            p.mapScaleX = mfW / dfW; p.mapScaleY = mfH / dfH;
            p.valScaleX = 1.f; p.valScaleY = 1.f;
            p.srcOffsetX = 0.f; p.srcOffsetY = 0.f;
            p.dstOffset = 0.f;
            break;
        case MapValueType::AbsoluteNormalized: {
            float ac = alignCorners ? 1.f : 0.f;
            p.srcScaleX = 0.f; p.srcScaleY = 0.f;
            p.mapScaleX = mfW / dfW; p.mapScaleY = mfH / dfH;
            p.valScaleX = (sfW - ac) / 2.f; p.valScaleY = (sfH - ac) / 2.f;
            p.srcOffsetX = p.valScaleX - (alignCorners ? 0.f : .5f);
            p.srcOffsetY = p.valScaleY - (alignCorners ? 0.f : .5f);
            p.dstOffset = 0.f;
            break;
        }
        case MapValueType::RelativeNormalized:
            p.srcScaleX = sfW / dfW; p.srcScaleY = sfH / dfH;
            p.mapScaleX = (mfW - 1.f) / dfW; p.mapScaleY = (mfH - 1.f) / dfH;
            p.valScaleX = sfW - 1.f; p.valScaleY = sfH - 1.f;
            p.dstOffset = alignCorners ? 0.f : .5f;
            p.srcOffsetX = p.srcScaleX * p.dstOffset - p.dstOffset;
            p.srcOffsetY = p.srcScaleY * p.dstOffset - p.dstOffset;
            break;
    }
    return p;
}

// Map border is ALWAYS REPLICATE (kMapBorderType in OpRemap.cu), regardless of
// the user's src border. The map BorderValue is unused under REPLICATE.
inline Border mapBorder() { return Border::Replicate; }

// Run remap on the given queue. src/dst are TensorView<T,C>; map is
// TensorView<float,2>. mapNumSamples = map.N (1 or N). borderValue holds the
// per-channel constant border for src (used only when border == Constant).
template <typename T, int C>
inline sycl::event run_remap(sycl::queue& q,
                             const TensorView<T, C>& src,
                             const TensorView<T, C>& dst,
                             const TensorView<float, 2>& map,
                             Interp srcInterp, Interp mapInterp,
                             MapValueType mvt, bool alignCorners,
                             Border border, BorderValue borderValue,
                             int mapNumSamples) {
    if (src.N != dst.N) {
        throw std::runtime_error("run_remap: src/dst N mismatch");
    }
    if (!(mapNumSamples == 1 || mapNumSamples == src.N)) {
        throw std::runtime_error("run_remap: map N must be 1 or src.N");
    }
    RemapParams params = getRemapParams(src.W, src.H, dst.W, dst.H,
                                        map.W, map.H, alignCorners, mvt);

    return q.submit([&](sycl::handler& h) {
        TensorView<T, C> s = src, d = dst;
        TensorView<float, 2> m = map;
        RemapParams p = params;
        Interp si = srcInterp, mi = mapInterp;
        Border b = border;
        BorderValue bv = borderValue;
        int mns = mapNumSamples;
        Border mb = mapBorder();

        h.parallel_for(sycl::range<3>{(size_t)dst.N, (size_t)dst.H, (size_t)dst.W},
            [=](sycl::item<3> it) {
                int z = (int)it[0];
                int dy = (int)it[1];
                int dx = (int)it[2];

                // 1) map coordinate (dst offset + dst->map scale).
                float mcx = (dx + p.dstOffset) * p.mapScaleX;
                float mcy = (dy + p.dstOffset) * p.mapScaleY;
                int mz = (mns == 1) ? 0 : z;

                // 2) sample map (float2, REPLICATE border) -> mapValue.
                BorderValue mbv{};  // unused under REPLICATE
                float mpix[2];
                sample<float, 2>(m, mcx, mcy, mz, mi, mb, mbv, mpix);
                float mapVx = mpix[0];
                float mapVy = mpix[1];

                // 3) source coordinate = dst*srcScale + mapValue*valScale + srcOffset.
                float scx = dx * p.srcScaleX + mapVx * p.valScaleX + p.srcOffsetX;
                float scy = dy * p.srcScaleY + mapVy * p.valScaleY + p.srcOffsetY;

                // 4) sample src (T, user border) -> write dst.
                T out[C];
                sample<T, C>(s, scx, scy, z, si, b, bv, out);
                d.write(dx, dy, z, out);
            });
    });
}

}  // namespace cvsycl

#endif  // CVSACL_REMAP_HPP
