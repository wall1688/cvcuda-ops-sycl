// CPU gold reference for Remap. Extracted verbatim from the cpu_remap() that
// lived in test_remap.cpp so that both the correctness test and the profile
// benchmark share one source of truth. Mirrors the run_remap kernel exactly
// (reuses the same getRemapParams + sample() from remap.hpp / sampler.hpp), so
// GPU-vs-gold is a self-consistency check, NOT a bit-level diff vs NVIDIA
// CV-CUDA. See README "验证局限".
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef __REMAP_GOLD_HPP__
#define __REMAP_GOLD_HPP__

#include "../remap.hpp"
#include "../common/tensorwrap.hpp"

#include <vector>

namespace remap_gold {

// Whole-image CPU reference. dst's W/H/N are read; dst.data is unused (output
// goes to refdst, sized N*H*W*C). mapNumSamples==1 broadcasts the map across
// the batch (map z=0).
template <typename T, int C>
void cpu_remap(const cvsycl::TensorView<T, C>& src, const cvsycl::TensorView<T, C>& dst,
               const cvsycl::TensorView<float, 2>& map, cvsycl::Interp si, cvsycl::Interp mi,
               cvsycl::MapValueType mvt, bool alignCorners, cvsycl::Border b,
               cvsycl::BorderValue bv, int mapNumSamples, std::vector<T>& refdst) {
    cvsycl::RemapParams p = cvsycl::getRemapParams(src.W, src.H, dst.W, dst.H, map.W, map.H,
                                                   alignCorners, mvt);
    cvsycl::Border mb = cvsycl::Border::Replicate;
    cvsycl::BorderValue mbv{};
    for (int z = 0; z < dst.N; ++z)
        for (int dy = 0; dy < dst.H; ++dy)
            for (int dx = 0; dx < dst.W; ++dx) {
                float mcx = (dx + p.dstOffset) * p.mapScaleX;
                float mcy = (dy + p.dstOffset) * p.mapScaleY;
                int   mz  = (mapNumSamples == 1) ? 0 : z;
                float mpix[2];
                cvsycl::sample<float, 2>(map, mcx, mcy, mz, mi, mb, mbv, mpix);
                float scx = dx * p.srcScaleX + mpix[0] * p.valScaleX + p.srcOffsetX;
                float scy = dy * p.srcScaleY + mpix[1] * p.valScaleY + p.srcOffsetY;
                T out[C];
                cvsycl::sample<T, C>(src, scx, scy, z, si, b, bv, out);
                for (int c = 0; c < C; ++c)
                    refdst[((z * dst.H + dy) * dst.W + dx) * C + c] = out[c];
            }
}

}  // namespace remap_gold

#endif  // __REMAP_GOLD_HPP__
