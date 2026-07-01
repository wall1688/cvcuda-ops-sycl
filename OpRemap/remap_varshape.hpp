// SYCL port of CV-CUDA's var-shape Remap (legacy/remap_var_shape.cu analog).
// Phase 7: each image in the batch has its own src/dst/map dimensions.
//
// Layout: pitched VarShapeBatch<T,C> for src/dst and VarShapeBatch<float,2> for
// map (see common/varshape.hpp). The kernel launches over the dst batch's MAX
// dims and each thread bounds-checks against its own image's dstW/dstH (exactly
// like the legacy var-shape kernels), returning early on padding.
//
// Per-image RemapParams are computed on the HOST (one getRemapParams call per
// image) and passed to the kernel as a device array — the kernel does only
// multiply-add, NO division (same Intel-FP-div avoidance as the fixed-shape
// path). Because imgW/imgH live in device USM, run_remap_varshape copies them
// to host, builds the params array, and copies it back. Var-shape is not the
// perf hotspot, so this host round-trip is acceptable.
//
// The sampler is REUSED (not duplicated): sample_impl/fetch_impl are
// shape-agnostic (W,H explicit), templated on the view, so the SAME interpolation
// + border math serves both TensorView (fixed) and VarShapeBatch (var). map is
// always sampled with REPLICATE border (kMapBorderType); the user border applies
// only to src. map.N may be 1 (broadcast) or N (per-image map).
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_REMAP_VARSHAPE_HPP
#define CVSACL_REMAP_VARSHAPE_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <vector>
#include "remap.hpp"            // RemapParams, getRemapParams, mapBorder, MapValueType
#include "common/varshape.hpp"

namespace cvsycl {

template <typename T, int C>
inline sycl::event run_remap_varshape(sycl::queue& q,
                                      const VarShapeBatch<T, C>& src,
                                      const VarShapeBatch<T, C>& dst,
                                      const VarShapeBatch<float, 2>& map,
                                      Interp srcInterp, Interp mapInterp,
                                      MapValueType mvt, bool alignCorners,
                                      Border border, BorderValue borderValue) {
    if (src.N != dst.N) {
        throw std::runtime_error("run_remap_varshape: src/dst N mismatch");
    }
    if (!(map.N == 1 || map.N == src.N)) {
        throw std::runtime_error("run_remap_varshape: map N must be 1 or src.N");
    }

    const int N = src.N;
    const int MN = map.N;

    // Pull per-image dims device->host to compute per-image params (no kernel div).
    std::vector<int> hsw(N), hsh(N), hdw(N), hdh(N), hmw(MN), hmh(MN);
    q.copy(src.imgW, hsw.data(), N); q.copy(src.imgH, hsh.data(), N);
    q.copy(dst.imgW, hdw.data(), N); q.copy(dst.imgH, hdh.data(), N);
    q.copy(map.imgW, hmw.data(), MN); q.copy(map.imgH, hmh.data(), MN);
    q.wait();

    std::vector<RemapParams> hparams(N);
    for (int n = 0; n < N; ++n) {
        int mn = (MN == 1) ? 0 : n;
        hparams[n] = getRemapParams(hsw[n], hsh[n], hdw[n], hdh[n],
                                    hmw[mn], hmh[mn], alignCorners, mvt);
    }
    RemapParams* d_params = sycl::malloc_device<RemapParams>(N, q);
    q.copy(hparams.data(), d_params, N);

    sycl::event ev = q.submit([&](sycl::handler& h) {
        VarShapeBatch<T, C> s = src, d = dst;
        VarShapeBatch<float, 2> m = map;
        RemapParams* params = d_params;
        Interp si = srcInterp, mi = mapInterp;
        Border b = border, mb = mapBorder();
        BorderValue bv = borderValue, mbv{};
        int mapN = MN;

        h.parallel_for(sycl::range<3>{(size_t)N, (size_t)dst.maxH, (size_t)dst.maxW},
            [=](sycl::item<3> it) {
                int n  = (int)it[0];
                int dy = (int)it[1];
                int dx = (int)it[2];
                int dw = d.imgW[n], dh = d.imgH[n];
                if (dx >= dw || dy >= dh) return;  // padding

                RemapParams p = params[n];
                int mn = (mapN == 1) ? 0 : n;
                int mw = m.imgW[mn], mh = m.imgH[mn];

                // 1) map coordinate, 2) sample map (float2, REPLICATE) -> mapValue.
                float mcx = (dx + p.dstOffset) * p.mapScaleX;
                float mcy = (dy + p.dstOffset) * p.mapScaleY;
                float mpix[2];
                sample_impl<VarShapeBatch<float, 2>>(m, mcx, mcy, mn, mw, mh,
                                                     mi, mb, mbv, mpix);

                // 3) source coordinate = dst*srcScale + mapValue*valScale + srcOffset.
                float scx = dx * p.srcScaleX + mpix[0] * p.valScaleX + p.srcOffsetX;
                float scy = dy * p.srcScaleY + mpix[1] * p.valScaleY + p.srcOffsetY;

                // 4) sample src (T, user border) -> write dst.
                int sw = s.imgW[n], sh = s.imgH[n];
                T out[C];
                sample_impl<VarShapeBatch<T, C>>(s, scx, scy, n, sw, sh,
                                                 si, b, bv, out);
                d.write(dx, dy, n, out);
            });
    });

    // Free the params buffer after the kernel consumes it (in_order queue).
    q.submit([&](sycl::handler& h) {
        h.depends_on(ev);
        h.host_task([q, d_params] { sycl::free(d_params, q); });
    });
    return ev;
}

}  // namespace cvsycl

#endif  // CVSACL_REMAP_VARSHAPE_HPP
