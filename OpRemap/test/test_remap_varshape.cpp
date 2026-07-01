// Var-shape correctness test for the SYCL remap port (Phase 7).
//
// Each image in the batch has its own src/dst/map dimensions (pitched
// VarShapeBatch). The CPU reference mirrors the kernel and reuses the SAME
// shape-agnostic sample_impl() over a host-pointer VarShapeBatch — so this is a
// self-consistency check (SYCL kernel == my CPU impl), NOT a bit-diff vs NVIDIA.
//
// Run: ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_remap_varshape
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../remap_varshape.hpp"
#include "../common/varshape.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

namespace cs = cvsycl;

// ---------------- pitched host batch -----------------------------------------
template <typename T, int C>
struct HostBatch {
    std::vector<T> data;
    std::vector<int> imgW, imgH;
    int N, maxW, maxH;
    long rowPitch()    const { return (long)maxW * C; }
    long imageStride() const { return (long)maxH * maxW * C; }
    T& at(int n, int y, int x, int c) {
        return data[(long)n * imageStride() + (long)y * rowPitch() + (long)x * C + c];
    }
};

template <typename T, int C>
static HostBatch<T, C> makeBatch(int N, const std::vector<int>& H,
                                 const std::vector<int>& W) {
    HostBatch<T, C> b;
    b.N = N;
    b.maxW = *std::max_element(W.begin(), W.end());
    b.maxH = *std::max_element(H.begin(), H.end());
    b.imgW = W; b.imgH = H;
    b.data.assign((size_t)N * b.maxH * b.maxW * C, (T)0);
    return b;
}

// Build a device VarShapeBatch (alloc + copy data + copy dims).
template <typename T, int C>
static cs::VarShapeBatch<T, C> toDevice(sycl::queue& q, const HostBatch<T, C>& hb) {
    cs::VarShapeBatch<T, C> vb;
    vb.N = hb.N; vb.maxW = hb.maxW; vb.maxH = hb.maxH;
    vb.data = sycl::malloc_device<T>(hb.data.size(), q);
    vb.imgW = sycl::malloc_device<int>(hb.N, q);
    vb.imgH = sycl::malloc_device<int>(hb.N, q);
    q.copy(hb.data.data(), vb.data, hb.data.size());
    q.copy(hb.imgW.data(), vb.imgW, hb.N);
    q.copy(hb.imgH.data(), vb.imgH, hb.N);
    return vb;
}
// Host VarShapeBatch view (pointers into host memory) for the CPU reference.
template <typename T, int C>
static cs::VarShapeBatch<T, C> toHostView(HostBatch<T, C>& hb) {
    return cs::VarShapeBatch<T, C>{hb.data.data(), hb.imgW.data(), hb.imgH.data(),
                                   hb.N, hb.maxW, hb.maxH};
}

// ---------------- CPU reference (mirrors run_remap_varshape kernel) ----------
template <typename T, int C>
static void cpu_remap_varshape(const cs::VarShapeBatch<T, C>& s,
                               const cs::VarShapeBatch<T, C>& d,
                               const cs::VarShapeBatch<float, 2>& m,
                               cs::Interp si, cs::Interp mi, cs::MapValueType mvt,
                               bool ac, cs::Border b, cs::BorderValue bv,
                               HostBatch<T, C>& ref) {
    cs::Border mb = cs::Border::Replicate;
    cs::BorderValue mbv{};
    for (int n = 0; n < s.N; ++n) {
        int mn = (m.N == 1) ? 0 : n;
        cs::RemapParams p = cs::getRemapParams(s.imgW[n], s.imgH[n], d.imgW[n], d.imgH[n],
                                               m.imgW[mn], m.imgH[mn], ac, mvt);
        for (int dy = 0; dy < d.imgH[n]; ++dy)
            for (int dx = 0; dx < d.imgW[n]; ++dx) {
                float mcx = (dx + p.dstOffset) * p.mapScaleX;
                float mcy = (dy + p.dstOffset) * p.mapScaleY;
                float mpix[2];
                cs::sample_impl<cs::VarShapeBatch<float, 2>>(m, mcx, mcy, mn,
                    m.imgW[mn], m.imgH[mn], mi, mb, mbv, mpix);
                float scx = dx * p.srcScaleX + mpix[0] * p.valScaleX + p.srcOffsetX;
                float scy = dy * p.srcScaleY + mpix[1] * p.valScaleY + p.srcOffsetY;
                T out[C];
                cs::sample_impl<cs::VarShapeBatch<T, C>>(s, scx, scy, n,
                    s.imgW[n], s.imgH[n], si, b, bv, out);
                for (int c = 0; c < C; ++c) ref.at(n, dy, dx, c) = out[c];
            }
    }
}

// ---------------- generic case runner ----------------------------------------
template <typename T, int C>
static int run_case(sycl::queue& q, const std::string& name,
                    const std::vector<int>& sw, const std::vector<int>& sh,
                    const std::vector<int>& dw, const std::vector<int>& dh,
                    const std::vector<int>& mw, const std::vector<int>& mh,
                    cs::Interp si, cs::Interp mi, cs::MapValueType mvt, bool ac,
                    cs::Border b, cs::BorderValue bv, double tol) {
    int N = (int)sw.size();

    HostBatch<T, C> hsrc = makeBatch<T, C>(N, sh, sw);
    for (int n = 0; n < N; ++n)
        for (int y = 0; y < sh[n]; ++y)
            for (int x = 0; x < sw[n]; ++x)
                for (int c = 0; c < C; ++c) {
                    int v = n * 100 + y * 10 + x + c;
                    hsrc.at(n, y, x, c) = (sizeof(T) == 1) ? (T)(v & 0xff) : (T)v;
                }

    HostBatch<float, 2> hmap = makeBatch<float, 2>((int)mw.size(), mh, mw);
    for (int n = 0; n < (int)mw.size(); ++n)
        for (int y = 0; y < mh[n]; ++y)
            for (int x = 0; x < mw[n]; ++x) {
                // Fractional srcCoord -> exercises src interp. Offsets 0.25/0.75
                // (NOT 0.5): a 0.5 offset makes mapValue land EXACTLY on the
                // NEAREST rounding boundary k+0.5, where a 1-ULP host/device FMA
                // difference flips floor(c+0.5) by 1 -> a 1-pixel NEAREST flip
                // (same FMA-ULP phenomenon as the fixed-shape 5.7e-6 cases, but
                // amplified by NEAREST's discontinuity). 0.25/0.75 keep mapValue
                // >=0.25 from any k+0.5 -> no flip, tight tolerance valid.
                hmap.at(n, y, x, 0) = (float)x + 0.25f;
                hmap.at(n, y, x, 1) = (float)y + 0.75f;
            }

    HostBatch<T, C> hdst = makeBatch<T, C>(N, dh, dw);  // zeroed (gpu writes valid only)

    // GPU run.
    cs::VarShapeBatch<T, C> dsrc = toDevice<T, C>(q, hsrc);
    cs::VarShapeBatch<T, C> ddst = toDevice<T, C>(q, hdst);
    cs::VarShapeBatch<float, 2> dmap = toDevice<float, 2>(q, hmap);
    cs::run_remap_varshape<T, C>(q, dsrc, ddst, dmap, si, mi, mvt, ac, b, bv).wait();
    HostBatch<T, C> gdst = makeBatch<T, C>(N, dh, dw);
    q.copy(ddst.data, gdst.data.data(), gdst.data.size()).wait();
    sycl::free(dsrc.data, q); sycl::free(ddst.data, q); sycl::free(dmap.data, q);
    sycl::free(dsrc.imgW, q); sycl::free(dsrc.imgH, q);
    sycl::free(ddst.imgW, q); sycl::free(ddst.imgH, q);
    sycl::free(dmap.imgW, q); sycl::free(dmap.imgH, q);

    // CPU reference.
    cs::VarShapeBatch<T, C> hsrcv = toHostView(hsrc);
    cs::VarShapeBatch<T, C> hdstv = toHostView(hdst);
    cs::VarShapeBatch<float, 2> hmapv = toHostView(hmap);
    HostBatch<T, C> ref = makeBatch<T, C>(N, dh, dw);
    cpu_remap_varshape<T, C>(hsrcv, hdstv, hmapv, si, mi, mvt, ac, b, bv, ref);

    // Compare valid dst pixels only (padding is unwritten garbage).
    double maxerr = 0; size_t npix = 0;
    for (int n = 0; n < N; ++n)
        for (int y = 0; y < dh[n]; ++y)
            for (int x = 0; x < dw[n]; ++x)
                for (int c = 0; c < C; ++c) {
                    double e = std::fabs((double)gdst.at(n, y, x, c) - (double)ref.at(n, y, x, c));
                    if (e > maxerr) maxerr = e;
                    ++npix;
                }
    bool pass = maxerr <= tol;
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    std::printf("[%s] %-20s %s C%d N%d  %zu pix  maxerr=%.3e\n",
                pass ? "PASS" : "FAIL", name.c_str(), dt, C, N, npix, maxerr);
    return pass ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Per-image dims (N=3, all different).
    const std::vector<int> sw{5, 7, 4}, sh{4, 6, 5};   // src
    const std::vector<int> dw{6, 5, 7}, dh{5, 4, 6};   // dst
    const std::vector<int> mw{4, 3, 5}, mh{4, 5, 3};   // map (per-image)
    // Single broadcast map (for the map.N==1 case): 4x4.
    const std::vector<int> mw1{4}, mh1{4};

    const cs::Interp   NE = cs::Interp::Nearest, LI = cs::Interp::Linear, CU = cs::Interp::Cubic;
    const cs::MapValueType ABS = cs::MapValueType::Absolute;
    const cs::MapValueType AN  = cs::MapValueType::AbsoluteNormalized;
    const cs::Border   REP = cs::Border::Replicate, RFL = cs::Border::Reflect;
    const cs::BorderValue BV{};
    const double TF = 1e-4, TU = 1.0, TZ = 0.0;

    int fails = 0;
    fails += run_case<float,1>(q, "f32_near_near_abs",   sw,sh, dw,dh, mw,mh, NE,NE, ABS,false,REP, BV, TZ);
    fails += run_case<float,1>(q, "f32_lin_near_abs",    sw,sh, dw,dh, mw,mh, LI,NE, ABS,false,REP, BV, TF);
    fails += run_case<float,1>(q, "f32_cub_near_refl",   sw,sh, dw,dh, mw,mh, CU,NE, ABS,false,RFL, BV, TF);
    fails += run_case<float,1>(q, "f32_near_lin_abs",    sw,sh, dw,dh, mw,mh, NE,LI, ABS,false,REP, BV, TF);
    fails += run_case<float,1>(q, "f32_absnorm_near",    sw,sh, dw,dh, mw,mh, NE,NE, AN, false,REP, BV, TZ);
    fails += run_case<uint8_t,4>(q, "u8c4_lin_near_abs", sw,sh, dw,dh, mw,mh, LI,NE, ABS,false,REP, BV, TU);
    // map broadcast (map.N==1) — one 4x4 map shared by all 3 images.
    fails += run_case<float,1>(q, "f32_mapbc_lin_abs",   sw,sh, dw,dh, mw1,mh1, LI,NE, ABS,false,REP, BV, TF);

    if (fails == 0) std::printf("\n=== ALL VARSHAPE TESTS PASSED ===\n");
    else            std::printf("\n=== %d VARSHAPE TEST(S) FAILED ===\n", fails);
    return fails ? 1 : 0;
}
