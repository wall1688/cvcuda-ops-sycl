// Synthetic correctness test for the SYCL remap port.
//
// Strategy (same as OpResize): build tiny synthetic src + map, run the GPU
// remap, and compare against a CPU reference that calls the EXACT SAME sample()
// / getRemapParams() functions (from remap.hpp + common/sampler.hpp). If they
// match within tolerance, the kernel is correct.
//
// Because the CPU ref reuses the device sampler code verbatim, this is a
// self-consistency check (SYCL kernel == my CPU implementation), NOT a
// bit-level diff vs NVIDIA CV-CUDA. See README "验证局限".
//
// Phase 1: F32 C1, ABSOLUTE, NEAREST x NEAREST, REPLICATE.
// Phase 2: + U8 C1/C3/C4 (CV-CUDA spec: F32 only C1; U8 supports C1/C3/C4).
//   The kernel/sampler are general; this phase only expands TEST coverage.
//
// Tolerance:
//   - NEAREST (Phase 1-2): exact (0) — pure copy with index rounding, no FP sum.
//
// Run on the Intel host:
//   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test_remap
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../remap.hpp"
#include "../common/tensorwrap.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace cs = cvsycl;

// ---------------- CPU reference (mirrors run_remap kernel verbatim) ----------
template <typename T, int C>
static void cpu_remap(const cs::TensorView<T, C>& src, const cs::TensorView<T, C>& dst,
                      const cs::TensorView<float, 2>& map, cs::Interp si, cs::Interp mi,
                      cs::MapValueType mvt, bool alignCorners, cs::Border b,
                      cs::BorderValue bv, int mapNumSamples, std::vector<T>& refdst) {
    cs::RemapParams p = cs::getRemapParams(src.W, src.H, dst.W, dst.H, map.W, map.H,
                                           alignCorners, mvt);
    cs::Border mb = cs::Border::Replicate;
    cs::BorderValue mbv{};
    for (int z = 0; z < dst.N; ++z)
        for (int dy = 0; dy < dst.H; ++dy)
            for (int dx = 0; dx < dst.W; ++dx) {
                float mcx = (dx + p.dstOffset) * p.mapScaleX;
                float mcy = (dy + p.dstOffset) * p.mapScaleY;
                int   mz  = (mapNumSamples == 1) ? 0 : z;
                float mpix[2];
                cs::sample<float, 2>(map, mcx, mcy, mz, mi, mb, mbv, mpix);
                float scx = dx * p.srcScaleX + mpix[0] * p.valScaleX + p.srcOffsetX;
                float scy = dy * p.srcScaleY + mpix[1] * p.valScaleY + p.srcOffsetY;
                T out[C];
                cs::sample<T, C>(src, scx, scy, z, si, b, bv, out);
                for (int c = 0; c < C; ++c)
                    refdst[((z * dst.H + dy) * dst.W + dx) * C + c] = out[c];
            }
}

// ---------------- GPU run helper --------------------------------------------
template <typename T, int C>
static void gpu_remap(sycl::queue& q, const std::vector<T>& hsrc,
                      const std::vector<float>& hmap, cs::TensorView<T, C> sv,
                      cs::TensorView<T, C> dv, cs::TensorView<float, 2> mv,
                      cs::Interp si, cs::Interp mi, cs::MapValueType mvt, bool ac,
                      cs::Border b, cs::BorderValue bv, int mapN,
                      std::vector<T>& gdst) {
    T* d_src = sycl::malloc_device<T>(hsrc.size(), q);
    T* d_dst = sycl::malloc_device<T>(gdst.size(), q);
    float* d_map = sycl::malloc_device<float>(hmap.size(), q);
    q.copy(hsrc.data(), d_src, hsrc.size());
    q.copy(hmap.data(), d_map, hmap.size());
    q.wait();
    sv.data = d_src; dv.data = d_dst; mv.data = d_map;
    cs::run_remap<T, C>(q, sv, dv, mv, si, mi, mvt, ac, b, bv, mapN).wait();
    q.copy(d_dst, gdst.data(), gdst.size()).wait();
    sycl::free(d_src, q); sycl::free(d_dst, q); sycl::free(d_map, q);
}

// ---------------- map fillers (ABSOLUTE: map pixel stores absolute srcCoord) -
static void map_set(std::vector<float>& map, int W, int H, int N, int z, int x, int y,
                    float vx, float vy) {
    size_t idx = ((size_t)z * H * W + y * W + x) * 2;
    map[idx] = vx; map[idx + 1] = vy;
}
static void fill_identity(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    map_set(m, W, H, N, z, x, y, (float)x, (float)y);
}
static void fill_shift_x(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    map_set(m, W, H, N, z, x, y, (float)(x + 1), (float)y);
}
static void fill_round(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    float off = (x % 2 == 0) ? 0.4f : 0.6f;
    map_set(m, W, H, N, z, x, y, (float)x + off, (float)y);
}
static void fill_oor(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    float vx = (x < W / 2) ? -2.f : (float)(W + 2);
    map_set(m, W, H, N, z, x, y, vx, (float)y);
}
// Fractional map for Phase 3: srcCoord = (x+0.5, y+0.5) -> exercises LINEAR/CUBIC
// interpolation in both axes; edge pixels go out-of-range -> exercises borders.
static void fill_half_xy(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    map_set(m, W, H, N, z, x, y, (float)x + 0.5f, (float)y + 0.5f);
}
// Phase 4 filler: distinct fractional ABSOLUTE srcCoord per map-pixel. Combined
// with mapW!=dstW (passed to run_case) -> fractional mapCoord -> the map is
// sampled with LINEAR/CUBIC, mixing distinct per-pixel values (real exercise of
// the map interp weighted sum). Values kept < srcW so srcCoord stays mostly
// in-range; map edges still exercise map REPLICATE border.
static void fill_frac_abs(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    float vx = (float)x + 0.25f * (float)y + 0.1f;
    float vy = (float)y + 0.25f * (float)x + 0.2f;
    map_set(m, W, H, N, z, x, y, vx, vy);
}
// Phase 5 fillers: map values in the semantic range of each normalized mvt.
// ABSOLUTE_NORMALIZED: mapValue in [-1,1], 0=center, +-1=edges (x=0->-1, x=W-1->+1).
static void fill_absnorm(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    float vx = (W > 1) ? (2.f * (float)x / (float)(W - 1) - 1.f) : 0.f;
    float vy = (H > 1) ? (2.f * (float)y / (float)(H - 1) - 1.f) : 0.f;
    map_set(m, W, H, N, z, x, y, vx, vy);
}
// RELATIVE_NORMALIZED: mapValue=0 -> default scaled pos; small relative offset in
// [-0.3,0.3] so edges push srcCoord out-of-range -> exercises REPLICATE border.
static void fill_relnorm(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    float vx = (W > 1) ? (0.3f * (2.f * (float)x / (float)(W - 1) - 1.f)) : 0.f;
    float vy = (H > 1) ? (0.3f * (2.f * (float)y / (float)(H - 1) - 1.f)) : 0.f;
    map_set(m, W, H, N, z, x, y, vx, vy);
}

// src patterns: distinct per pixel + per channel, in type range.
static float    f32_src(int z, int y, int x) { return (float)(z * 1000 + y * 10 + x); }
static uint8_t  u8_src (int z, int y, int x) { return (uint8_t)((z * 50 + y * 10 + x) & 0xff); }

// ---------------- generic case runner ---------------------------------------
// mapW/mapH default to W/H (Phase 1-3: src/dst/map share dims -> mapScale=1 ->
// integer mapCoord -> only NEAREST map path). Phase 4 passes mapW!=W so
// mapScale!=1 -> fractional mapCoord -> exercises map LINEAR/CUBIC sampling.
template <typename T, int C, T (*SrcFn)(int, int, int)>
static int run_case(sycl::queue& q, const std::string& name, int N, int H, int W,
                    cs::Interp si, cs::Interp mi, cs::MapValueType mvt, bool ac,
                    cs::Border b, cs::BorderValue bv,
                    void (*fill)(std::vector<float>&, int, int, int, int, int, int),
                    double tol, int mapW = -1, int mapH = -1) {
    if (mapW < 0) mapW = W;
    if (mapH < 0) mapH = H;

    std::vector<T> hsrc((size_t)N * H * W * C);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c)
                    hsrc[((z * H + y) * W + x) * C + c] = (T)(SrcFn(z, y, x) + c);

    // Map has its own dims (mapW x mapH); filler receives map dims.
    std::vector<float> hmap((size_t)N * mapH * mapW * 2);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < mapH; ++y)
            for (int x = 0; x < mapW; ++x)
                fill(hmap, mapW, mapH, N, z, x, y);

    std::vector<T> gdst((size_t)N * H * W * C, 0);
    std::vector<T> refdst((size_t)N * H * W * C, 0);

    cs::TensorView<T, C> sv{nullptr, N, H, W};
    cs::TensorView<T, C> dv{nullptr, N, H, W};
    cs::TensorView<float, 2> mv{nullptr, N, mapH, mapW};

    gpu_remap<T, C>(q, hsrc, hmap, sv, dv, mv, si, mi, mvt, ac, b, bv, N, gdst);

    cs::TensorView<T, C> hsv{hsrc.data(), N, H, W};
    cs::TensorView<T, C> hdv{nullptr, N, H, W};
    cs::TensorView<float, 2> hmv{hmap.data(), N, mapH, mapW};
    cpu_remap<T, C>(hsv, hdv, hmv, si, mi, mvt, ac, b, bv, N, refdst);

    double maxerr = 0;
    for (size_t i = 0; i < refdst.size(); ++i) {
        double e = std::fabs((double)gdst[i] - (double)refdst[i]);
        if (e > maxerr) maxerr = e;
    }
    bool pass = maxerr <= tol;
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    char dims[80];
    if (mapW != W || mapH != H)
        std::snprintf(dims, sizeof(dims), "%dx%dx%d(map%dx%d)", W, H, N, mapW, mapH);
    else
        std::snprintf(dims, sizeof(dims), "%dx%dx%d", W, H, N);
    std::printf("[%s] %-22s %s C%d %s  maxerr=%.3e\n",
                pass ? "PASS" : "FAIL", name.c_str(), dt, C, dims, maxerr);
    return pass ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Common phase-1/2 config: ABSOLUTE, NEAREST x NEAREST, REPLICATE.
    const cs::Interp   SI = cs::Interp::Nearest, MI = cs::Interp::Nearest;
    const cs::MapValueType MVT = cs::MapValueType::Absolute;
    const cs::Border   B  = cs::Border::Replicate;
    const bool AC = false;
    const double TOL = 0.0;  // NEAREST is exact copy
    const cs::BorderValue BV{};  // unused under REPLICATE

    int fails = 0;
    // ---- Phase 1: F32 C1 ----
    fails += run_case<float,1, f32_src>(q, "f32_identity",     1,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);
    fails += run_case<float,1, f32_src>(q, "f32_shift_x",      1,4,4, SI,MI,MVT,AC,B, BV, fill_shift_x,  TOL);
    fails += run_case<float,1, f32_src>(q, "f32_round_halfup", 1,4,4, SI,MI,MVT,AC,B, BV, fill_round,    TOL);
    fails += run_case<float,1, f32_src>(q, "f32_oor_replicate",1,4,4, SI,MI,MVT,AC,B, BV, fill_oor,      TOL);
    fails += run_case<float,1, f32_src>(q, "f32_batch2_id",    2,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);

    // ---- Phase 2: U8 C1/C3/C4 ----
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_identity",     1,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_shift_x",      1,4,4, SI,MI,MVT,AC,B, BV, fill_shift_x,  TOL);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_oor_replicate",1,4,4, SI,MI,MVT,AC,B, BV, fill_oor,      TOL);
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_identity",     1,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_batch2_id",    2,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);
    fails += run_case<uint8_t,4, u8_src>(q, "u8c4_identity",     1,4,4, SI,MI,MVT,AC,B, BV, fill_identity, TOL);

    // ---- Phase 3: src LINEAR/CUBIC + all 5 borders ----
    // Fractional map (x+0.5,y+0.5) -> interpolation; edges out-of-range -> border.
    // mapInterp stays NEAREST (Phase 4 covers map LINEAR/CUBIC). 6x6 gives interior
    // pixels with all-neighbors-in-range for CUBIC. F32 ~1e-4 (FP weighted sum),
    // U8 ±1 (round-half-to-even at x.5 boundaries — same Intel-GPU trait as Resize).
    const cs::Interp LIN = cs::Interp::Linear, CUB = cs::Interp::Cubic;
    const double TF = 1e-4, TU = 1.0;
    const cs::BorderValue BV7{{7.f,7.f,7.f,7.f}};       // F32 constant border = 7
    const cs::BorderValue BV200{{200.f,200.f,200.f,200.f}}; // U8 constant border = 200
    // F32 C1: LINEAR under each of the 5 borders.
    fails += run_case<float,1, f32_src>(q, "f32_lin_replicate",  1,6,6, LIN,MI,MVT,AC,cs::Border::Replicate,  BV,   fill_half_xy, TF);
    fails += run_case<float,1, f32_src>(q, "f32_lin_constant",   1,6,6, LIN,MI,MVT,AC,cs::Border::Constant,   BV7,  fill_half_xy, TF);
    fails += run_case<float,1, f32_src>(q, "f32_lin_reflect",    1,6,6, LIN,MI,MVT,AC,cs::Border::Reflect,    BV,   fill_half_xy, TF);
    fails += run_case<float,1, f32_src>(q, "f32_lin_reflect101", 1,6,6, LIN,MI,MVT,AC,cs::Border::Reflect101, BV,   fill_half_xy, TF);
    fails += run_case<float,1, f32_src>(q, "f32_lin_wrap",       1,6,6, LIN,MI,MVT,AC,cs::Border::Wrap,       BV,   fill_half_xy, TF);
    // F32 C1: CUBIC (4x4 neighborhood -> heavy border exercise).
    fails += run_case<float,1, f32_src>(q, "f32_cub_replicate",  1,6,6, CUB,MI,MVT,AC,cs::Border::Replicate,  BV,   fill_half_xy, TF);
    fails += run_case<float,1, f32_src>(q, "f32_cub_constant",   1,6,6, CUB,MI,MVT,AC,cs::Border::Constant,   BV7,  fill_half_xy, TF);
    // U8 C1: LINEAR/CUBIC + CONSTANT (validates SaturateCast on interpolated values).
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_lin_replicate",1,6,6, LIN,MI,MVT,AC,cs::Border::Replicate,  BV,   fill_half_xy, TU);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_cub_replicate",1,6,6, CUB,MI,MVT,AC,cs::Border::Replicate,  BV,   fill_half_xy, TU);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_lin_constant", 1,6,6, LIN,MI,MVT,AC,cs::Border::Constant,   BV200,fill_half_xy, TU);
    // U8 C3: LINEAR (multi-channel interpolation).
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_lin_replicate",1,6,6, LIN,MI,MVT,AC,cs::Border::Replicate,  BV,   fill_half_xy, TU);

    // ---- Phase 4: map LINEAR/CUBIC (map is float2, always REPLICATE border) ----
    // mapW!=dstW (4x4 map, 6x6 dst) -> mapScale=4/6 -> fractional mapCoord ->
    // the map itself is sampled with LINEAR/CUBIC (mixing distinct per-pixel
    // coords via fill_frac_abs). Previously all mapInterp was NEAREST.
    // srcInterp starts NEAREST (isolate map interp), then combined with src LINEAR.
    // Tolerances: same safety nets (F32 1e-4, U8 1.0); expected maxerr=0 since
    // sample<float,2> is the identical code path proven in Phase 3, and ABSOLUTE
    // srcCoord = mapValue exactly (no further FP). MW/MH = map dims.
    const int MW = 4, MH = 4;
    // F32 C1: map LINEAR / CUBIC, src NEAREST, REPLICATE.
    fails += run_case<float,1, f32_src>(q, "f32_maplin_nearestsrc", 1,6,6, cs::Interp::Nearest,LIN,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TF, MW,MH);
    fails += run_case<float,1, f32_src>(q, "f32_mapcub_nearestsrc", 1,6,6, cs::Interp::Nearest,CUB,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TF, MW,MH);
    // F32 C1: both interp (map CUBIC + src LINEAR).
    fails += run_case<float,1, f32_src>(q, "f32_mapcub_linsrc",     1,6,6, LIN,CUB,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TF, MW,MH);
    // F32 C1: map LINEAR + src CONSTANT border (map REPLICATE border exercised too).
    fails += run_case<float,1, f32_src>(q, "f32_maplin_constant",   1,6,6, cs::Interp::Nearest,LIN,MVT,AC,cs::Border::Constant,   BV7, fill_frac_abs, TF, MW,MH);
    // U8 C1: map LINEAR / CUBIC, src NEAREST, REPLICATE (SaturateCast on map-interp path).
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_maplin_nearestsrc",1,6,6, cs::Interp::Nearest,LIN,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TU, MW,MH);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_mapcub_nearestsrc",1,6,6, cs::Interp::Nearest,CUB,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TU, MW,MH);
    // U8 C3: map LINEAR (multi-channel).
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_maplin_nearestsrc",1,6,6, cs::Interp::Nearest,LIN,MVT,AC,cs::Border::Replicate, BV,  fill_frac_abs, TU, MW,MH);

    // ---- Phase 5: 3 mapValueType x alignCorners matrix ----
    // Absolute ignores alignCorners (its getRemapParams branch has no ac);
    // AbsNorm & RelNorm use it. mapW=dstW=6, NEAREST x NEAREST, REPLICATE ->
    // isolates the getRemapParams coordinate transform. RelNorm's mapScale is
    // (mapW-1)/dstW (differs from Absolute's mapW/dstW) -> exercises that branch.
    // fill_absnorm: mapValue in [-1,1] (0=center). fill_relnorm: small relative
    // offset; edges push srcCoord OOB -> REPLICATE. NEAREST -> integer pixel
    // read -> maxerr 0 if srcCoord bit-identical (same getRemapParams on host/dev).
    const cs::MapValueType ABSN = cs::MapValueType::AbsoluteNormalized;
    const cs::MapValueType RELN = cs::MapValueType::RelativeNormalized;
    // Absolute x alignCorners (ac irrelevant -> both should match identity).
    fails += run_case<float,1, f32_src>(q, "f32_abs_acF",      1,6,6, SI,SI,MVT,  false,cs::Border::Replicate, BV, fill_identity, TF);
    fails += run_case<float,1, f32_src>(q, "f32_abs_acT",      1,6,6, SI,SI,MVT,  true, cs::Border::Replicate, BV, fill_identity, TF);
    // AbsoluteNormalized x ac.
    fails += run_case<float,1, f32_src>(q, "f32_absn_acF",     1,6,6, SI,SI,ABSN, false,cs::Border::Replicate, BV, fill_absnorm,  TF);
    fails += run_case<float,1, f32_src>(q, "f32_absn_acT",     1,6,6, SI,SI,ABSN, true, cs::Border::Replicate, BV, fill_absnorm,  TF);
    // RelativeNormalized x ac.
    fails += run_case<float,1, f32_src>(q, "f32_reln_acF",     1,6,6, SI,SI,RELN, false,cs::Border::Replicate, BV, fill_relnorm,  TF);
    fails += run_case<float,1, f32_src>(q, "f32_reln_acT",     1,6,6, SI,SI,RELN, true, cs::Border::Replicate, BV, fill_relnorm,  TF);
    // AbsNorm + src LINEAR (mvt transform feeding interp).
    fails += run_case<float,1, f32_src>(q, "f32_absn_acF_lin", 1,6,6, LIN,SI,ABSN, false,cs::Border::Replicate, BV, fill_absnorm,  TF);
    // U8 on normalized paths (SaturateCast).
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_absn_acF",   1,6,6, SI,SI,ABSN, false,cs::Border::Replicate, BV, fill_absnorm,  TU);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_reln_acT",   1,6,6, SI,SI,RELN, true, cs::Border::Replicate, BV, fill_relnorm,  TU);

    if (fails == 0) std::printf("\n=== ALL TESTS PASSED ===\n");
    else            std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return fails ? 1 : 0;
}
