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

// src patterns: distinct per pixel + per channel, in type range.
static float    f32_src(int z, int y, int x) { return (float)(z * 1000 + y * 10 + x); }
static uint8_t  u8_src (int z, int y, int x) { return (uint8_t)((z * 50 + y * 10 + x) & 0xff); }

// ---------------- generic case runner ---------------------------------------
template <typename T, int C, T (*SrcFn)(int, int, int)>
static int run_case(sycl::queue& q, const std::string& name, int N, int H, int W,
                    cs::Interp si, cs::Interp mi, cs::MapValueType mvt, bool ac,
                    cs::Border b, void (*fill)(std::vector<float>&, int, int, int, int, int, int),
                    double tol) {
    std::vector<T> hsrc((size_t)N * H * W * C);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c)
                    hsrc[((z * H + y) * W + x) * C + c] = (T)(SrcFn(z, y, x) + c);

    std::vector<float> hmap((size_t)N * H * W * 2);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                fill(hmap, W, H, N, z, x, y);

    std::vector<T> gdst((size_t)N * H * W * C, 0);
    std::vector<T> refdst((size_t)N * H * W * C, 0);

    cs::TensorView<T, C> sv{nullptr, N, H, W};
    cs::TensorView<T, C> dv{nullptr, N, H, W};
    cs::TensorView<float, 2> mv{nullptr, N, H, W};
    cs::BorderValue bv{};  // unused: REPLICATE

    gpu_remap<T, C>(q, hsrc, hmap, sv, dv, mv, si, mi, mvt, ac, b, bv, N, gdst);

    cs::TensorView<T, C> hsv{hsrc.data(), N, H, W};
    cs::TensorView<T, C> hdv{nullptr, N, H, W};
    cs::TensorView<float, 2> hmv{hmap.data(), N, H, W};
    cpu_remap<T, C>(hsv, hdv, hmv, si, mi, mvt, ac, b, bv, N, refdst);

    double maxerr = 0;
    for (size_t i = 0; i < refdst.size(); ++i) {
        double e = std::fabs((double)gdst[i] - (double)refdst[i]);
        if (e > maxerr) maxerr = e;
    }
    bool pass = maxerr <= tol;
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    std::printf("[%s] %-22s %s C%d %dx%dx%d  maxerr=%.3e\n",
                pass ? "PASS" : "FAIL", name.c_str(), dt, C, W, H, N, maxerr);
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

    int fails = 0;
    // ---- Phase 1: F32 C1 ----
    fails += run_case<float,1, f32_src>(q, "f32_identity",     1,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);
    fails += run_case<float,1, f32_src>(q, "f32_shift_x",      1,4,4, SI,MI,MVT,AC,B, fill_shift_x,  TOL);
    fails += run_case<float,1, f32_src>(q, "f32_round_halfup", 1,4,4, SI,MI,MVT,AC,B, fill_round,    TOL);
    fails += run_case<float,1, f32_src>(q, "f32_oor_replicate",1,4,4, SI,MI,MVT,AC,B, fill_oor,      TOL);
    fails += run_case<float,1, f32_src>(q, "f32_batch2_id",    2,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);

    // ---- Phase 2: U8 C1/C3/C4 ----
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_identity",     1,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_shift_x",      1,4,4, SI,MI,MVT,AC,B, fill_shift_x,  TOL);
    fails += run_case<uint8_t,1, u8_src>(q, "u8c1_oor_replicate",1,4,4, SI,MI,MVT,AC,B, fill_oor,      TOL);
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_identity",     1,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);
    fails += run_case<uint8_t,3, u8_src>(q, "u8c3_batch2_id",    2,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);
    fails += run_case<uint8_t,4, u8_src>(q, "u8c4_identity",     1,4,4, SI,MI,MVT,AC,B, fill_identity, TOL);

    if (fails == 0) std::printf("\n=== ALL TESTS PASSED ===\n");
    else            std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return fails ? 1 : 0;
}
