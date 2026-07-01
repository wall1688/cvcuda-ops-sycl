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
// Phase 1 coverage: F32 C1, ABSOLUTE mapValueType, NEAREST x NEAREST,
// REPLICATE border. The kernel/sampler are general; later phases expand the
// test matrix (U8/C3/C4, LINEAR/CUBIC, all borders, normalized mapValueType,
// var-shape).
//
// Tolerance:
//   - NEAREST (Phase 1): exact (0) — pure copy with index rounding, no FP sum.
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
// map layout: NHWC, C=2. map(z,y,x) = (valX, valY).
static void map_set(std::vector<float>& map, int W, int H, int N, int z, int x, int y,
                    float vx, float vy) {
    size_t idx = ((size_t)z * H * W + y * W + x) * 2;
    map[idx] = vx; map[idx + 1] = vy;
}

// src pattern: unique recognizable value per pixel.
static float src_val(int z, int y, int x) { return (float)(z * 1000 + y * 10 + x); }

// ---------------- a single F32 C1 case --------------------------------------
struct Case {
    std::string name;
    int N, H, W;               // src/dst/map all same size in phase 1
    cs::Interp si, mi;
    cs::MapValueType mvt;
    bool alignCorners;
    cs::Border border;
    // how to fill map pixel (dx,dy) in sample z -> (vx,vy)
    void (*fill)(std::vector<float>&, int, int, int, int, int, int);
};

static void fill_identity(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    map_set(m, W, H, N, z, x, y, (float)x, (float)y);
}
static void fill_shift_x(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    map_set(m, W, H, N, z, x, y, (float)(x + 1), (float)y);
}
static void fill_round(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    // even x -> +0.4 (rounds down to x), odd x -> +0.6 (rounds up to x+1)
    float off = (x % 2 == 0) ? 0.4f : 0.6f;
    map_set(m, W, H, N, z, x, y, (float)x + off, (float)y);
}
static void fill_oor(std::vector<float>& m, int W, int H, int N, int z, int x, int y) {
    // left half points off-left (-2 -> clamp 0), right half off-right (W+2 -> clamp W-1)
    float vx = (x < W / 2) ? -2.f : (float)(W + 2);
    map_set(m, W, H, N, z, x, y, vx, (float)y);
}

static int run_case_f32c1(sycl::queue& q, const Case& c, float tol) {
    const int N = c.N, H = c.H, W = c.W, C = 1;
    std::vector<float> hsrc((size_t)N * H * W * C);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                hsrc[((z * H + y) * W + x)] = src_val(z, y, x);

    std::vector<float> hmap((size_t)N * H * W * 2);
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                c.fill(hmap, W, H, N, z, x, y);

    std::vector<float> gdst((size_t)N * H * W * C, 0);
    std::vector<float> refdst((size_t)N * H * W * C, 0);

    cs::TensorView<float, 1> sv{nullptr, N, H, W};
    cs::TensorView<float, 1> dv{nullptr, N, H, W};
    cs::TensorView<float, 2> mv{nullptr, N, H, W};
    cs::BorderValue bv{};  // unused: REPLICATE

    gpu_remap<float, 1>(q, hsrc, hmap, sv, dv, mv, c.si, c.mi, c.mvt, c.alignCorners,
                        c.border, bv, N, gdst);

    // CPU ref over host pointers.
    cs::TensorView<float, 1> hsv{hsrc.data(), N, H, W};
    cs::TensorView<float, 1> hdv{nullptr, N, H, W};
    cs::TensorView<float, 2> hmv{hmap.data(), N, H, W};
    cpu_remap<float, 1>(hsv, hdv, hmv, c.si, c.mi, c.mvt, c.alignCorners, c.border, bv, N, refdst);

    double maxerr = 0;
    for (size_t i = 0; i < refdst.size(); ++i) {
        double e = std::fabs((double)gdst[i] - (double)refdst[i]);
        if (e > maxerr) maxerr = e;
    }
    bool pass = maxerr <= tol;
    std::printf("[%s] %-26s F32 C1 %dx%dx%d  maxerr=%.3e\n",
                pass ? "PASS" : "FAIL", c.name.c_str(), W, H, N, maxerr);
    return pass ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Phase 1: F32 C1, ABSOLUTE, NEAREST x NEAREST, REPLICATE.
    std::vector<Case> cases = {
        {"f32_identity",     1, 4, 4, cs::Interp::Nearest, cs::Interp::Nearest,
                              cs::MapValueType::Absolute, false, cs::Border::Replicate, fill_identity},
        {"f32_shift_x",      1, 4, 4, cs::Interp::Nearest, cs::Interp::Nearest,
                              cs::MapValueType::Absolute, false, cs::Border::Replicate, fill_shift_x},
        {"f32_round_halfup", 1, 4, 4, cs::Interp::Nearest, cs::Interp::Nearest,
                              cs::MapValueType::Absolute, false, cs::Border::Replicate, fill_round},
        {"f32_oor_replicate",1, 4, 4, cs::Interp::Nearest, cs::Interp::Nearest,
                              cs::MapValueType::Absolute, false, cs::Border::Replicate, fill_oor},
        {"f32_batch2_id",    2, 4, 4, cs::Interp::Nearest, cs::Interp::Nearest,
                              cs::MapValueType::Absolute, false, cs::Border::Replicate, fill_identity},
    };

    int fails = 0;
    for (const auto& c : cases) fails += run_case_f32c1(q, c, 0.0);
    // NEAREST is exact copy -> tol 0.

    if (fails == 0) std::printf("\n=== ALL TESTS PASSED ===\n");
    else            std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return fails ? 1 : 0;
}
