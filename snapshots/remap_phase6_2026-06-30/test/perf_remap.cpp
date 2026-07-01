// Performance benchmark for the SYCL remap port (Phase 6).
//
// Measures output throughput (Gpix/s) and effective memory bandwidth (GB/s,
// counting ALL src/map/dst bytes touched — including the redundant reads a
// LINEAR/CUBIC interpolation makes — so GB/s reflects real memory traffic, not
// just unique bytes) across interp x border x dtype configs on a large image.
//
// Setup: ABSOLUTE map, mapW=dstW, mapInterp=NEAREST, identity map (mapVx=x,
// mapVy=y) -> srcCoord=(x,y) integer. This isolates the SRC interp cost (the
// map is a cheap 1-fetch NEAREST lookup). mapNumSamples=1 (broadcast).
//
// Arc GPU perf note (see memory arc-gpu-perf-measurement-noise): cross-run
// variance is ~10%+, so we warm up then take the BEST of N iterations.
//
// Run: ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/perf_remap
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../remap.hpp"
#include "../common/tensorwrap.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace cs = cvsycl;

template <typename T, int C>
static double bench(sycl::queue& q, int N, int H, int W,
                    cs::Interp si, cs::Interp mi, cs::Border b, int iters,
                    double& out_gpix, double& out_gbs) {
    std::vector<T> hsrc((size_t)N * H * W * C);
    for (size_t i = 0; i < hsrc.size(); ++i)
        hsrc[i] = (T)((i % 255) + 1);  // nonzero, in type range

    // Identity ABSOLUTE map: srcCoord = (x, y). mapW=dstW -> mapScale=1 ->
    // integer mapCoord -> mi=NEAREST picks map[x,y] (1 fetch).
    std::vector<float> hmap((size_t)H * W * 2);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t k = ((size_t)y * W + x) * 2;
            hmap[k] = (float)x;
            hmap[k + 1] = (float)y;
        }

    T* d_src = sycl::malloc_device<T>(hsrc.size(), q);
    T* d_dst = sycl::malloc_device<T>(hsrc.size(), q);
    float* d_map = sycl::malloc_device<float>(hmap.size(), q);
    q.copy(hsrc.data(), d_src, hsrc.size());
    q.copy(hmap.data(), d_map, hmap.size());
    q.wait();

    cs::TensorView<T, C> sv{d_src, N, H, W};
    cs::TensorView<T, C> dv{d_dst, N, H, W};
    cs::TensorView<float, 2> mv{d_map, 1, H, W};
    cs::BorderValue bv{};

    // Warmup (5 iters) — clears JIT/cold-cache so the first config isn't penalized.
    for (int i = 0; i < 5; ++i)
        cs::run_remap<T, C>(q, sv, dv, mv, si, mi, cs::MapValueType::Absolute,
                            false, b, bv, 1);
    q.wait();

    // Timed: profile each kernel individually, wait between, take the BEST.
    // Best-of-N is the most reproducible on Arc (cross-run noise ~10%+); waiting
    // between kernels serializes them but for a ~1ms kernel launch overhead is
    // ~1%, and the metric is consistent for before/after A/B.
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = cs::run_remap<T, C>(q, sv, dv, mv, si, mi,
                        cs::MapValueType::Absolute, false, b, bv, 1);
        q.wait();
        double ms = (double)(
            e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
        if (ms < best) best = ms;
    }
    double ms = best;

    double pixels = (double)N * H * W;
    out_gpix = pixels / (ms * 1e-3) / 1e9;

    int fetch_si = (si == cs::Interp::Nearest) ? 1 : (si == cs::Interp::Linear ? 4 : 16);
    int fetch_mi = (mi == cs::Interp::Nearest) ? 1 : (mi == cs::Interp::Linear ? 4 : 16);
    double bytes = (double)N * H * W * C * sizeof(T) * fetch_si   // src reads
                 + (double)H * W * 8 * fetch_mi                   // map reads (broadcast)
                 + (double)N * H * W * C * sizeof(T);             // dst writes
    out_gbs = bytes / (ms * 1e-3) / 1e9;

    sycl::free(d_src, q); sycl::free(d_dst, q); sycl::free(d_map, q);
    return ms;
}

static const char* iname(cs::Interp i) {
    return i == cs::Interp::Nearest ? "NEAR" : (i == cs::Interp::Linear ? "LIN " : "CUB ");
}
static const char* bname(cs::Border b) {
    switch (b) {
        case cs::Border::Replicate:   return "REPL";
        case cs::Border::Constant:    return "CONS";
        case cs::Border::Reflect:     return "REFL";
        case cs::Border::Wrap:        return "WRAP";
        case cs::Border::Reflect101:  return "RF01";
    }
    return "?";
}

template <typename T, int C>
static void run(sycl::queue& q, const char* tag, int N, int H, int W,
                cs::Interp si, cs::Interp mi, cs::Border b, int iters) {
    double gpix = 0, gbs = 0, ms = bench<T, C>(q, N, H, W, si, mi, b, iters, gpix, gbs);
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    std::printf("%-8s %s C%d  src=%s map=%s b=%s  %dx%dx%d  %7.3f ms  %6.2f Gpix/s  %6.1f GB/s\n",
                tag, dt, C, iname(si), iname(mi), bname(b), W, H, N, ms, gpix, gbs);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  {sycl::property::queue::in_order{},
                   sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int N = 4, H = 1080, W = 1920, ITERS = 20;
    std::printf("image %dx%d x%d batch, best of %d iters\n\n", W, H, N, ITERS);

    const cs::Interp NE = cs::Interp::Nearest, LI = cs::Interp::Linear, CU = cs::Interp::Cubic;
    const cs::Interp MN = cs::Interp::Nearest;  // map always NEAREST here (isolate src)
    const cs::Border REP = cs::Border::Replicate, RFL = cs::Border::Reflect;

    // F32 C1: interp scaling under REPLICATE (border math cheap).
    run<float,1>(q, "f32c1", N,H,W, NE,MN,REP, ITERS);
    run<float,1>(q, "f32c1", N,H,W, LI,MN,REP, ITERS);
    run<float,1>(q, "f32c1", N,H,W, CU,MN,REP, ITERS);
    // F32 C1: LINEAR under REFLECT (expensive integer-modulo border) vs REPLICATE.
    run<float,1>(q, "f32c1", N,H,W, LI,MN,RFL, ITERS);
    run<float,1>(q, "f32c1", N,H,W, CU,MN,RFL, ITERS);
    // U8 C4: dtype/channel effect.
    run<uint8_t,4>(q, "u8c4",  N,H,W, NE,MN,REP, ITERS);
    run<uint8_t,4>(q, "u8c4",  N,H,W, LI,MN,REP, ITERS);
    run<uint8_t,4>(q, "u8c4",  N,H,W, CU,MN,REP, ITERS);
    return 0;
}
