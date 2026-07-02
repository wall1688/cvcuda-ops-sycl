// Performance profiling for the SYCL resize port.
// Uses a SYCL queue with enable_profiling to measure device kernel time
// (best-of-N) for each interpolation on large images, plus dst throughput.
//
// Same measurement discipline as the lidar voxelization port's profile test:
// warmup then best-of-N device-time (via event profiling), because the Arc GPU
// has cross-run ±10%+ clock/thermal noise — best-of-N is more stable than avg.
//
// Run on the Intel host:
//   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_resize_profile
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../resize.hpp"
#include "../common/tensorwrap.hpp"
#include "resize_gold.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <string>

namespace cs = cvsycl;

static const char* interp_name(cs::Interp i) {
    switch (i) {
        case cs::Interp::Nearest: return "NEAREST";
        case cs::Interp::Linear:  return "LINEAR ";
        case cs::Interp::Cubic:   return "CUBIC  ";
        case cs::Interp::Area:    return "AREA   ";
    }
    return "?";
}

// Reads-per-dst-pixel (for memory-throughput estimate; area is a rough avg).
static int reads_per_dst(cs::Interp i) {
    switch (i) {
        case cs::Interp::Nearest: return 1;
        case cs::Interp::Linear:  return 4;
        case cs::Interp::Cubic:   return 16;
        case cs::Interp::Area:    return 4;   // downscale ~scale^2; rough
    }
    return 0;
}

template <typename T, int C>
static void bench(sycl::queue& q, int sh, int sw, int dh, int dw,
                  cs::Interp interp, int iters) {
    size_t src_n = (size_t)sh * sw * C;
    size_t dst_n = (size_t)dh * dw * C;

    std::vector<T> hsrc(src_n);
    for (size_t i = 0; i < src_n; ++i) hsrc[i] = (T)(i & 0xff);

    T* d_src = sycl::malloc_device<T>(src_n, q);
    T* d_dst = sycl::malloc_device<T>(dst_n, q);
    q.copy<T>(hsrc.data(), d_src, src_n).wait();

    cs::TensorView<T, C> sv{d_src, 1, sh, sw};
    cs::TensorView<T, C> dv{d_dst, 1, dh, dw};

    // warmup
    for (int i = 0; i < 5; ++i) cs::run_resize<T, C>(q, sv, dv, interp);
    q.wait();

    double best = 1e30, sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = cs::run_resize<T, C>(q, sv, dv, interp);
        e.wait();
        uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        double ms = (double)(end - start) / 1e6;
        best = std::min(best, ms);
        sum += ms;
    }
    double avg = sum / iters;

    double dst_pixels = (double)dh * dw;
    double gpix = dst_pixels / (best * 1e-3) / 1e9;          // dst Gpix/s
    double bytes = (double)(reads_per_dst(interp) + 1) * dst_n * sizeof(T);
    double gbs = bytes / (best * 1e-3) / 1e9;                // effective GB/s

    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    std::printf("  %s C%d %-7s %dx%d->%dx%d  best=%.4f avg=%.4f ms  %.2f Gpix/s  %.1f GB/s\n",
                dt, C, interp_name(interp), sw, sh, dw, dh, best, avg, gpix, gbs);

    // Optional CPU gold reference timing (single-thread, 1 run). 4K CUBIC can
    // take ~tens of seconds, hence 1 run, not best-of-N.
    const char* ce = std::getenv("VOX_CPUTIME");
    if (ce && ce[0] == '1') {
        std::vector<T> cpu;
        auto t0 = std::chrono::steady_clock::now();
        resize_gold::gold_resize<T, C>(cpu, hsrc, 1, sh, sw, dh, dw, interp);
        auto t1 = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("          [CPU ref time: %.3f ms, single-thread, 1 run]  speedup = %.2fx\n",
                    cpu_ms, cpu_ms / best);
    }

    sycl::free(d_src, q);
    sycl::free(d_dst, q);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("profiling: SYCL event (device kernel time), best-of-20\n\n");

    const int iters = 20;
    const int SW = 1920, SH = 1080;

    // --- F32 C1: downscale 2x ---
    std::printf("[F32 C1  downscale 1920x1080 -> 960x540]\n");
    bench<float, 1>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Nearest, iters);
    bench<float, 1>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Linear,  iters);
    bench<float, 1>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Cubic,   iters);
    bench<float, 1>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Area,    iters);

    // --- F32 C1: upscale 2x ---
    std::printf("\n[F32 C1  upscale 1920x1080 -> 3840x2160]\n");
    bench<float, 1>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Nearest, iters);
    bench<float, 1>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Linear,  iters);
    bench<float, 1>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Cubic,   iters);

    // --- U8 C3: downscale 2x (typical video path) ---
    std::printf("\n[U8 C3   downscale 1920x1080 -> 960x540]\n");
    bench<std::uint8_t, 3>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Nearest, iters);
    bench<std::uint8_t, 3>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Linear,  iters);
    bench<std::uint8_t, 3>(q, SH, SW, SH / 2, SW / 2, cs::Interp::Area,    iters);

    // --- U8 C3: upscale 2x ---
    std::printf("\n[U8 C3   upscale 1920x1080 -> 3840x2160]\n");
    bench<std::uint8_t, 3>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Nearest, iters);
    bench<std::uint8_t, 3>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Linear,  iters);

    // --- U8 C1: NIX=16 path check ---
    std::printf("\n[U8 C1   upscale 1920x1080 -> 3840x2160]\n");
    bench<std::uint8_t, 1>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Nearest, iters);
    bench<std::uint8_t, 1>(q, SH, SW, SH * 2, SW * 2, cs::Interp::Linear,  iters);

    std::printf("\n[done]\n");
    return 0;
}
