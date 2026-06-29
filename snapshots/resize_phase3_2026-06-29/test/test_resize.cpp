// Synthetic correctness test for the SYCL resize port.
// Phase 2: covers F32 and U8, channels C=1/3/4, NEAREST + LINEAR.
//
// Strategy (same as the lidar voxelization port's correctness test):
// build tiny synthetic images, run the GPU resize, and compare against a CPU
// reference that implements the EXACT same coordinate-mapping + interpolation
// math (and the SAME SaturateCast) as resize.hpp. If they match within
// tolerance, the kernel is correct.
//
// Tolerance notes:
//   - F32 LINEAR: ~1e-4 (float non-associativity in the weighted sum).
//   - U8 LINEAR:  ±1  (round-half-to-even at x.5 boundaries can flip by 1
//     between CPU/GPU float results — the same Intel-GPU FP characteristic the
//     voxelization port handled with ±1-neighbour analysis).
//   - NEAREST:    0   (pure copy, exact).
//
// Run on the Intel host:
//   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test_resize
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../resize.hpp"
#include "../common/tensorwrap.hpp"
#include "../common/saturate_cast.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace cs = cvsycl;

// ---------------- CPU reference (mirrors resize.hpp numerics) ----------------
template <typename T, int C>
static void cpu_nearest(const std::vector<T>& src, int sw, int sh,
                        int x, int y, int z, float sx, float sy, T* out) {
    int isx = (int)std::floor((x + 0.5f) * sx);
    int isy = (int)std::floor((y + 0.5f) * sy);
    isx = std::min(isx, sw - 1);
    isy = std::min(isy, sh - 1);
    const T* p = src.data() + ((z * sh + isy) * sw + isx) * C;
    for (int c = 0; c < C; ++c) out[c] = p[c];
}

template <typename T, int C>
static void cpu_linear(const std::vector<T>& src, int sw, int sh,
                       int x, int y, int z, float sx, float sy, T* out) {
    float fx = (x + 0.5f) * sx - 0.5f;
    float fy = (y + 0.5f) * sy - 0.5f;
    int isx = (int)std::floor(fx);
    int isy = (int)std::floor(fy);
    float wx = (isx < 0) ? 0.f : (isx > sw - 2) ? 1.f : (fx - isx);
    float wy = (isy < 0) ? 0.f : (isy > sh - 2) ? 1.f : (fy - isy);
    isx = std::max(0, std::min(isx, sw - 2));
    isy = std::max(0, std::min(isy, sh - 2));
    auto at = [&](int xx, int yy, int c) -> float {
        return (float)src[((z * sh + yy) * sw + xx) * C + c];
    };
    float w00 = (1.f - wx) * (1.f - wy);
    float w10 = wx * (1.f - wy);
    float w01 = (1.f - wx) * wy;
    float w11 = wx * wy;
    for (int c = 0; c < C; ++c) {
        float v = at(isx, isy, c) * w00 + at(isx + 1, isy, c) * w10
                + at(isx, isy + 1, c) * w01 + at(isx + 1, isy + 1, c) * w11;
        out[c] = cs::SaturateCast<T>(v);
    }
}

template <typename T, int C>
static void cpu_cubic(const std::vector<T>& src, int sw, int sh,
                      int x, int y, int z, float sx, float sy, T* out) {
    float coordX = (x + 0.5f) * sx - 0.5f;
    float coordY = (y + 0.5f) * sy - 0.5f;
    int isx = (int)std::floor(coordX);
    int isy = (int)std::floor(coordY);
    float fx = coordX - isx;
    fx = (isx < 1 || isx >= sw - 3) ? 0.f : fx;   // asymmetric edge (matches CUDA)
    isy = std::max(1, std::min(isy, sh - 3));
    isx = std::max(1, std::min(isx, sw - 3));
    float wx[4], wy[4];
    cs::get_cubic_coeffs(fx, wx);
    cs::get_cubic_coeffs(coordY - isy, wy);
    auto at = [&](int xx, int yy, int c) -> float {
        return (float)src[((z * sh + yy) * sw + xx) * C + c];
    };
    for (int c = 0; c < C; ++c) {
        float sum = 0.f;
        for (int cy = -1; cy <= 2; ++cy)
            for (int cx = -1; cx <= 2; ++cx)
                sum += at(isx + cx, isy + cy, c) * (wx[cx + 1] * wy[cy + 1]);
        out[c] = cs::SaturateCast<T>(std::fabs(sum));
    }
}

// ---------------- GPU runner ----------------
template <typename T, int C>
static std::vector<T> run_gpu(sycl::queue& q, const std::vector<T>& src,
                              int n, int sh, int sw, int dh, int dw, cs::Interp interp) {
    size_t src_n = (size_t)n * sh * sw * C;
    size_t dst_n = (size_t)n * dh * dw * C;

    T* d_src = sycl::malloc_device<T>(src_n, q);
    T* d_dst = sycl::malloc_device<T>(dst_n, q);
    if (!d_src || !d_dst) throw std::runtime_error("USM alloc failed");

    q.copy<T>(src.data(), d_src, src_n).wait();

    cs::TensorView<T, C> sv{d_src, n, sh, sw};
    cs::TensorView<T, C> dv{d_dst, n, dh, dw};
    cs::run_resize<T, C>(q, sv, dv, interp).wait();

    std::vector<T> out(dst_n);
    q.copy<T>(d_dst, out.data(), dst_n).wait();

    sycl::free(d_src, q);
    sycl::free(d_dst, q);
    return out;
}

// ---------------- one test case ----------------
template <typename T, int C>
static int run_case(sycl::queue& q, const std::string& name, int n, int sh, int sw,
                    int dh, int dw, const std::vector<T>& src, cs::Interp interp, float tol) {
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;

    std::vector<T> gpu = run_gpu<T, C>(q, src, n, sh, sw, dh, dw, interp);

    int mism = 0;
    float maxerr = 0.f;
    T ref[C];
    for (int z = 0; z < n; ++z)
    for (int y = 0; y < dh; ++y)
    for (int x = 0; x < dw; ++x) {
        if (interp == cs::Interp::Nearest)
            cpu_nearest<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
        else if (interp == cs::Interp::Linear)
            cpu_linear<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
        else
            cpu_cubic<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
        for (int c = 0; c < C; ++c) {
            T got = gpu[((z * dh + y) * dw + x) * C + c];
            float err = std::fabs((float)got - (float)ref[c]);
            maxerr = std::max(maxerr, err);
            if (err > tol) {
                if (mism < 8) {
                    std::printf("  MISMATCH [%s] z=%d y=%d x=%d c=%d: gpu=%d ref=%d err=%.2e\n",
                                name.c_str(), z, y, x, c, (int)got, (int)ref[c], err);
                }
                ++mism;
            }
        }
    }

    const char* ip = (interp == cs::Interp::Nearest) ? "NEAR"
                    : (interp == cs::Interp::Linear) ? "LIN " : "CUB ";
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    if (mism == 0) {
        std::printf("[PASS] %-26s %s C%d %dx%d->%dx%d (N=%d)  maxerr=%.2e\n",
                    name.c_str(), dt, C, sw, sh, dw, dh, n, maxerr);
        return 0;
    }
    std::printf("[FAIL] %-26s %s C%d  %d mismatches, maxerr=%.2e\n",
                name.c_str(), dt, C, mism, maxerr);
    return 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;

    // ===== F32, C=1 (phase-1 cases, kept) =====
    fails += run_case<float,1>(q, "f32_up2x_near", 1,2,2,4,4, {1.f,3.f,7.f,9.f}, cs::Interp::Nearest, 0.f);
    fails += run_case<float,1>(q, "f32_up2x_lin",  1,2,2,4,4, {1.f,3.f,7.f,9.f}, cs::Interp::Linear, 1e-4f);
    fails += run_case<float,1>(q, "f32_nonint_lin",1,3,3,5,5, {0,10,20, 30,40,50, 60,70,80}, cs::Interp::Linear, 1e-4f);
    fails += run_case<float,1>(q, "f32_down_near", 1,4,4,2,2, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, cs::Interp::Nearest, 0.f);
    fails += run_case<float,1>(q, "f32_down_lin",  1,4,4,2,2, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, cs::Interp::Linear, 1e-4f);
    fails += run_case<float,1>(q, "f32_batch2_lin",2,2,2,4,4, {1.f,3.f,7.f,9.f, 0.f,100.f,200.f,255.f}, cs::Interp::Linear, 1e-3f);

    // ===== U8, C=1 =====
    fails += run_case<std::uint8_t,1>(q, "u8_up2x_near", 1,2,2,4,4, {10,200,50,255}, cs::Interp::Nearest, 0.f);
    fails += run_case<std::uint8_t,1>(q, "u8_up2x_lin",  1,2,2,4,4, {10,200,50,255}, cs::Interp::Linear, 1.f);
    fails += run_case<std::uint8_t,1>(q, "u8_down_lin",  1,4,4,2,2, {0,40,80,120,160,200,240,255,0,40,80,120,160,200,240,255}, cs::Interp::Linear, 1.f);

    // ===== U8, C=3 (RGB) — 2x2 -> 4x4 =====
    {
        // pixel(0,0)=(10,20,30) (1,0)=(200,100,50) (0,1)=(60,70,80) (1,1)=(255,0,128)
        std::vector<std::uint8_t> img = {10,20,30,  200,100,50,  60,70,80,  255,0,128};
        fails += run_case<std::uint8_t,3>(q, "u8c3_up2x_near", 1,2,2,4,4, img, cs::Interp::Nearest, 0.f);
        fails += run_case<std::uint8_t,3>(q, "u8c3_up2x_lin",  1,2,2,4,4, img, cs::Interp::Linear, 1.f);
    }

    // ===== U8, C=4 (RGBA) — 2x2 -> 4x4 =====
    {
        std::vector<std::uint8_t> img = {10,20,30,255,  200,100,50,128,  60,70,80,0,  255,0,128,200};
        fails += run_case<std::uint8_t,4>(q, "u8c4_up2x_lin", 1,2,2,4,4, img, cs::Interp::Linear, 1.f);
    }

    // ===== F32, C=3 — 3x3 -> 5x5 =====
    {
        std::vector<float> img;
        float v = 0.f;
        for (int i = 0; i < 3*3*3; ++i) img.push_back(v), v += 7.f;
        fails += run_case<float,3>(q, "f32c3_nonint_lin", 1,3,3,5,5, img, cs::Interp::Linear, 1e-3f);
    }

    // ===== CUBIC (needs src W,H >= 4) =====
    // 4x4 ramp F32 C1 -> 8x8 and 6x6; downscale 5x5 -> 3x3.
    {
        std::vector<float> img{0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};
        fails += run_case<float,1>(q, "f32_cubic_up2x",  1,4,4,8,8, img, cs::Interp::Cubic, 1e-3f);
        fails += run_case<float,1>(q, "f32_cubic_nonint",1,4,4,6,6, img, cs::Interp::Cubic, 1e-3f);
    }
    {
        std::vector<float> img{0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24};
        fails += run_case<float,1>(q, "f32_cubic_down", 1,5,5,3,3, img, cs::Interp::Cubic, 1e-3f);
    }
    // U8 C1 cubic
    {
        std::vector<std::uint8_t> img{0,60,120,180, 20,80,140,200, 40,100,160,220, 60,120,180,240};
        fails += run_case<std::uint8_t,1>(q, "u8_cubic_up2x", 1,4,4,8,8, img, cs::Interp::Cubic, 1.f);
    }
    // U8 C3 cubic
    {
        std::vector<std::uint8_t> img;
        for (int i = 0; i < 4*4*3; ++i) img.push_back((std::uint8_t)((i * 7) % 256));
        fails += run_case<std::uint8_t,3>(q, "u8c3_cubic_up2x", 1,4,4,8,8, img, cs::Interp::Cubic, 1.f);
    }
    // F32 C3 cubic
    {
        std::vector<float> img;
        for (int i = 0; i < 4*4*3; ++i) img.push_back((float)(i * 1.5f));
        fails += run_case<float,3>(q, "f32c3_cubic_up2x", 1,4,4,8,8, img, cs::Interp::Cubic, 1e-2f);
    }

    if (fails == 0) {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    }
    std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return 1;
}
