// Var-shape correctness test for the SYCL resize port.
// N images, each with its own src/dst dimensions; pitched VarShapeBatch layout.
// Compares GPU output to a CPU reference using the SAME interp math + layout.
//
// Run on the Intel host:
//   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_resize_varshape
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../resize.hpp"
#include "../resize_varshape.hpp"
#include "../common/varshape.hpp"
#include "../common/saturate_cast.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace cs = cvsycl;

// CPU reference: one dst pixel of image n. src is pitched host buffer.
template <typename T, int C>
static void cpu_ref(const std::vector<T>& src, int /*N*/, int maxSW, int maxSH,
                    const int* sw, const int* sh, int n, int x, int y,
                    int dw, int dh, cs::Interp interp, T* out) {
    float sx = (float)sw[n] / dw, sy = (float)sh[n] / dh;
    auto at = [&](int xx, int yy, int c) -> float {
        // pitched: pixel (xx,yy,n,c)
        return (float)src[((n * maxSH + yy) * maxSW + xx) * C + c];
    };
    if (interp == cs::Interp::Nearest) {
        int isx = (int)std::floor((x + 0.5f) * sx); isx = std::min(isx, sw[n] - 1);
        int isy = (int)std::floor((y + 0.5f) * sy); isy = std::min(isy, sh[n] - 1);
        for (int c = 0; c < C; ++c) out[c] = (T)at(isx, isy, c);
    } else if (interp == cs::Interp::Linear) {
        float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
        int isx = (int)std::floor(cx), isy = (int)std::floor(cy);
        float wx = (isx < 0) ? 0.f : (isx > sw[n] - 2) ? 1.f : (cx - isx);
        float wy = (isy < 0) ? 0.f : (isy > sh[n] - 2) ? 1.f : (cy - isy);
        isx = std::max(0, std::min(isx, sw[n] - 2));
        isy = std::max(0, std::min(isy, sh[n] - 2));
        float w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy), w01 = (1 - wx) * wy, w11 = wx * wy;
        for (int c = 0; c < C; ++c) {
            float v = at(isx, isy, c) * w00 + at(isx + 1, isy, c) * w10
                    + at(isx, isy + 1, c) * w01 + at(isx + 1, isy + 1, c) * w11;
            out[c] = cs::SaturateCast<T>(v);
        }
    } else {  // Cubic
        float cx = (x + 0.5f) * sx - 0.5f, cy = (y + 0.5f) * sy - 0.5f;
        int isx = (int)std::floor(cx), isy = (int)std::floor(cy);
        float fx = cx - isx;
        fx = (isx < 1 || isx >= sw[n] - 3) ? 0.f : fx;
        isy = std::max(1, std::min(isy, sh[n] - 3));
        isx = std::max(1, std::min(isx, sw[n] - 3));
        float wx[4], wy[4];
        cs::get_cubic_coeffs(fx, wx);
        cs::get_cubic_coeffs(cy - isy, wy);
        for (int c = 0; c < C; ++c) {
            float sum = 0.f;
            for (int cyi = -1; cyi <= 2; ++cyi)
                for (int cxi = -1; cxi <= 2; ++cxi)
                    sum += at(isx + cxi, isy + cyi, c) * (wx[cxi + 1] * wy[cyi + 1]);
            out[c] = cs::SaturateCast<T>(std::fabs(sum));
        }
    }
}

template <typename T, int C>
static int run_case(sycl::queue& q, const std::string& name, int N,
                    const std::vector<int>& sw, const std::vector<int>& sh,
                    const std::vector<int>& dw, const std::vector<int>& dh,
                    cs::Interp interp, float tol) {
    int maxSW = 0, maxSH = 0, maxDW = 0, maxDH = 0;
    for (int n = 0; n < N; ++n) {
        maxSW = std::max(maxSW, sw[n]); maxSH = std::max(maxSH, sh[n]);
        maxDW = std::max(maxDW, dw[n]); maxDH = std::max(maxDH, dh[n]);
    }
    size_t src_n = (size_t)N * maxSW * maxSH * C;
    size_t dst_n = (size_t)N * maxDW * maxDH * C;

    // Build pitched src host buffer with a pattern.
    std::vector<T> hsrc(src_n, (T)0);
    for (int n = 0; n < N; ++n)
        for (int y = 0; y < sh[n]; ++y)
            for (int x = 0; x < sw[n]; ++x)
                for (int c = 0; c < C; ++c) {
                    long idx = ((n * maxSH + y) * maxSW + x) * C + c;
                    hsrc[idx] = (T)((x + y * 3 + n * 7 + c * 5) & 0xff);
                }

    // Device batches.
    T* d_src = sycl::malloc_device<T>(src_n, q);
    T* d_dst = sycl::malloc_device<T>(dst_n, q);
    int* d_sw = sycl::malloc_device<int>(N, q);
    int* d_sh = sycl::malloc_device<int>(N, q);
    int* d_dw = sycl::malloc_device<int>(N, q);
    int* d_dh = sycl::malloc_device<int>(N, q);
    q.copy<T>(hsrc.data(), d_src, src_n).wait();
    q.copy<int>(sw.data(), d_sw, N);
    q.copy<int>(sh.data(), d_sh, N);
    q.copy<int>(dw.data(), d_dw, N);
    q.copy<int>(dh.data(), d_dh, N).wait();

    cs::VarShapeBatch<T, C> sv{d_src, d_sw, d_sh, N, maxSW, maxSH};
    cs::VarShapeBatch<T, C> dv{d_dst, d_dw, d_dh, N, maxDW, maxDH};
    cs::run_resize_varshape<T, C>(q, sv, dv, interp).wait();

    std::vector<T> hdst(dst_n);
    q.copy<T>(d_dst, hdst.data(), dst_n).wait();

    sycl::free(d_src, q); sycl::free(d_dst, q);
    sycl::free(d_sw, q); sycl::free(d_sh, q); sycl::free(d_dw, q); sycl::free(d_dh, q);

    // Compare to CPU ref.
    int mism = 0; float maxerr = 0.f; T ref[C];
    for (int n = 0; n < N; ++n)
        for (int y = 0; y < dh[n]; ++y)
            for (int x = 0; x < dw[n]; ++x) {
                cpu_ref<T, C>(hsrc, N, maxSW, maxSH, sw.data(), sh.data(), n, x, y,
                              dw[n], dh[n], interp, ref);
                for (int c = 0; c < C; ++c) {
                    T got = hdst[((n * maxDH + y) * maxDW + x) * C + c];
                    float err = std::fabs((float)got - (float)ref[c]);
                    maxerr = std::max(maxerr, err);
                    if (err > tol) {
                        if (mism < 8)
                            std::printf("  MISMATCH [%s] n=%d y=%d x=%d c=%d: gpu=%d ref=%d err=%.2e\n",
                                        name.c_str(), n, y, x, c, (int)got, (int)ref[c], err);
                        ++mism;
                    }
                }
            }

    const char* ip = (interp == cs::Interp::Nearest) ? "NEAR"
                    : (interp == cs::Interp::Linear) ? "LIN " : "CUB ";
    const char* dt = (sizeof(T) == 1) ? "U8" : "F32";
    if (mism == 0) {
        std::printf("[PASS] %-22s %s C%d N=%d  maxerr=%.2e\n", name.c_str(), dt, C, N, maxerr);
        return 0;
    }
    std::printf("[FAIL] %-22s %s C%d  %d mismatches, maxerr=%.2e\n", name.c_str(), dt, C, mism, maxerr);
    return 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;

    // F32 C1, N=3 images of varying shapes (up/down/non-integer)
    {
        std::vector<int> sw{4, 6, 5}, sh{4, 5, 5}, dw{8, 3, 7}, dh{8, 3, 4};
        fails += run_case<float,1>(q, "f32c1_near", 3, sw, sh, dw, dh, cs::Interp::Nearest, 0.f);
        fails += run_case<float,1>(q, "f32c1_lin",  3, sw, sh, dw, dh, cs::Interp::Linear,  1e-4f);
        fails += run_case<float,1>(q, "f32c1_cub",  3, sw, sh, dw, dh, cs::Interp::Cubic,   1e-3f);
    }
    // U8 C1
    {
        std::vector<int> sw{4, 6, 5}, sh{4, 5, 5}, dw{8, 3, 7}, dh{8, 3, 4};
        fails += run_case<std::uint8_t,1>(q, "u8c1_near", 3, sw, sh, dw, dh, cs::Interp::Nearest, 0.f);
        fails += run_case<std::uint8_t,1>(q, "u8c1_lin",  3, sw, sh, dw, dh, cs::Interp::Linear,  1.f);
        fails += run_case<std::uint8_t,1>(q, "u8c1_cub",  3, sw, sh, dw, dh, cs::Interp::Cubic,   1.f);
    }
    // U8 C3, N=2
    {
        std::vector<int> sw{4, 5}, sh{4, 6}, dw{8, 3}, dh{8, 3};
        fails += run_case<std::uint8_t,3>(q, "u8c3_near", 2, sw, sh, dw, dh, cs::Interp::Nearest, 0.f);
        fails += run_case<std::uint8_t,3>(q, "u8c3_lin",  2, sw, sh, dw, dh, cs::Interp::Linear,  1.f);
        fails += run_case<std::uint8_t,3>(q, "u8c3_cub",  2, sw, sh, dw, dh, cs::Interp::Cubic,   1.f);
    }
    // F32 C3, N=2 (cubic needs src>=4; 5x6 ok)
    {
        std::vector<int> sw{5, 6}, sh{5, 6}, dw{3, 9}, dh{3, 4};
        fails += run_case<float,3>(q, "f32c3_lin", 2, sw, sh, dw, dh, cs::Interp::Linear, 1e-3f);
        fails += run_case<float,3>(q, "f32c3_cub", 2, sw, sh, dw, dh, cs::Interp::Cubic,  1e-2f);
    }

    if (fails == 0) {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    }
    std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return 1;
}
