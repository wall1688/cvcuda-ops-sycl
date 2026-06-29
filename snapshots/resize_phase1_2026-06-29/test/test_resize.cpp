// Synthetic correctness test for the SYCL resize port (phase 1: F32, C=1).
//
// Strategy (same as the lidar voxelization port's correctness test):
// build tiny synthetic images, run the GPU resize, and compare against a CPU
// reference that implements the EXACT same coordinate-mapping + interpolation
// math as resize.cpp. If they match within tolerance, the kernel is correct.
//
// Run on the Intel host:
//   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test_resize
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#include "../resize.hpp"
#include "../common/tensorwrap.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

// ---------------- CPU reference (mirrors resize.cpp numerics) ----------------
static float cpu_nearest(const std::vector<float>& src, int sw, int sh,
                         int x, int y, int z, int c, float sx, float sy) {
    int isx = (int)std::floor((x + 0.5f) * sx);
    int isy = (int)std::floor((y + 0.5f) * sy);
    isx = std::min(isx, sw - 1);
    isy = std::min(isy, sh - 1);
    // NHWC, C==1
    return src[((z * sh + isy) * sw + isx) * c];
}

static float cpu_linear(const std::vector<float>& src, int sw, int sh,
                        int x, int y, int z, int c, float sx, float sy) {
    float fx = (x + 0.5f) * sx - 0.5f;
    float fy = (y + 0.5f) * sy - 0.5f;
    int isx = (int)std::floor(fx);
    int isy = (int)std::floor(fy);
    float wx = (isx < 0) ? 0.f : (isx > sw - 2) ? 1.f : (fx - isx);
    float wy = (isy < 0) ? 0.f : (isy > sh - 2) ? 1.f : (fy - isy);
    isx = std::max(0, std::min(isx, sw - 2));
    isy = std::max(0, std::min(isy, sh - 2));
    auto at = [&](int xx, int yy) { return src[((z * sh + yy) * sw + xx) * c]; };
    float v = at(isx, isy)     * ((1.f - wx) * (1.f - wy))
            + at(isx + 1, isy) * (wx * (1.f - wy))
            + at(isx, isy + 1) * ((1.f - wx) * wy)
            + at(isx + 1, isy + 1) * (wx * wy);
    return v;
}

// ---------------- GPU runner ----------------
static std::vector<float> run_gpu(sycl::queue& q, const std::vector<float>& src,
                                  int n, int sh, int sw, int c,
                                  int dh, int dw, cvsycl::Interp interp) {
    size_t src_bytes = (size_t)n * sh * sw * c * sizeof(float);
    size_t dst_bytes = (size_t)n * dh * dw * c * sizeof(float);

    float* d_src = sycl::malloc_device<float>(src_bytes / sizeof(float), q);
    float* d_dst = sycl::malloc_device<float>(dst_bytes / sizeof(float), q);
    if (!d_src || !d_dst) throw std::runtime_error("USM alloc failed");

    q.copy<float>(src.data(), d_src, src_bytes / sizeof(float)).wait();

    cvsycl::TensorView sv{d_src, n, sh, sw, c};
    cvsycl::TensorView dv{d_dst, n, dh, dw, c};
    cvsycl::run_resize(q, sv, dv, interp).wait();

    std::vector<float> out(dst_bytes / sizeof(float));
    q.copy<float>(d_dst, out.data(), dst_bytes / sizeof(float)).wait();

    sycl::free(d_src, q);
    sycl::free(d_dst, q);
    return out;
}

// ---------------- one test case ----------------
struct Case {
    std::string name;
    int n, sh, sw, dh, dw;
    std::vector<float> src;  // n*sh*sw values (C==1)
    cvsycl::Interp interp;
    float tol;
};

static int run_case(sycl::queue& q, const Case& cs) {
    const int c = 1;
    float sx = (float)cs.sw / cs.dw;
    float sy = (float)cs.sh / cs.dh;

    std::vector<float> gpu = run_gpu(q, cs.src, cs.n, cs.sh, cs.sw, c, cs.dh, cs.dw, cs.interp);

    int mism = 0;
    float maxerr = 0.f;
    for (int z = 0; z < cs.n; ++z)
    for (int y = 0; y < cs.dh; ++y)
    for (int x = 0; x < cs.dw; ++x) {
        float ref = (cs.interp == cvsycl::Interp::Nearest)
                        ? cpu_nearest(cs.src, cs.sw, cs.sh, x, y, z, c, sx, sy)
                        : cpu_linear (cs.src, cs.sw, cs.sh, x, y, z, c, sx, sy);
        float got = gpu[((z * cs.dh + y) * cs.dw + x) * c];
        float err = std::fabs(got - ref);
        maxerr = std::max(maxerr, err);
        if (err > cs.tol) {
            if (mism < 8) {
                std::printf("  MISMATCH [%s] z=%d y=%d x=%d: gpu=%.6f ref=%.6f err=%.2e\n",
                            cs.name.c_str(), z, y, x, got, ref, err);
            }
            ++mism;
        }
    }

    const char* ip = (cs.interp == cvsycl::Interp::Nearest) ? "NEAREST" : "LINEAR";
    if (mism == 0) {
        std::printf("[PASS] %-28s %s  %dx%d->%dx%d (N=%d)  maxerr=%.2e\n",
                    cs.name.c_str(), ip, cs.sw, cs.sh, cs.dw, cs.dh, cs.n, maxerr);
        return 0;
    }
    std::printf("[FAIL] %-28s %s  %d mismatches, maxerr=%.2e\n",
                cs.name.c_str(), ip, mism, maxerr);
    return 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;

    // Case 1: 2x2 -> 4x4 NEAREST (integer 2x upscale)
    {
        Case cs;
        cs.name = "upscale2x_nearest";
        cs.n = 1; cs.sh = 2; cs.sw = 2; cs.dh = 4; cs.dw = 4;
        cs.src = {1.f, 3.f, 7.f, 9.f};
        cs.interp = cvsycl::Interp::Nearest;
        cs.tol = 0.f;
        fails += run_case(q, cs);
    }
    // Case 2: 2x2 -> 4x4 LINEAR
    {
        Case cs;
        cs.name = "upscale2x_linear";
        cs.n = 1; cs.sh = 2; cs.sw = 2; cs.dh = 4; cs.dw = 4;
        cs.src = {1.f, 3.f, 7.f, 9.f};
        cs.interp = cvsycl::Interp::Linear;
        cs.tol = 1e-4f;
        fails += run_case(q, cs);
    }
    // Case 3: 3x3 -> 5x5 LINEAR (non-integer scale)
    {
        Case cs;
        cs.name = "noninteger_linear";
        cs.n = 1; cs.sh = 3; cs.sw = 3; cs.dh = 5; cs.dw = 5;
        cs.src = {0,10,20, 30,40,50, 60,70,80};
        cs.interp = cvsycl::Interp::Linear;
        cs.tol = 1e-4f;
        fails += run_case(q, cs);
    }
    // Case 4: 4x4 -> 2x2 NEAREST + LINEAR (downscale)
    {
        Case cs;
        cs.name = "downscale_nearest";
        cs.n = 1; cs.sh = 4; cs.sw = 4; cs.dh = 2; cs.dw = 2;
        cs.src = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};
        cs.interp = cvsycl::Interp::Nearest;
        cs.tol = 0.f;
        fails += run_case(q, cs);
    }
    {
        Case cs;
        cs.name = "downscale_linear";
        cs.n = 1; cs.sh = 4; cs.sw = 4; cs.dh = 2; cs.dw = 2;
        cs.src = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};
        cs.interp = cvsycl::Interp::Linear;
        cs.tol = 1e-4f;
        fails += run_case(q, cs);
    }
    // Case 5: batch N=2, 2x2 -> 4x4 LINEAR (different data per sample)
    {
        Case cs;
        cs.name = "batch2_linear";
        cs.n = 2; cs.sh = 2; cs.sw = 2; cs.dh = 4; cs.dw = 4;
        // sample 0: {1,3,7,9}, sample 1: {0,100,200,255}
        cs.src = {1.f,3.f,7.f,9.f,  0.f,100.f,200.f,255.f};
        cs.interp = cvsycl::Interp::Linear;
        cs.tol = 1e-3f;
        fails += run_case(q, cs);
    }

    if (fails == 0) {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    }
    std::printf("\n=== %d TEST(S) FAILED ===\n", fails);
    return 1;
}
