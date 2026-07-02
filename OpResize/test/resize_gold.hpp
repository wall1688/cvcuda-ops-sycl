// CPU gold reference for Resize. Extracted verbatim from the per-pixel
// cpu_nearest/linear/cubic/area helpers that lived in test_resize.cpp, plus a
// whole-image gold_resize() wrapper so the profile benchmark can time the CPU
// reference without duplicating the interpolation math. Mirrors resize.hpp
// numerics exactly (same coordinate mapping + SaturateCast + get_cubic_coeffs),
// so GPU-vs-gold is a self-consistency check, NOT a bit-level diff vs NVIDIA
// CV-CUDA. See README "验证局限".
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef __RESIZE_GOLD_HPP__
#define __RESIZE_GOLD_HPP__

#include "../resize.hpp"
#include "../common/tensorwrap.hpp"
#include "../common/saturate_cast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace resize_gold {

template <typename T, int C>
void cpu_nearest(const std::vector<T>& src, int sw, int sh,
                 int x, int y, int z, float sx, float sy, T* out) {
    int isx = (int)std::floor((x + 0.5f) * sx);
    int isy = (int)std::floor((y + 0.5f) * sy);
    isx = std::min(isx, sw - 1);
    isy = std::min(isy, sh - 1);
    const T* p = src.data() + ((z * sh + isy) * sw + isx) * C;
    for (int c = 0; c < C; ++c) out[c] = p[c];
}

template <typename T, int C>
void cpu_linear(const std::vector<T>& src, int sw, int sh,
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
        out[c] = cvsycl::SaturateCast<T>(v);
    }
}

template <typename T, int C>
void cpu_cubic(const std::vector<T>& src, int sw, int sh,
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
    cvsycl::get_cubic_coeffs(fx, wx);
    cvsycl::get_cubic_coeffs(coordY - isy, wy);
    auto at = [&](int xx, int yy, int c) -> float {
        return (float)src[((z * sh + yy) * sw + xx) * C + c];
    };
    for (int c = 0; c < C; ++c) {
        float sum = 0.f;
        for (int cy = -1; cy <= 2; ++cy)
            for (int cx = -1; cx <= 2; ++cx)
                sum += at(isx + cx, isy + cy, c) * (wx[cx + 1] * wy[cy + 1]);
        out[c] = cvsycl::SaturateCast<T>(std::fabs(sum));
    }
}

template <typename T, int C>
void cpu_area(const std::vector<T>& src, int sw, int sh,
              int x, int y, int z, float sx, float sy, T* out) {
    float fsx1 = x * sx;
    float fsy1 = y * sy;
    float fsx2 = fsx1 + sx;
    float fsy2 = fsy1 + sy;
    int xmin = (int)std::ceil(fsx1);
    int xmax = (int)std::floor(fsx2);
    int ymin = (int)std::ceil(fsy1);
    int ymax = (int)std::floor(fsy2);

    float acc[C];
    for (int c = 0; c < C; ++c) acc[c] = 0.f;

    auto bord = [&](int xx, int yy, int c) -> float {
        if (xx < 0 || xx >= sw || yy < 0 || yy >= sh) return 0.f;
        return (float)src[((z * sh + yy) * sw + xx) * C + c];
    };
    auto add = [&](int xx, int yy, float w) {
        for (int c = 0; c < C; ++c) acc[c] += bord(xx, yy, c) * w;
    };

    bool intArea = ((float)(int)std::ceil(sx) == sx) && ((float)(int)std::ceil(sy) == sy);
    if (intArea) {
        float scale = 1.f / (sx * sy);
        for (int cy = ymin; cy < ymax; ++cy)
            for (int cx = xmin; cx < xmax; ++cx)
                add(cx, cy, scale);
    } else {
        float scale = 1.f / (std::min(sx, (float)sw - fsx1) * std::min(sy, (float)sh - fsy1));
        for (int cy = ymin; cy < ymax; ++cy) {
            for (int cx = xmin; cx < xmax; ++cx) add(cx, cy, scale);
            if (xmin > fsx1) add(xmin - 1, cy, (xmin - fsx1) * scale);
            if (xmax < fsx2) add(xmax,     cy, (fsx2 - xmax) * scale);
        }
        if (ymin > fsy1) {
            for (int cx = xmin; cx < xmax; ++cx) add(cx, ymin - 1, (ymin - fsy1) * scale);
            if (xmin > fsx1) add(xmin - 1, ymin - 1, (ymin - fsy1) * (xmin - fsx1) * scale);
            if (xmax < fsx2) add(xmax,     ymin - 1, (ymin - fsy1) * (fsx2 - xmax) * scale);
        }
        if (ymax < fsy2) {
            for (int cx = xmin; cx < xmax; ++cx) add(cx, ymax, (fsy2 - ymax) * scale);
            if (xmax < fsx2) add(xmax,     ymax, (fsy2 - ymax) * (fsx2 - xmax) * scale);
            if (xmin > fsx1) add(xmin - 1, ymax, (fsy2 - ymax) * (xmin - fsx1) * scale);
        }
    }
    for (int c = 0; c < C; ++c) out[c] = cvsycl::SaturateCast<T>(acc[c]);
}

// Whole-image CPU reference. dst is resized to n*dh*dw*C. interp selects the
// per-pixel helper. Single-threaded — used by the profile benchmark for the
// GPU-vs-CPU speedup number.
template <typename T, int C>
void gold_resize(std::vector<T>& dst, const std::vector<T>& src,
                 int n, int sh, int sw, int dh, int dw, cvsycl::Interp interp) {
    dst.assign((size_t)n * dh * dw * C, T{});
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;
    T ref[C];
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                if (interp == cvsycl::Interp::Nearest)
                    cpu_nearest<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
                else if (interp == cvsycl::Interp::Linear)
                    cpu_linear<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
                else if (interp == cvsycl::Interp::Cubic)
                    cpu_cubic<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
                else
                    cpu_area<T, C>(src, sw, sh, x, y, z, sx, sy, ref);
                for (int c = 0; c < C; ++c)
                    dst[((z * dh + y) * dw + x) * C + c] = ref[c];
            }
}

}  // namespace resize_gold

#endif  // __RESIZE_GOLD_HPP__
