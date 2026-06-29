// Minimal NHWC tensor wrapper for the SYCL resize port.
// Ported concept from CV-CUDA's cuda_tools/TensorWrap.hpp (CreateTensorWrapNHW),
// stripped to the bare minimum needed by the resize kernels.
//
// Phase 1 scope: F32 elements, single-channel (C=1), NHWC layout, packed
// (no row padding). Multi-channel (C=3/4) and other dtypes are a later phase —
// the wrapper is intentionally tiny so it can be templated/extended then.
//
// A TensorView is a trivially-copyable view over a USM device pointer + shape;
// it is passed BY VALUE into kernels (like CV-CUDA's SrcWrapper/DstWrapper).
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_TENSORWRAP_HPP
#define CVSACL_TENSORWRAP_HPP

#include <cstdint>
#include <cstddef>

namespace cvsycl {

struct TensorView {
    float* data;     // USM device pointer (NHWC packed, F32, C=1 in phase 1)
    int N;           // batch
    int H;           // height
    int W;           // width
    int C;           // channels (==1 in phase 1; kept for forward-compat)

    // Strides in elements (NHWC packed: c varies fastest, then w, then h, then n).
    long colStride()    const { return C; }
    long rowStride()    const { return (long)W * C; }
    long sampleStride() const { return (long)H * W * C; }

    // --- C==1 fast path (phase 1) ---
    // Read one pixel (one float) at (x, y, z).
    float read1(int x, int y, int z) const {
        return data[(long)z * sampleStride() + (long)y * rowStride() + (long)x];
    }
    // Write one pixel (one float) at (x, y, z).
    void write1(int x, int y, int z, float v) const {
        data[(long)z * sampleStride() + (long)y * rowStride() + (long)x] = v;
    }
};

}  // namespace cvsycl

#endif  // CVSACL_TENSORWRAP_HPP
