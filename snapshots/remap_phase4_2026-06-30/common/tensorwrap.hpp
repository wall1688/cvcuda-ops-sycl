// Minimal NHWC tensor wrapper for the SYCL resize port.
// Ported concept from CV-CUDA's cuda_tools/TensorWrap.hpp (CreateTensorWrapNHW),
// stripped to the bare minimum the resize kernels need.
//
// Phase 2: templated on element type T (float / uint8_t) and channel count C
// (1/3/4). NHWC packed layout (c varies fastest, then w, then h, then n).
// A TensorView is a trivially-copyable view over a USM device pointer + shape;
// passed BY VALUE into kernels (like CV-CUDA's SrcWrapper/DstWrapper).
//
// Pixels are read/written as T[C] arrays (no sycl::vec quirks); the kernels
// interpolate per-channel in float and SaturateCast back to T.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_TENSORWRAP_HPP
#define CVSACL_TENSORWRAP_HPP

#include <cstdint>
#include <cstddef>

namespace cvsycl {

template <typename T, int C>
struct TensorView {
    static_assert(C >= 1 && C <= 4, "phase 2 supports C in {1,3,4}");

    using value_type = T;
    static constexpr int channels = C;

    T* data;   // USM device pointer (NHWC packed)
    int N;     // batch
    int H;     // height
    int W;     // width
    // C is a compile-time template parameter.

    long colStride()    const { return C; }
    long rowStride()    const { return (long)W * C; }
    long sampleStride() const { return (long)H * W * C; }

    // Read pixel at (x,y,z) into out[C].
    void read(int x, int y, int z, T* out) const {
        const T* p = data + (long)z * sampleStride() + (long)y * rowStride() + (long)x * C;
        #pragma unroll
        for (int c = 0; c < C; ++c) out[c] = p[c];
    }
    // Write pixel at (x,y,z) from v[C].
    void write(int x, int y, int z, const T* v) const {
        T* p = data + (long)z * sampleStride() + (long)y * rowStride() + (long)x * C;
        #pragma unroll
        for (int c = 0; c < C; ++c) p[c] = v[c];
    }
};

}  // namespace cvsycl

#endif  // CVSACL_TENSORWRAP_HPP
