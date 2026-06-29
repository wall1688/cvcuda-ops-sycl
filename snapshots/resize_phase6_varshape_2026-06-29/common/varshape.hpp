// Var-shape (variable per-image dimensions) batch wrapper for the SYCL resize
// port. Ported concept from CV-CUDA's cuda_tools/ImageBatchVarShapeWrap, used
// by legacy/resize_var_shape.cu.
//
// A batch of N images, each with its own width/height (imgW[n], imgH[n]) but
// sharing a channel count C and a uniform row pitch. Storage is pitched:
//   pixel (x, y, n) at  data + n*imageStride + y*rowPitch + x*C
// with rowPitch = maxW*C and imageStride = maxH*maxW*C. Images smaller than
// maxW/maxW leave padding at row/image ends (wasted, but uniform-pitch simple).
//
// imgW/imgH are USM device int arrays of length N (per-image dims, readable
// inside kernels). src and dst batches have INDEPENDENT shapes (different
// maxW/maxH and per-image dims); the kernel indexes both by batch index n.
//
// Phase 6 scope: F32/U8, C=1/3/4, NHWC-style pitched. Same element types as
// the fixed-shape port.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_VARSHAPE_HPP
#define CVSACL_VARSHAPE_HPP

#include <cstdint>
#include <cstddef>

namespace cvsycl {

template <typename T, int C>
struct VarShapeBatch {
    static_assert(C >= 1 && C <= 4, "phase 6 supports C in {1,3,4}");
    using value_type = T;
    static constexpr int channels = C;

    T*   data;    // USM device pointer (pitched batch)
    int* imgW;    // USM device array [N]: actual width  of each image
    int* imgH;    // USM device array [N]: actual height of each image
    int  N;       // number of images
    int  maxW;    // row pitch width  (rowPitch = maxW*C elements)
    int  maxH;    // image stride height (imageStride = maxH*maxW*C)

    long rowPitch()    const { return (long)maxW * C; }
    long imageStride() const { return (long)maxH * maxW * C; }

    void read(int x, int y, int n, T* out) const {
        const T* p = data + (long)n * imageStride() + (long)y * rowPitch() + (long)x * C;
        #pragma unroll
        for (int c = 0; c < C; ++c) out[c] = p[c];
    }
    void write(int x, int y, int n, const T* v) const {
        T* p = data + (long)n * imageStride() + (long)y * rowPitch() + (long)x * C;
        #pragma unroll
        for (int c = 0; c < C; ++c) p[c] = v[c];
    }
};

}  // namespace cvsycl

#endif  // CVSACL_VARSHAPE_HPP
