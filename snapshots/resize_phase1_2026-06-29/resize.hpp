// SYCL port of CV-CUDA's Resize operator (priv/OpResize.cu).
// Phase 1: F32, single-channel (C=1), NHWC layout. NEAREST + LINEAR only.
//
// Public host interface: run_resize(q, src, dst, interp) launches the kernel.
// Memory is USM device pointers wrapped in TensorView; the caller owns H2D/D2H
// copies (same pattern as the lidar voxelization port).
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_RESIZE_HPP
#define CVSACL_RESIZE_HPP

#include <sycl/sycl.hpp>
#include "common/tensorwrap.hpp"

namespace cvsycl {

enum class Interp {
    Nearest,
    Linear,
    // Cubic, Area — later phases
};

// Launch the resize kernel on `q`. src and dst are USM device TensorViews
// (NHWC, F32, C=1). src and dst may differ in H and W; N and C must match.
// Returns the SYCL event for the submitted kernel (for profiling).
sycl::event run_resize(sycl::queue& q, const TensorView& src, const TensorView& dst,
                       Interp interp);

}  // namespace cvsycl

#endif  // CVSACL_RESIZE_HPP
