// Small math helpers for the SYCL resize port.
// Ported concept from CV-CUDA's cuda_tools MathOps.hpp / MathWrappers.hpp /
// DropCast.hpp / StaticCast.hpp — only the handful the resize kernels use.
// SYCL already provides sycl::min/max/clamp/floor; these are thin aliases so
// the kernel code reads close to the CUDA original (cuda::min, cuda::max ...).
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_MATH_OPS_HPP
#define CVSACL_MATH_OPS_HPP

#include <sycl/sycl.hpp>

namespace cvsycl {

template <typename T>
inline T min(T a, T b) { return a < b ? a : b; }

template <typename T>
inline T max(T a, T b) { return a > b ? a : b; }

template <typename T>
inline T clamp(T v, T lo, T hi) { return min(max(v, lo), hi); }

}  // namespace cvsycl

#endif  // CVSACL_MATH_OPS_HPP
