// CV-CUDA Rotate port — shared helpers (TypeTraits min/max, NHWC indexing,
// border clamp, cubic coefficients). Ported from cvcuda/cuda_tools/
// {SaturateCast,TypeTraits,TensorWrap,InterpolationWrap}.hpp.
// CPU-safe (no SYCL symbols) so the gold header can include it too.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __ROTATE_HELPERS_HPP__
#define __ROTATE_HELPERS_HPP__

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cvcuda {
namespace rotate {

// TypeTraits::max equivalent (per-dtype clamp upper bound for SaturateCast).
template <typename T>
constexpr T TypeMax() {
  if constexpr (std::is_same_v<T, uint8_t>)
    return 255;
  else if constexpr (std::is_same_v<T, int16_t>)
    return 32767;
  else if constexpr (std::is_same_v<T, uint16_t>)
    return 65535;
  else if constexpr (std::is_floating_point_v<T>)
    return T(1);
  else
    return std::numeric_limits<T>::max();
}

// TypeTraits::min equivalent (per-dtype clamp lower bound). Signed -> lowest,
// unsigned -> 0.
template <typename T>
constexpr T TypeMin() {
  if constexpr (std::is_signed_v<T>)
    return std::numeric_limits<T>::lowest();
  else
    return T(0);
}

// NHWC contiguous pixel (n,y,x) base offset (in elements); caller adds channel.
template <typename T>
inline T* nhwc_ptr(T* base, int n, int y, int x, int H, int W, int C) {
  return base + ((static_cast<size_t>(n) * H + y) * W + x) * C;
}
template <typename T>
inline const T* nhwc_ptr(const T* base, int n, int y, int x, int H, int W, int C) {
  return base + ((static_cast<size_t>(n) * H + y) * W + x) * C;
}

// BORDER_REPLICATE index: clamp to [0, dim-1]. Pure arithmetic (CPU+device).
inline int border_replicate(int i, int dim) {
  if (i < 0) return 0;
  if (i >= dim) return dim - 1;
  return i;
}

// GetCubicCoeffs (InterpolationWrap.hpp:83-101), verbatim. Pure float
// arithmetic (no transcendentals) -> bit-identical on CPU and GPU. delta must
// be in [0,1) (the fractional part x - floor(x)).
inline void cubic_coeffs(float delta, float& w0, float& w1, float& w2, float& w3) {
  w0 = -.5f;
  w0 = w0 * delta + 1.f;
  w0 = w0 * delta - .5f;
  w0 = w0 * delta;

  w1 = 1.5f;
  w1 = w1 * delta - 2.5f;
  w1 = w1 * delta;
  w1 = w1 * delta + 1.f;

  w2 = -1.5f;
  w2 = w2 * delta + 2.f;
  w2 = w2 * delta + .5f;
  w2 = w2 * delta;

  w3 = 1 - w0 - w1 - w2;
}

}  // namespace rotate
}  // namespace cvcuda

#endif  // __ROTATE_HELPERS_HPP__
