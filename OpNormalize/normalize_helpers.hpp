// CV-CUDA Normalize port — shared helpers (TypeTraits min/max, NHWC indexing).
// Ported from cvcuda/cuda_tools/{SaturateCast,TypeTraits,TensorWrap}.hpp.
// CPU-safe (no SYCL symbols) so the gold header can include it too.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __NORMALIZE_HELPERS_HPP__
#define __NORMALIZE_HELPERS_HPP__

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cvcuda {
namespace normalize {

// TypeTraits::max equivalent (per-dtype clamp upper bound for SaturateCast).
// Float is unused by the saturate path (float output = identity), so any value.
template <typename T>
constexpr T TypeMax() {
  if constexpr (std::is_same_v<T, uint8_t>)
    return 255;
  else if constexpr (std::is_same_v<T, int8_t>)
    return 127;
  else if constexpr (std::is_same_v<T, uint16_t>)
    return 65535;
  else if constexpr (std::is_same_v<T, int16_t>)
    return 32767;
  else if constexpr (std::is_same_v<T, int32_t>)
    return std::numeric_limits<int32_t>::max();
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

}  // namespace normalize
}  // namespace cvcuda

#endif  // __NORMALIZE_HELPERS_HPP__
