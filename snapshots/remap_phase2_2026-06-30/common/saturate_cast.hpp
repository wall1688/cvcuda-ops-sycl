// SaturateCast for the SYCL resize port.
// Ported concept from CV-CUDA's cuda_tools/SaturateCast.hpp.
//
// Interpolation (LINEAR/CUBIC) produces float results that may fall outside the
// target pixel type's range (e.g. 258.7 for U8). A naive (uchar) cast wraps
// around; SaturateCast rounds (half-to-even, matching CUDA __float2int_rn) and
// clamps to the type's representable range before converting.
//
// Implemented with sycl:: math functions so the SAME code runs on host (CPU
// reference in the test) and device (kernel) — guaranteeing the two agree.
//
// Phase 2: float (identity) + uint8_t (round + clamp [0,255]).
// U16/S16 added when those dtypes land.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_SATURATE_CAST_HPP
#define CVSACL_SATURATE_CAST_HPP

#include <sycl/sycl.hpp>
#include <cstdint>

namespace cvsycl {

// Primary template (default: plain static_cast). Must precede specializations.
template <typename DstT>
inline DstT SaturateCast(float v) {
    return static_cast<DstT>(v);
}

// F32 -> F32: no clamping needed.
template <>
inline float SaturateCast<float>(float v) {
    return v;
}

// F32 -> U8: round half-to-even, clamp to [0,255]. NaN -> 0.
template <>
inline std::uint8_t SaturateCast<std::uint8_t>(float v) {
    float r = sycl::round(v);          // round half to even (matches __float2int_rn)
    if (!(r > 0.f)) r = 0.f;           // NaN or negative -> 0
    else if (r > 255.f) r = 255.f;
    return static_cast<std::uint8_t>(r);
}

}  // namespace cvsycl

#endif  // CVSACL_SATURATE_CAST_HPP
