// SaturateCast for the SYCL resize port.
// Ported concept from CV-CUDA's cuda_tools/SaturateCast.hpp.
//
// Interpolation (LINEAR/CUBIC) produces float results that may fall outside the
// target pixel type's range (e.g. 258.7 for U8). A naive (uchar) cast wraps
// around; SaturateCast clamps to the type's representable range first, then
// rounds. For F32 the range is unbounded, so SaturateCast<float> is just a
// plain conversion — which is why phase 1 (F32) can use a trivial identity.
//
// Phase 1: F32 identity only. U8/U16/S16 are added when those dtypes land.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_SATURATE_CAST_HPP
#define CVSACL_SATURATE_CAST_HPP

#include <cmath>

namespace cvsycl {

// F32 -> F32: no clamping needed.
template <typename DstT>
inline DstT SaturateCast(float v) {
    // Default phase-1 specialization assumes DstT == float.
    return static_cast<DstT>(v);
}

// TODO(phase 2): specializations for unsigned char / short / etc.
// template<> inline unsigned char SaturateCast<unsigned char>(float v) {
//     return (unsigned char)std::min(255.f, std::max(0.f, std::roundf(v)));
// }

}  // namespace cvsycl

#endif  // CVSACL_SATURATE_CAST_HPP
