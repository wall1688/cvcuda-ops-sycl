// CV-CUDA CvtColor port — shared helpers (SaturateCast, Alpha, color constants,
// NHWC indexing). Ported from cvcuda/cuda_tools/{SaturateCast,TypeTraits}.hpp
// and cvt_color.cu constants. Helpers are added per stage as needed.
// SPDX-License-Identifier: Apache-2.0 (NVIDIA original)

#ifndef __CVT_HELPERS_HPP__
#define __CVT_HELPERS_HPP__

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cvcuda {
namespace cvt {

// ---- TypeTraits::max equivalent (for Alpha fill) ----
template <typename T>
constexpr T TypeMax() {
  if constexpr (std::is_floating_point_v<T>)
    return T(1);
  else if constexpr (std::is_same_v<T, uint8_t>)
    return 255;
  else if constexpr (std::is_same_v<T, int8_t>)
    return 127;
  else if constexpr (std::is_same_v<T, uint16_t>)
    return 65535;
  else if constexpr (std::is_same_v<T, int16_t>)
    return 32767;
  else if constexpr (std::is_same_v<T, int32_t>)
    return std::numeric_limits<int32_t>::max();
  else
    return std::numeric_limits<T>::max();
}

// TypeTraits::min equivalent (for SaturateCast clamp lower bound).
template <typename T>
constexpr T TypeMin() {
  if constexpr (std::is_signed_v<T>)
    return std::numeric_limits<T>::lowest();
  else
    return T(0);
}

// Alpha fill value: 1 for float types, max for integral (matches CV-CUDA Alpha<T>).
template <typename T>
constexpr T AlphaVal() {
  return TypeMax<T>();
}

// ---- NHWC contiguous indexing ----
// pixel (n,y,x) base offset (in elements), channel c added by caller.
template <typename T>
inline T* nhwc_ptr(T* base, int n, int y, int x, int H, int W, int C) {
  return base + ((static_cast<size_t>(n) * H + y) * W + x) * C;
}
template <typename T>
inline const T* nhwc_ptr(const T* base, int n, int y, int x, int H, int W, int C) {
  return base + ((static_cast<size_t>(n) * H + y) * W + x) * C;
}

// ---- Color conversion constants (ported verbatim from cvt_color.cu:36-80) ----
// Float RGB->luma (Rec.601).
constexpr float R2YF = 0.299f;
constexpr float G2YF = 0.587f;
constexpr float B2YF = 0.114f;

constexpr int gray_shift = 15;
constexpr int yuv_shift  = 14;
constexpr int RY15       = 9798;   // R2YF*32768 + 0.5
constexpr int GY15       = 19235;
constexpr int BY15       = 3735;

constexpr int R2Y  = 4899;   // R2YF*16384
constexpr int G2Y  = 9617;
constexpr int B2Y  = 1868;
constexpr int R2VI = 14369;  // R2VF*16384
constexpr int B2UI = 8061;   // B2UF*16384

constexpr float B2UF = 0.492f;
constexpr float R2VF = 0.877f;

constexpr int U2BI = 33292;
constexpr int U2GI = -6472;
constexpr int V2GI = -9519;
constexpr int V2RI = 18678;

constexpr float U2BF = 2.032f;
constexpr float U2GF = -0.395f;
constexpr float V2GF = -0.581f;
constexpr float V2RF = 1.140f;

// ITU-R BT.601 YUV420<->RGB fixed-point coeffs.
constexpr int ITUR_BT_601_CY    = 1220542;
constexpr int ITUR_BT_601_CUB   = 2116026;
constexpr int ITUR_BT_601_CUG   = -409993;
constexpr int ITUR_BT_601_CVG   = -852492;
constexpr int ITUR_BT_601_CVR   = 1673527;
constexpr int ITUR_BT_601_SHIFT = 20;
constexpr int ITUR_BT_601_CRY = 269484;
constexpr int ITUR_BT_601_CGY = 528482;
constexpr int ITUR_BT_601_CBY = 102760;
constexpr int ITUR_BT_601_CRU = -155188;
constexpr int ITUR_BT_601_CGU = -305135;
constexpr int ITUR_BT_601_CBU = 460324;
constexpr int ITUR_BT_601_CGV = -385875;
constexpr int ITUR_BT_601_CBV = -74448;

// Rounding right-shift: CV_DESCALE(x,n) = ((x) + (1<<(n-1))) >> n
template <typename T>
inline T cv_descale(T x, int n) {
  return (x + (T(1) << (n - 1))) >> n;
}

}  // namespace cvt
}  // namespace cvcuda

#endif  // __CVT_HELPERS_HPP__
