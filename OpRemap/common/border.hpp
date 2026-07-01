// Border handling for the SYCL remap port.
// Ported concept from CV-CUDA's cuda_tools/BorderWrap.hpp — the border index
// remap math (GetIndexWithBorder) and outside test (IsOutside), stripped of the
// nvcv tensor framework.
//
// Remap samples src and map at ARBITRARY float coordinates (not a fixed resize
// grid), so a generic border-aware fetch is needed. Five border types per the
// CV-CUDA Remap spec: CONSTANT / REPLICATE / REFLECT / WRAP / REFLECT101.
//
// GetIndexWithBorder is a VERBATIM port of the CUDA original (BorderWrap.hpp
// lines 75-127), including the integer modular arithmetic for REFLECT
// (abs(2*c+1-s2)>>1) and REFLECT101 (2*s-2 period). Do not "simplify" — the
// bit-exact behavior must match CV-CUDA. CONSTANT is NOT handled here (caller
// returns borderValue on out-of-range); getIndexWithBorder is a no-op for it.
//
// Implemented host+device portable (no std::abs — uses a local iabs) so the
// SAME code runs in the CPU reference and the device kernel.
//
// SPDX-License-Identifier: Apache-2.0 (NVIDIA CV-CUDA original algorithm)

#ifndef CVSACL_BORDER_HPP
#define CVSACL_BORDER_HPP

namespace cvsycl {

enum class Border {
    Constant,
    Replicate,
    Reflect,
    Wrap,
    Reflect101,
};

inline int iabs(int x) { return x < 0 ? -x : x; }

// Is c outside the valid range [0, s)?
inline bool isOutside(int c, int s) { return c < 0 || c >= s; }

// Remap coordinate c into [0, s) according to border type B.
// For Border::Constant this is a no-op (caller must handle out-of-range by
// returning the border value). Mirrors BorderWrap.hpp::GetIndexWithBorder.
inline int getIndexWithBorder(Border b, int c, int s) {
    switch (b) {
        case Border::Replicate:
            c = (c < 0) ? 0 : (c >= s ? s - 1 : c);
            break;
        case Border::Wrap:
            c = c % s;
            if (c < 0) c += s;
            break;
        case Border::Reflect: {
            int s2 = s * 2;
            c = c % s2;
            if (c < 0) c += s2;
            c = s - 1 - (iabs(2 * c + 1 - s2) >> 1);
            break;
        }
        case Border::Reflect101:
            if (s == 1) {
                c = 0;
            } else {
                c = c % (2 * s - 2);
                if (c < 0) c += 2 * s - 2;
                c = s - 1 - iabs(s - 1 - c);
            }
            break;
        case Border::Constant:
            // Caller handles out-of-range; identity here.
            break;
    }
    return c;
}

}  // namespace cvsycl

#endif  // CVSACL_BORDER_HPP
