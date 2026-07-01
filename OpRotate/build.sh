#!/usr/bin/env bash
# Build script for the SYCL Rotate port (CV-CUDA Rotate, tensor path).
# Header-only kernel (rotate.hpp) — only the 3 test binaries need compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as the OpResize / OpNMS / OpColorCvt
# / OpNormalize ports.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_rotate
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh
  set -u
else
  echo "[build.sh] no /opt/intel/setvars.sh (host build — relying on PATH icpx)"
fi

CXX=${CXX:-icpx}
# Precise FP flags kept for consistency with the other cvcuda-ops-sycl ports.
# LINEAR/CUBIC use only mul/add (no div/sqrt), so the Intel GPU FP-division
# quirk does NOT apply; residual ULP noise is FMA-contraction only. Flags do not
# change results, kept for parity.
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"

# 1. synthetic correctness (bit-exact vs CPU gold; NEAREST any-angle, LINEAR 0/180, CUBIC 0)
"${CXX}" "${FLAGS[@]}" test/test_rotate.cpp -o test/test_rotate
echo "[build.sh] built test/test_rotate"

# 2. real random data: 3 interps, U8+F32, FP-boundary analysis + wall perf
"${CXX}" "${FLAGS[@]}" test/test_rotate_real.cpp -o test/test_rotate_real
echo "[build.sh] built test/test_rotate_real"

# 3. per-kernel profiling (VOX_PROFILE=1 at runtime)
"${CXX}" "${FLAGS[@]}" test/test_rotate_profile.cpp -o test/test_rotate_profile
echo "[build.sh] built test/test_rotate_profile"

echo "[build.sh] all 3 Rotate targets built."
