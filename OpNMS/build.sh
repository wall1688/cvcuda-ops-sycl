#!/usr/bin/env bash
# Build script for the SYCL NMS port (CV-CUDA NonMaximumSuppression).
# Header-only kernel (nms.hpp) — only the 3 test binaries need compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as the OpResize port.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_nms
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
# Precise FP flags kept for consistency with the other cvcuda-ops-sycl ports
# (they do NOT change NMS results — only the final IoU division is float, and the
# GPU div is reciprocal-approx regardless).
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"

# 1. synthetic correctness (bit-exact vs CPU gold)
"${CXX}" "${FLAGS[@]}" test/test_nms.cpp -o test/test_nms
echo "[build.sh] built test/test_nms"

# 2. real random boxes: correctness + FP-boundary analysis + wall perf
"${CXX}" "${FLAGS[@]}" test/test_nms_real.cpp -o test/test_nms_real
echo "[build.sh] built test/test_nms_real"

# 3. per-kernel profiling (VOX_PROFILE=1 at runtime)
"${CXX}" "${FLAGS[@]}" test/test_nms_profile.cpp -o test/test_nms_profile
echo "[build.sh] built test/test_nms_profile"

echo "[build.sh] all 3 NMS targets built."
