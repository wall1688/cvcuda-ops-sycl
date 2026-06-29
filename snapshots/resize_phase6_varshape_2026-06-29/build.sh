#!/usr/bin/env bash
# Build script for the SYCL resize port.
# Phase 2: F32/U8, C=1/3/4, NHWC, NEAREST+LINEAR. Header-only kernels
# (resize.hpp) — only the test needs compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as the lidar voxelization port.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_resize
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
# Precise FP flags kept for consistency with the voxelization port (the resize
# kernels do float weighted sums; precise flags reduce non-associativity drift
# vs the CPU reference, though the test already tolerates small differences).
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"
"${CXX}" "${FLAGS[@]}" test/test_resize.cpp -o test/test_resize
echo "[build.sh] built test/test_resize"
"${CXX}" "${FLAGS[@]}" test/test_resize_profile.cpp -o test/test_resize_profile
echo "[build.sh] built test/test_resize_profile"
"${CXX}" "${FLAGS[@]}" test/test_resize_varshape.cpp -o test/test_resize_varshape
echo "[build.sh] built test/test_resize_varshape"
