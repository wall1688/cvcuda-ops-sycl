#!/usr/bin/env bash
# Build script for the SYCL remap port (phase 1).
# Header-only kernels (remap.hpp + common/sampler.hpp + common/border.hpp) —
# only the test needs compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as OpResize.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_remap
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
# Precise FP flags for consistency with the voxelization / resize ports.
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"
"${CXX}" "${FLAGS[@]}" test/test_remap.cpp -o test/test_remap
echo "[build.sh] built test/test_remap"
