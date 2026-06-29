#!/usr/bin/env bash
# Build script for the SYCL resize port (phase 1: F32, C=1, NEAREST+LINEAR).
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present, e.g. inside the container — but the container has no
# icpx, so always build on the host). Same convention as the lidar voxelization
# port's build.sh.
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
"${CXX}" "${FLAGS[@]}" resize.cpp test/test_resize.cpp -o test/test_resize
echo "[build.sh] built test/test_resize"
