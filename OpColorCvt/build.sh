#!/usr/bin/env bash
# Build script for the SYCL CvtColor port (CV-CUDA CvtColor, tensor path).
# Header-only kernel (colorcvt.hpp) — only the 2 test binaries need compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as the OpResize / OpNMS ports.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_cvt
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
# (CvtColor float paths do float dot products / divisions; precise flags reduce
# non-associativity drift vs the CPU reference, though the tests already tolerate
# small differences per CV-CUDA's own tolerances).
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"

# 1. correctness (all 6 families / 71 codes vs CPU gold, + negative tests)
"${CXX}" "${FLAGS[@]}" test/test_cvt.cpp -o test/test_cvt
echo "[build.sh] built test/test_cvt"

# 2. per-kernel profiling (VOX_PROFILE=1 self-contained at runtime)
"${CXX}" "${FLAGS[@]}" test/test_cvt_profile.cpp -o test/test_cvt_profile
echo "[build.sh] built test/test_cvt_profile"

echo "[build.sh] all 2 CvtColor targets built."
