#!/usr/bin/env bash
# Build script for the SYCL Normalize port (CV-CUDA Normalize, tensor path).
# Header-only kernel (normalize.hpp) — only the 3 test binaries need compiling.
# Compile on the Intel HOST (icpx in PATH via ~/.bashrc; /opt/intel/setvars is
# sourced only if present). Same convention as the OpResize / OpNMS / OpColorCvt
# ports.
#
# Run:   bash build.sh
# Then:  ONEAPI_DEVICE_SELECTOR=opencl:gpu ./test/test_normalize
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
# stddev mode has one float division (1/sqrt) whose result the Intel GPU computes
# via reciprocal approximation regardless of these flags (see intel-gpu-fp-quirk);
# plain mode is exact. Flags do not change results, only kept for parity.
FLAGS=(-fsycl -std=c++17 -O2 -g -I "${SCRIPT_DIR}" \
       -ffp-model=precise -fno-fast-math -fimf-precision=high)

echo "[build.sh] using ${CXX}"

# 1. synthetic correctness (bit-exact vs CPU gold; plain + stddev, all 6 dtypes)
"${CXX}" "${FLAGS[@]}" test/test_normalize.cpp -o test/test_normalize
echo "[build.sh] built test/test_normalize"

# 2. real random data: correctness + stddev FP-boundary analysis + wall perf
"${CXX}" "${FLAGS[@]}" test/test_normalize_real.cpp -o test/test_normalize_real
echo "[build.sh] built test/test_normalize_real"

# 3. per-kernel profiling (VOX_PROFILE=1 at runtime)
"${CXX}" "${FLAGS[@]}" test/test_normalize_profile.cpp -o test/test_normalize_profile
echo "[build.sh] built test/test_normalize_profile"

echo "[build.sh] all 3 Normalize targets built."
