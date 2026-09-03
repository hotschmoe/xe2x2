#!/usr/bin/env bash
# CPU-only AOT for extra K6 TUs. No --device. No gpu-run.
# Usage: ./compile_extra.sh nibble_lut_sc.cpp
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/xe2x2}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi}"
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 src.cpp [src.cpp ...]" >&2
  exit 2
fi

SRCS="$*"
mkdir -p "${ROOT}/kernels/nvfp4/bin" "${ROOT}/results/k6"

docker run --rm \
  -e SRCS="$SRCS" \
  -v "${ROOT}:/work" \
  -v "${ONEAPI}:/opt/intel/oneapi:ro" \
  --entrypoint bash "${IMG}" \
  -lc 'set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cd /work/kernels/nvfp4
mkdir -p bin /work/results/k6
CXXFLAGS="-fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++17 -Wall"
fail=0
for src in $SRCS; do
  stem="${src%.cpp}"
  dst="bin/${stem}"
  slog="/work/results/k6/${stem}.compile.log"
  echo "COMPILE_${stem}"
  {
    echo "compile start $(date -u +%Y-%m-%d_%H:%M:%SZ)"
    icpx --version
  } >"$slog"
  set +e
  icpx $CXXFLAGS -o "$dst" "$src" >>"$slog" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "COMPILE_OK ${dst}" | tee -a "$slog"
    ls -l "$dst" | tee -a "$slog"
  else
    echo "COMPILE_REFUSED ${src} rc=${rc} log=${slog}" | tee -a "$slog"
    tail -n 80 "$slog"
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo COMPILE_OK
else
  echo COMPILE_PARTIAL
  exit 1
fi
'
