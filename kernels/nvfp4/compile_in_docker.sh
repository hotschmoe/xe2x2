#!/usr/bin/env bash
# CPU-only AOT for K6 nibble LUT. No --device. No gpu-run.
set -euo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/github/xe2x2}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi}"
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
mkdir -p "${ROOT}/kernels/nvfp4/bin" "${ROOT}/results/k6"
docker run --rm \
  -v "${ROOT}:/work" \
  -v "${ONEAPI}:/opt/intel/oneapi:ro" \
  --entrypoint bash "${IMG}" \
  -lc 'set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
cd /work/kernels/nvfp4
mkdir -p bin /work/results/k6
LOG=/work/results/k6/compile.log
icpx --version | tee "$LOG"
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++17 -Wall \
  -o bin/nibble_lut_s8 nibble_lut_s8.cpp 2>&1 | tee -a "$LOG"
ls -l bin/nibble_lut_s8 | tee -a "$LOG"
echo COMPILE_OK | tee -a "$LOG"
'
