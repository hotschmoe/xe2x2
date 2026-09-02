#!/usr/bin/env bash
# CPU-only AOT for K3 precision-compose micros. No --device. No gpu-run.
# Host has no g++. Compile inside the K0-proven runtime image with the
# host oneAPI 2026.1.1 tree bind-mounted.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/xe2x2}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi}"
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
LOGDIR="${ROOT}/results/k3"

mkdir -p "${ROOT}/kernels/precision_compose/bin" "${LOGDIR}"

docker run --rm \
  -v "${ROOT}:/work" \
  -v "${ONEAPI}:/opt/intel/oneapi:ro" \
  --entrypoint bash "${IMG}" \
  -lc 'set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cd /work/kernels/precision_compose
mkdir -p bin /work/results/k3
LOG=/work/results/k3/compile.log
: > "$LOG"
{
  echo "compile start $(date -u +%Y-%m-%d_%H:%M:%SZ)"
  echo "icpx=$(command -v icpx)"
  icpx --version
  echo "g++=$(command -v g++ || true) $(g++ --version 2>/dev/null | head -1 || true)"
} | tee -a "$LOG"
CXXFLAGS="-fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++17 -Wall"
fail=0
for src in compose_s8_from_s4.cpp compose_s8_from_s4_karatsuba.cpp compose_e2m1_two_term.cpp; do
  stem="${src%.cpp}"
  dst="bin/${stem}"
  slog="/work/results/k3/${stem}.compile.log"
  echo "COMPILE_${stem}" | tee -a "$LOG"
  set +e
  icpx $CXXFLAGS -o "$dst" "$src" >"$slog" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "COMPILE_OK ${dst}" | tee -a "$LOG"
    ls -l "$dst" | tee -a "$LOG"
  else
    echo "COMPILE_REFUSED ${src} rc=${rc} log=${slog}" | tee -a "$LOG"
    echo "----- ${stem} refusal (tail) -----" | tee -a "$LOG"
    tail -n 80 "$slog" | tee -a "$LOG"
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo COMPILE_OK | tee -a "$LOG"
else
  echo COMPILE_PARTIAL | tee -a "$LOG"
fi
'
