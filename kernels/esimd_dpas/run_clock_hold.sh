#!/usr/bin/env bash
# Long occupancy DPAS so GT clocks can rise. Sample sysfs during the run.
# Usage: gpu-run --card N bash this.sh N ARM
# ARM = s8|s4|s2|s2xs8
set -euo pipefail
CARD="${1:?card}"
ARM="${2:?arm}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_${ARM}"
OUT="$ROOT/results/k2/clockhold_${ARM}_card${CARD}.txt"
# shellcheck disable=SC1091
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "=== dpas_${ARM} long ==="
  "$BIN" --m 1024 --n 1024 --k 1024 --iters 80 --warmup 20
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
# sample once more after
echo "wrote $OUT"
