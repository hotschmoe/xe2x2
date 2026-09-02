#!/usr/bin/env bash
# Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/nibble_lut_reg"
OUT="$ROOT/results/k6/reg_card${CARD}.txt"
mkdir -p "$ROOT/results/k6"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "=== nibble_lut_reg ==="
  "$BIN" --m 1024 --n 1024 --k 1024 --warmup 5 --iters 20
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT"
