#!/usr/bin/env bash
# Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_slm"
OUT="$ROOT/results/k2/slm_card${CARD}.txt"
mkdir -p "$ROOT/results/k2"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_slm wg=16 slmA RC=4"
  echo "oneDNN W8A8 floor M=1/64/256 5120: 42-46 / 46-49 / 74-76 us"
  for shape in "4 5120 5120" "64 5120 5120" "256 5120 5120"; do
    set -- $shape
    echo "=== slm m=$1 n=$2 k=$3 ==="
    "$BIN" --m "$1" --n "$2" --k "$3" --warmup 5 --iters 20
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT"
