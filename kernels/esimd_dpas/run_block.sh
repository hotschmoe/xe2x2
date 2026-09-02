#!/usr/bin/env bash
# Blocked s8 DPAS vs oneDNN 45 us. Usage: gpu-run --card N bash this.sh N NT
# NT = 2 (8x32) or 4 (8x64)
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_block"
OUT="$ROOT/results/k2/block_n${NT}_card${CARD}.txt"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_block nt=$NT"
  echo "oneDNN W8A8 floor M=1/64/256 5120: 42-46 / 46-49 / 74-76 us"
  for shape in "8 5120 5120" "64 5120 5120" "256 5120 5120"; do
    set -- $shape
    echo "=== nt=$NT m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 5 --iters 20
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT"
