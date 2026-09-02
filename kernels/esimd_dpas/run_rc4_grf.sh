#!/usr/bin/env bash
# Usage: gpu-run --card N bash this.sh N rc4|grf256
set -euo pipefail
CARD="${1:?card}"
ARM="${2:?arm}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_${ARM}"
OUT="$ROOT/results/k2/${ARM}_card${CARD}.txt"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_${ARM}"
  echo "oneDNN W8A8 floor M=1/64/256 5120: 42-46 / 46-49 / 74-76 us"
  if [ "$ARM" = "rc4" ]; then
    shapes="4 5120 5120
64 5120 5120
256 5120 5120"
  else
    shapes="8 5120 5120
64 5120 5120
256 5120 5120"
  fi
  echo "$shapes" | while read -r m n k; do
    [ -z "$m" ] && continue
    echo "=== ${ARM} m=$m n=$n k=$k ==="
    "$BIN" --m "$m" --n "$n" --k "$k" --warmup 5 --iters 20
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT"
