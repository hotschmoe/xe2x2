#!/usr/bin/env bash
# WG-256 RMSNorm-quant producer then s8 GEMM, M=1 N=17408 K=5120.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_prod"
OUT="$ROOT/results/k5/prod_n17408_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k5/prod_n17408_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k5"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
sample() {
  while true; do
    echo "freq_sample t=$(date +%s.%N) power=$(cat /sys/class/drm/card${CARD}/device/power_state) act=$(cat "$GT/act_freq") cur=$(cat "$GT/cur_freq") throttle=$(cat "$GT/throttle/status" 2>/dev/null || echo n/a)"
    sleep 0.05
  done
}
sample >"$FREQ" &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_prod nt=$NT spin=$SPIN prod+gemm n=17408 k=5120"
  echo "square pair 44 us. s8 GEMM N=17408 141.6. Napkin 44*17408/5120 ~151."
  for shape in "1 17408 5120" "4 17408 5120"; do
    set -- $shape
    echo "=== prod nt=$NT spin=$SPIN m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 50 --iters 40 \
      --card "$CARD" --spin "$SPIN" --mhz 2400
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
