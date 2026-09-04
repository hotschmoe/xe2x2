#!/usr/bin/env bash
# K8: 16x16 E2M1 product-LUT expert-up/down GEMV hail-mary.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/prod_lut_gemv_n1856"
OUT="$ROOT/results/k8/prod_lut_gemv_down_m1_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/prod_lut_gemv_down_m1_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k8"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=prod_lut_gemv_n1856 table16x16 m=1 n=1856 k=2688 lightning_expert_down_gemv never_bitcast_s4 spin=$SPIN"
  echo "Expert down GEMV orientation requested as n=1856 k=2688. Rank pipe_host."
  echo "=== product LUT GEMV m=1 n=1856 k=2688 spin=$SPIN ==="
  "$BIN" --m 1 --n 1856 --k 2688 --warmup 50 --iters 40 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"

