#!/usr/bin/env bash
# Held-clock s4 RC=4 8x2-N scale-to-f16, M=1 pad, no SLM.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/compose_e2m1_sc"
OUT="$ROOT/results/k6/e2m1sc_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k6/e2m1sc_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=compose_e2m1_sc two_term_s4 nt=$NT spin=$SPIN heat=none out=f16 never_bitcast_s4"
  echo "s4 decode 16.5 us. nibble_lut_sc 158 us. Two s4 DPAS w_lo+8*w_hi."
  for shape in "1 5120 5120" "4 5120 5120"; do
    set -- $shape
    echo "=== e2m1sc nt=$NT spin=$SPIN m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 50 --iters 40 \
      --card "$CARD" --spin "$SPIN" --mhz 2400
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
