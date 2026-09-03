#!/usr/bin/env bash
# Held-clock s8xs4 RC=4 8x2-N scale-to-f16, M=1 N=5120 K=17408.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8xs4_sc"
OUT="$ROOT/results/k2/s8xs4sc_k17408_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k2/s8xs4sc_k17408_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k2"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8xs4_sc nt=$NT spin=$SPIN heat=none out=f16 packB=s4 A=s8 k=17408"
  echo "s8xs4 square 22.1. N-wide 38.6. s8 K=17408 261.6. s4 53.4. Napkin 22.1*17408/5120 ~75."
  for shape in "1 5120 17408" "4 5120 17408"; do
    set -- $shape
    echo "=== s8xs4sc nt=$NT spin=$SPIN m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 50 --iters 40 \
      --card "$CARD" --spin "$SPIN" --mhz 2400
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
