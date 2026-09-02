#!/usr/bin/env bash
# Heat the card, then time decode M=1/4/64 with freq samples.
# Usage: gpu-run --card N bash this.sh N NT
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
HEAT="$ROOT/kernels/esimd_dpas/bin/dpas_s8"
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_dec"
OUT="$ROOT/results/k2/hold_n${NT}_card${CARD}.txt"
FREQ="$ROOT/results/k2/hold_n${NT}_card${CARD}.freq"
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
    sleep 0.2
  done
}
sample >"$FREQ" &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_dec nt=$NT heat=dpas_s8_1024^3_80iters"
  echo "oneDNN W8A8 floor M=1 5120: 42-46 us; wgn M=4 NT=2 47-50 us"
  echo "=== heat dpas_s8 1024^3 ==="
  "$HEAT" --m 1024 --n 1024 --k 1024 --warmup 20 --iters 80
  echo "=== clocks after heat ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  for shape in "1 5120 5120" "4 5120 5120" "64 5120 5120"; do
    set -- $shape
    echo "=== hold dec nt=$NT m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 5 --iters 20
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
