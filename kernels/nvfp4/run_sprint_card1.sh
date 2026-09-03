#!/usr/bin/env bash
# Sprint card1: mixed dpas probe, then 16x16 product LUT GEMV.
# Usage: gpu-run --card 1 bash this.sh 1
set -u
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
OUT="$ROOT/results/k6/sprint_mix_prod_card${CARD}.txt"
FREQ="$ROOT/results/k6/sprint_mix_prod_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD sprint=mix+prod_lut_gemv never_bitcast_s4"
  echo "=== mix ==="
  "$ROOT/kernels/nvfp4/bin/sprint_dpas_mix"
  echo "mix_rc=$?"
  echo "=== prod_lut_gemv ==="
  "$ROOT/kernels/nvfp4/bin/prod_lut_gemv"
  echo "prod_rc=$?"
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
