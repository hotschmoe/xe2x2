#!/usr/bin/env bash
# Sprint card0: mixed dpas probe, then lo-only compose + E2M1 bitcast.
# Usage: gpu-run --card 0 bash this.sh 0
set -u
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
OUT="$ROOT/results/k6/sprint_mix_lo_bitcast_card${CARD}.txt"
FREQ="$ROOT/results/k6/sprint_mix_lo_bitcast_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD sprint=mix+loonly+bitcast never_bitcast_s4"
  echo "=== mix ==="
  "$ROOT/kernels/nvfp4/bin/sprint_dpas_mix"
  echo "mix_rc=$?"
  echo "=== loonly ==="
  "$ROOT/kernels/nvfp4/bin/compose_e2m1_loonly" --nt 2 --m 1 --n 5120 --k 5120 \
    --warmup 50 --iters 40 --card "$CARD" --spin 4000 --mhz 2400
  echo "loonly_rc=$?"
  echo "=== bitcast ==="
  "$ROOT/kernels/nvfp4/bin/bitcast_e2m1_s4" --check-m 8 --check-n 16 --check-k 64 \
    --m 256 --n 256 --k 256 --warmup 3 --iters 10
  echo "bitcast_rc=$?"
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
