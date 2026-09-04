#!/usr/bin/env bash
# K8: E2M1 two-term s4 compose u14 routed-up M=256 n=1856 k=2688.
# Never bitcast. Prior: M=1 15.518, M=64 29.637.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/compose_e2m1_sc_u14"
NT=2
OUT="$ROOT/results/k8/e2m1_twoterm_moe_up_m256_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/e2m1_twoterm_moe_up_m256_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=compose_e2m1_sc_u14 two_term_s4 nt=$NT spin=$SPIN m=256 n=1856 k=2688 lightning_moe_up never_bitcast_s4"
  echo "M=1 15.518 M=64 29.637. Do not use spin=4000."
  echo "=== e2m1sc_u14 moe-up m=256 n=1856 k=2688 ==="
  "$BIN" --nt "$NT" --m 256 --n 1856 --k 2688 --warmup 10 --iters 10 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
