#!/usr/bin/env bash
# K8: s8-A x s4-B Lightning routed-up control.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8xs4_sc_u14"
OUT="$ROOT/results/k8/s8xs4_moe_up_u14_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/s8xs4_moe_up_u14_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8xs4_sc_u14 s8_A_s4_B_control nt=2 unroll=14 inner_k=448 m=1 n=1856 k=2688 lightning_moe_up spin=$SPIN"
  echo "This is the labeled s8-A x s4-B control, not an E2M1 bitcast arm."
  echo "=== s8xs4 routed-up m=1 n=1856 k=2688 nt=2 u=14 spin=$SPIN ==="
  "$BIN" --nt 2 --m 1 --n 1856 --k 2688 --warmup 50 --iters 40 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"

