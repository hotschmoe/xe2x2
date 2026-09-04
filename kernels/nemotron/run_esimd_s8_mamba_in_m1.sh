#!/usr/bin/env bash
# K8: ESIMD s8 mamba in_proj M=1 n=10304 k=2688 U=14.
# N napkin: z/x 4096+4096 + B/C 1024+1024 + dt 64. Confirm at runtime later.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_sc_u14"
NT=2
OUT="$ROOT/results/k8/esimd_s8_mamba_in_m1_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/esimd_s8_mamba_in_m1_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_sc_u14 nt=$NT spin=$SPIN m=1 n=10304 k=2688 lightning_mamba_in"
  echo "Official PTQ is FP8. s8 is the beat-me control. packed qkv 16.609 launch-class."
  echo "=== esimd s8 mamba-in m=1 n=10304 k=2688 nt=$NT u=14 spin=$SPIN ==="
  "$BIN" --nt "$NT" --m 1 --n 10304 --k 2688 --warmup 50 --iters 40 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
