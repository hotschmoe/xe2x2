#!/usr/bin/env bash
# K8: ESIMD s8 routed-up M=64 n=1856 k=2688 U=14.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_sc_u14"
NT=2
OUT="$ROOT/results/k8/esimd_s8_moe_up_m64_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/esimd_s8_moe_up_m64_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_sc_u14 nt=$NT spin=$SPIN m=64 n=1856 k=2688 lightning_moe_up"
  echo "M=1 s8 16.060. W8A8 M=1 44.285. Prior M=64 4x8 A-db 75 at 5120."
  echo "=== esimd s8 moe-up m=64 n=1856 k=2688 nt=$NT u=14 spin=$SPIN ==="
  "$BIN" --nt "$NT" --m 64 --n 1856 --k 2688 --warmup 20 --iters 20 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
