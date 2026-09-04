#!/usr/bin/env bash
# K8: synthetic GPTQ-style s4 g128 control at Lightning routed-up M=1.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s4_gptq_sc_u14"
OUT="$ROOT/results/k8/gptq_s4_moe_up_u14_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k8/gptq_s4_moe_up_u14_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s4_gptq_sc_u14 weights=SYNTHETIC gs=128 nt=2 unroll=14 m=1 n=1856 k=2688 lightning_moe_up spin=$SPIN"
  echo "No Lightning GPTQ checkpoint is present; omit --b-bin for deterministic synthetic s4 B and g128 f16 scales."
  echo "=== synthetic GPTQ s4 routed-up m=1 n=1856 k=2688 nt=2 u=14 spin=$SPIN ==="
  "$BIN" --nt 2 --m 1 --n 1856 --k 2688 --warmup 50 --iters 40 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"

