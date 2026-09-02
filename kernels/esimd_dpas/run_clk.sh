#!/usr/bin/env bash
# Tight decode loop. No 1024^3 heat. Optional batched spin of the same kernel.
# Usage: gpu-run --card N bash this.sh N NT PRIME [SPIN]
# PRIME extra in-order launches before each timed event (keep 0; host gaps).
# SPIN batched same-kernel occupancy before timed (no per-launch wait).
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
PRIME="${3:?prime}"
SPIN="${4:-0}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_clk"
OUT="$ROOT/results/k2/clk_n${NT}_p${PRIME}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k2/clk_n${NT}_p${PRIME}_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_clk nt=$NT prime=$PRIME spin=$SPIN heat=none"
  echo "oneDNN W8A8 floor M=1 5120: 42-46 us; warm wgn M=4 NT=2 47-50 us"
  echo "Do not quote a beat unless mean_us sits under 42-46 while .freq cur>=2400"
  for shape in "1 5120 5120" "4 5120 5120"; do
    set -- $shape
    echo "=== clk nt=$NT prime=$PRIME spin=$SPIN m=$1 n=$2 k=$3 ==="
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 50 --iters 40 \
      --card "$CARD" --prime "$PRIME" --spin "$SPIN" --sample-every 0 --mhz 2400
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
