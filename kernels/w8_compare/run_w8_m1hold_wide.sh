#!/usr/bin/env bash
# W8A8 M=1 held-clock. Usage: gpu-run --card C bash this.sh C [N] [K]
# Default N=17408 K=5120 (FFN-up). Down-proj is N=5120 K=17408.
set -euo pipefail
CARD="${1:?card}"
NDIM="${2:-17408}"
KDIM="${3:-5120}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
TAG="n${NDIM}_k${KDIM}"
OUT="$ROOT/results/k2/w8a8_m1hold_${TAG}_card${CARD}.txt"
FREQ="$ROOT/results/k2/w8a8_m1hold_${TAG}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k2"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD spin=M1 time=M1 n=$NDIM k=$KDIM"
  echo "hand s8 decode N=17408 141.6 us / K=17408 261.6 us at 2800. s4 29.5 / 53.4."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e W8_N="$NDIM" \
    -e W8_K="$KDIM" \
    --entrypoint python3 \
    "$IMG" /work/kernels/w8_compare/bench_w8_m1_wide.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
