#!/usr/bin/env bash
# Held-clock oneDNN int8_gemm_w8a8 M=64 N=17408 K=5120.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k2/w8a8_m64_n17408_hold_card${CARD}.txt"
FREQ="$ROOT/results/k2/w8a8_m64_n17408_hold_card${CARD}.freq"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=int8_gemm_w8a8 spin=$SPIN time=M64 n=17408 k=5120"
  echo "W8A8 M=64 square 46. w4a16 N=17408 142. s8 338.9. s4 94.7. Napkin 46*17408/5120 ~156."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e W8_SPIN="$SPIN" \
    -e W8_M=64 \
    -e W8_N=17408 \
    -e W8_K=5120 \
    --entrypoint python3 \
    "$IMG" /work/kernels/w8_compare/bench_w8_m256_wide.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
