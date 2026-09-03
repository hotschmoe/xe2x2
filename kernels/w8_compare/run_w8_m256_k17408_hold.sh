#!/usr/bin/env bash
# Held-clock oneDNN int8_gemm_w8a8 M=256 N=5120 K=17408.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k2/w8a8_m256_k17408_hold_card${CARD}.txt"
FREQ="$ROOT/results/k2/w8a8_m256_k17408_hold_card${CARD}.freq"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=int8_gemm_w8a8 spin=$SPIN time=M256 n=5120 k=17408"
  echo "W8A8 M=256 square 75. W8A8 N=17408 248. w4a16 K=17408 377. s8 477.4. s4 149.0. Napkin 75*155.3/44 ~265."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e W8_SPIN="$SPIN" \
    -e W8_N=5120 \
    -e W8_K=17408 \
    --entrypoint python3 \
    "$IMG" /work/kernels/w8_compare/bench_w8_m256_wide.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
