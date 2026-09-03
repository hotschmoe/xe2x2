#!/usr/bin/env bash
# Held-clock Qwen3.8 GDN recurrent delta update. Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k7/delta_recurrent_card${CARD}.txt"
FREQ="$ROOT/results/k7/delta_recurrent_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k7"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=gdn_delta_recurrent"
  echo "48 v-heads 128x128 state. s2 decode 11.5. W8A8 44."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e ZES_ENABLE_SYSMAN=1 \
    --entrypoint python3 \
    "$IMG" /work/kernels/gdn/bench_delta_recurrent.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
