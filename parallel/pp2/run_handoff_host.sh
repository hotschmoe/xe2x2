#!/usr/bin/env bash
# P3 synthetic host-staged activation handoff. Owns BOTH cards.
# Usage: gpu-run bash this.sh
set -euo pipefail
ROOT=/mnt/vm_8tb/github/xe2x2
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
OUT="$ROOT/results/p3/handoff_host.txt"
FREQ0="$ROOT/results/p3/handoff_host_card0.freq"
FREQ1="$ROOT/results/p3/handoff_host_card1.freq"
mkdir -p "$ROOT/results/p3"

sample() {
  local card="$1" dest="$2"
  local gt="/sys/class/drm/card${card}/device/tile0/gt0/freq0"
  while true; do
    echo "freq_sample card=${card} t=$(date +%s.%N) power=$(cat /sys/class/drm/card${card}/device/power_state) act=$(cat "$gt/act_freq") cur=$(cat "$gt/cur_freq") throttle=$(cat "$gt/throttle/status" 2>/dev/null || echo n/a)"
    sleep 0.05
  done >"$dest"
}
sample 0 "$FREQ0" &
sp0=$!
sample 1 "$FREQ1" &
sp1=$!
cleanup() { kill "$sp0" "$sp1" 2>/dev/null || true; wait "$sp0" 2>/dev/null || true; wait "$sp1" 2>/dev/null || true; }
trap cleanup EXIT

{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" 0
  bash "$ROOT/scripts/clocks.sh" 1
  echo "CONFIG backend=pytorch-xpu on sycl+l0 fabric=host_staged_pp2 p2p=0 img=$IMG"
  echo "P3 synthetic stage0 -> host -> stage1. Identity of the tensor."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
    -v "$ROOT:/work:ro" \
    -e CCL_TOPO_P2P_ACCESS=0 \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
    -e ZE_AFFINITY_MASK=0,1 \
    --entrypoint python3 \
    "$IMG" /work/parallel/pp2/bench_handoff_host.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" 0
  bash "$ROOT/scripts/clocks.sh" 1
} | tee "$OUT"
echo "wrote $OUT $FREQ0 $FREQ1"
