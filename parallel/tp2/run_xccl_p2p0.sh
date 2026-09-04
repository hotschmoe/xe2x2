#!/usr/bin/env bash
# P2 synthetic XCCL us sweep, P2P off. Owns BOTH cards.
# Usage: gpu-run bash this.sh
# Do not nest gpu-run. Pre/post health is the agent's job.
set -euo pipefail
ROOT=/mnt/vm_8tb/github/xe2x2
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
CCL_ROOT="${CCL_ROOT:-/opt/venv}"
OUT="$ROOT/results/p2/xccl_p2p0.txt"
FREQ0="$ROOT/results/p2/xccl_p2p0_card0.freq"
FREQ1="$ROOT/results/p2/xccl_p2p0_card1.freq"
mkdir -p "$ROOT/results/p2"

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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 fabric=xccl p2p=0 img=$IMG"
  echo "P2 synthetic all_reduce / all_gather / sendrecv. Decode hidden through 8MiB."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$ROOT:/work:ro" \
    -e CCL_ATL_TRANSPORT=ofi \
    -e CCL_TOPO_P2P_ACCESS=0 \
    -e CCL_LOG_LEVEL=warn \
    -e CCL_KERNEL_PATH="$CCL_ROOT/lib/ccl/kernels" \
    -e FI_TCP_IFACE=eth0 \
    -e CCL_KVS_IFACE=eth0 \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e LD_PRELOAD="$CCL_ROOT/lib/libccl.so.1.0" \
    -e LD_LIBRARY_PATH="$CCL_ROOT/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib" \
    --entrypoint /opt/venv/bin/torchrun \
    "$IMG" \
    --standalone --nproc-per-node=2 /work/parallel/tp2/bench_xccl_p2p0.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" 0
  bash "$ROOT/scripts/clocks.sh" 1
} | tee "$OUT"
echo "wrote $OUT $FREQ0 $FREQ1"
