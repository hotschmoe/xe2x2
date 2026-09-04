#!/usr/bin/env bash
# P4 mixed 2x2 decode-sized. Timeout 90s. Owns BOTH cards.
# Usage: gpu-run bash this.sh
set -euo pipefail
ROOT=/mnt/vm_8tb/github/xe2x2
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
CCL_ROOT="${CCL_ROOT:-/opt/venv}"
OUT="$ROOT/results/p4/2x2_decode.txt"
mkdir -p "$ROOT/results/p4"
{
  echo "CONFIG backend=pytorch-xpu on sycl+l0 fabric=2x2_decode p2p=0 timeout=90s"
  echo "P4: PP sendrecv + TP all_reduce at hidden 5120 bf16."
  set +e
  timeout --signal=TERM --kill-after=10 90 docker run --rm --device /dev/dri \
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
    --standalone --nproc-per-node=2 /work/parallel/2x2/bench_decode.py
  rc=$?
  set -e
  echo "TIMEOUT_OR_EXIT rc=$rc"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "RESULT op=2x2_decode HANG timeout=90s p2p=0"
  fi
} | tee "$OUT"
echo "wrote $OUT"
exit 0
