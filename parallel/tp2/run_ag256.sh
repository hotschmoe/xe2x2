#!/usr/bin/env bash
# P2 leftover: XCCL all_gather+sendrecv at 2.5 MiB only. Outer timeout 45s.
# Usage: gpu-run bash this.sh
set -euo pipefail
ROOT=/mnt/vm_8tb/github/xe2x2
IMG="${IMG:-b70-sglang-xpu-int8-runtime:20260826-mtp6}"
CCL_ROOT="${CCL_ROOT:-/opt/venv}"
OUT="$ROOT/results/p2/xccl_ag256.txt"
mkdir -p "$ROOT/results/p2"
{
  echo "CONFIG backend=pytorch-xpu on sycl+l0 fabric=xccl p2p=0 timeout=45s"
  echo "P2 leftover: all_gather 2.5MiB. Hang is a RESULT if timeout."
  set +e
  timeout --signal=TERM --kill-after=8 45 docker run --rm --device /dev/dri \
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
    --standalone --nproc-per-node=2 /work/parallel/tp2/bench_ag256.py
  rc=$?
  set -e
  echo "TIMEOUT_OR_EXIT rc=$rc"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "RESULT op=all_gather name=prefill_256h HANG timeout=45s p2p=0"
  fi
} | tee "$OUT"
echo "wrote $OUT"
exit 0
