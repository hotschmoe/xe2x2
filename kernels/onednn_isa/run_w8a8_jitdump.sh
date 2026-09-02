#!/usr/bin/env bash
# ONEDNN_JIT_DUMP int8_gemm_w8a8 ngen bins. Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
DUMP="$ROOT/results/k1/igc_card${CARD}_int8a8_jit"
mkdir -p "$DUMP"
export B70_AGENT="${B70_AGENT:-xe2x2-k1-w8a8jit-c${CARD}}"
/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run --card "$CARD" \
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -v "$DUMP:/igc" \
    -w /igc \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e IGC_ShaderDumpEnable=1 \
    -e IGC_DumpToCustomDir=/igc \
    -e ONEDNN_JIT_DUMP=1 \
    -e ONEDNN_VERBOSE=all \
    -e SYCL_CACHE_PERSISTENT=0 \
    --entrypoint python3 \
    "$IMG" /work/kernels/onednn_isa/dump_incumbent.py --op int8a8 --warmup 2 --iters 3
echo "dump files: $(find "$DUMP" -type f | wc -l)"
ls -la "$DUMP" | head -40
