#!/usr/bin/env bash
# K4 launcher. Usage: run_w8.sh CARD fp8|int8a8 IMAGE
set -euo pipefail
CARD="${1:?card}"
OP="${2:?op}"
IMG="${3:?image}"
ROOT=/mnt/vm_8tb/github/xe2x2
mkdir -p "$ROOT/results/k4"
export B70_AGENT="${B70_AGENT:-xe2x2-k4-c${CARD}-${OP}}"
/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run --card "$CARD" \
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    --entrypoint python3 \
    "$IMG" /work/kernels/w8_compare/bench_w8.py --op "$OP"
