#!/usr/bin/env bash
# K1 launcher. Pair with gpu-run --card N and ZE_AFFINITY_MASK=N.
# Usage: run_dump.sh CARD OP IMAGE
#   CARD 0|1
#   OP   int8|fp8|both
#   IMAGE docker tag
set -euo pipefail
CARD="${1:?card}"
OP="${2:?op}"
IMG="${3:?image}"
ROOT=/mnt/vm_8tb/github/xe2x2
OUT="$ROOT/results/k1"
DUMP="$OUT/igc_card${CARD}_${OP}"
mkdir -p "$OUT" "$DUMP"
export B70_AGENT="${B70_AGENT:-xe2x2-k1-c${CARD}-${OP}}"
/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run --card "$CARD" \
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -v "$DUMP:/igc" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e IGC_ShaderDumpEnable=1 \
    -e IGC_DumpToCustomDir=/igc \
    --entrypoint python3 \
    "$IMG" /work/kernels/onednn_isa/dump_incumbent.py --op "$OP"
echo "IGC dump files: $(find "$DUMP" -type f | wc -l)"
