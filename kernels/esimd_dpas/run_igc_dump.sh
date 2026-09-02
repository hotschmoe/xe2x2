#!/usr/bin/env bash
# Dump IGC ISA for a standalone K2 DPAS binary. One card.
# Usage: gpu-run --card N bash this.sh N ARM
# ARM = s8|s4|s2|s2xs8
set -euo pipefail
CARD="${1:?card}"
ARM="${2:?arm}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_${ARM}"
DUMP="$ROOT/results/k2/igc_card${CARD}_${ARM}"
OUT="$ROOT/results/k2/igc_card${CARD}_${ARM}.txt"
mkdir -p "$DUMP"
# shellcheck disable=SC1091
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir="$DUMP"
export SYCL_CACHE_PERSISTENT=0
export SYCL_CACHE_DIR="$DUMP/sycl_cache"
case "$ARM" in
  s8|s2xs8) K=32 ;;
  s4|s2) K=64 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "=== dpas_${ARM} igc dump (tiny tile) ==="
  echo "CONFIG backend=sycl+l0 card=$CARD arm=$ARM IGC_DumpToCustomDir=$DUMP"
  "$BIN" --check-m 8 --check-n 16 --check-k "$K" --m 8 --n 16 --k "$K" \
    --warmup 1 --iters 1
  echo "=== dump files ==="
  find "$DUMP" -type f | sort
  echo "=== dpas / DPAS / s8 / s4 / s2 hits ==="
  grep -RIn -E 'dpas|DPAS|s8|s4|s2' "$DUMP" --include='*.asm' --include='*.isaasm' \
    --include='*.visaasm' --include='*.ll' --include='*.cl' --include='*.txt' \
    | head -n 200 || true
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT dumpdir=$DUMP"
