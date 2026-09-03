#!/usr/bin/env bash
# Held-clock NVFP4 nibble LUT on the s8 RC=4 8x2-N decode tile.
# Packed E2M1 B in HBM. Never bitcast onto s4.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/nibble_lut_sct"
OUT="$ROOT/results/k6/sct_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k6/sct_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=nibble_lut_sct table16 nt=$NT spin=$SPIN heat=none out=f16 never_bitcast_s4"
  echo "nibble_lut_sc M=1 5120 is 158 us. s8 34. Table-16 iselect steal of the merge LUT."
  brc=0
  for shape in "1 5120 5120" "4 5120 5120"; do
    set -- $shape
    echo "=== lut_sct nt=$NT spin=$SPIN m=$1 n=$2 k=$3 ==="
    set +e
    "$BIN" --nt "$NT" --m "$1" --n "$2" --k "$3" --warmup 50 --iters 40 \
      --card "$CARD" --spin "$SPIN" --mhz 2400
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then brc=$rc; fi
  done
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  exit "$brc"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
