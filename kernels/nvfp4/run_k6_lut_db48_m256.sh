#!/usr/bin/env bash
# Held-clock NVFP4 nibble LUT on 4x8 A-db, M=256 N=K=5120.
# Packed E2M1 B in HBM. Never bitcast onto s4.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/nibble_lut_db48"
OUT="$ROOT/results/k6/lutdb48_m256_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k6/lutdb48_m256_n${NT}_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=nibble_lut_db48 nt=$NT spin=$SPIN m=256 RC=8 A_db wg=4x8 no_slm out=f16 never_bitcast_s4"
  echo "LUT 4x8 M=64 392.4 us. s8 4-acc 128. compose 194.9. W8A8 75. Napkin M-linear 1570."
  echo "=== lut_db48 nt=$NT m=256 n=5120 k=5120 ==="
  set +e
  "$BIN" --nt "$NT" --m 256 --n 5120 --k 5120 --warmup 10 --iters 20 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  rc=$?
  set -e
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  exit "$rc"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
