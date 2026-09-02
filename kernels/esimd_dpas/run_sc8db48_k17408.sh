#!/usr/bin/env bash
# Held-clock s8 RC=8 A-db wg 4x8 f16, M=64 K=17408, no SLM.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_sc8db48"
OUT="$ROOT/results/k2/sc8db48_m64_k17408_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k2/sc8db48_m64_k17408_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k2"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_sc8db48 nt=$NT spin=$SPIN m=64 k=17408 RC=8 A_db wg=4x8 no_slm out=f16"
  echo "s8 M=64 K=5120 is 75 us. s4 M=64 K=17408 is 106.0 us. Napkin K-linear 75*17408/5120 ~255 us."
  echo "=== sc8db48 nt=$NT m=64 n=5120 k=17408 ==="
  "$BIN" --nt "$NT" --m 64 --n 5120 --k 17408 --warmup 10 --iters 20 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
