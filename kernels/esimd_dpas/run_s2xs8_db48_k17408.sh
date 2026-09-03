#!/usr/bin/env bash
# Held-clock s2xs8 RC=8 A-db wg 4x8, M=64 N=5120 K=17408.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s2xs8_db48"
OUT="$ROOT/results/k2/s2xs8db48_m64_k17408_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k2/s2xs8db48_m64_k17408_n${NT}_s${SPIN}_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s2xs8_db48 nt=$NT spin=$SPIN m=64 k=17408 RC=8 A_db wg=4x8 packB=s2 A=s8"
  echo "s2xs8 4x8 33.2. N=17408 100.5. s2 64. mix 144.7. W8A8 181. Napkin 33.2*17408/5120 ~113."
  echo "=== s2xs8db48 nt=$NT m=64 n=5120 k=17408 ==="
  "$BIN" --nt "$NT" --m 64 --n 5120 --k 17408 --warmup 10 --iters 20 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
