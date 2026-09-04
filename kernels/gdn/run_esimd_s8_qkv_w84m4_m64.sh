#!/usr/bin/env bash
# NEW s8 4-acc wg 8x4 k128 at packed qkv M=64 n=10240 k=5120 vs W8A8 140.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_sc8w84m4"
NT=2
OUT="$ROOT/results/k7/esimd_s8_qkv_w84m4_m64_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k7/esimd_s8_qkv_w84m4_m64_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k7"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_sc8w84m4 nt=$NT spin=$SPIN m=64 n=10240 k=5120 wg=8x4 4acc out=f16"
  echo "oneDNN packed qkv W8A8 M=64 is 138-142 us. 4x8 A-db 214 lost. square 4-acc 75."
  echo "=== esimd s8 qkv w84m4 m=64 n=10240 k=5120 nt=$NT spin=$SPIN ==="
  "$BIN" --nt "$NT" --m 64 --n 10240 --k 5120 --warmup 10 --iters 20 \
    --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
