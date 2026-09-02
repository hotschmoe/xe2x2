#!/usr/bin/env bash
# In-kernel GEMM repeats so GT stays busy while sysfs is sampled.
# Usage: gpu-run --card N bash this.sh N NT
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s8_rep"
OUT="$ROOT/results/k2/rep_n${NT}_card${CARD}.txt"
FREQ="$ROOT/results/k2/rep_n${NT}_card${CARD}.freq"
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
    sleep 0.1
  done
}
sample >"$FREQ" &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT
{
  echo "=== clocks start ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s8_rep nt=$NT reps=4096 hold=in-kernel"
  echo "oneDNN W8A8 floor M=1 5120: 42-46 us; warm wgn M=4 NT=2 47-50 us"
  echo "=== rep nt=$NT m=1 reps=4096 ==="
  "$BIN" --nt "$NT" --m 1 --n 5120 --k 5120 --reps 4096 --warmup 1 --iters 3
  echo "=== clocks mid ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
  echo "=== rep nt=$NT m=4 reps=2048 ==="
  "$BIN" --nt "$NT" --m 4 --n 5120 --k 5120 --reps 2048 --warmup 1 --iters 3
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
