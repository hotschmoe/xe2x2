#!/usr/bin/env bash
# Held-clock GPTQ s4 RC=4 decode, M=1 N=5120 K=17408 (down_proj).
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-4000}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s4_gptq_sc"
DUMP="$ROOT/results/k6/gptq_s4_down0_k17408_sc.bin"
OUT="$ROOT/results/k6/gptq_s4sc_k17408_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k6/gptq_s4sc_k17408_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
export TILE_K=17408
export TILE_N=5120
export GPTQ_WANT=down_proj
export GPTQ_DUMP="$ROOT/results/k6/gptq_s4_down0_k17408.bin"
export GPTQ_DUMP_SC="$DUMP"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s4_gptq_sc nt=$NT spin=$SPIN k=17408 gptq g128"
  echo "GPTQ square 29.9. N-wide 100. s4 K=17408 53.4. s8 261.6. Napkin 29.9*17408/5120 ~102."
  echo "=== dump (CPU) ==="
  python3 "$ROOT/kernels/nvfp4/gptq_s4_dump.py"
  echo "dump_rc=$?"
  echo "=== decode ==="
  set +e
  for shape in "1 5120 17408" "4 5120 17408"; do
    set -- $shape
    echo "=== gptqsc nt=$NT spin=$SPIN m=$1 n=$2 k=$3 ==="
    "$BIN" --b-bin "$DUMP" --nt "$NT" --m "$1" --n "$2" --k "$3" \
      --warmup 50 --iters 40 --card "$CARD" --spin "$SPIN" --mhz 2400
  done
  echo "bin_rc=$?"
  set -e
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
