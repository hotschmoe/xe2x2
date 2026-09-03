#!/usr/bin/env bash
# Held-clock GPTQ s4 RC=8 A-db wg 4x8, M=64 N=K=5120.
# Usage: gpu-run --card N bash this.sh N NT [SPIN]
set -euo pipefail
CARD="${1:?card}"
NT="${2:?nt}"
SPIN="${3:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s4_gptq_db48"
DUMP="$ROOT/results/k6/gptq_s4_down0_5120_sc.bin"
OUT="$ROOT/results/k6/gptq_s4db48_m64_n${NT}_s${SPIN}_card${CARD}.txt"
FREQ="$ROOT/results/k6/gptq_s4db48_m64_n${NT}_s${SPIN}_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
export TILE_K=5120
export TILE_N=5120
export GPTQ_DUMP="$ROOT/results/k6/gptq_s4_down0_5120.bin"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s4_gptq_db48 nt=$NT spin=$SPIN m=64 RC=8 A_db wg=4x8 gptq g128"
  echo "GPTQ 8x2-N M=64 123.5 a loss. s4 4x8 33.6. mix 43.3. W8A8 46. Napkin 33.6*1.81 ~61."
  echo "=== dump (CPU) ==="
  if [[ -s "$DUMP" ]]; then
    echo "reuse $DUMP"
    echo "dump_rc=0"
  else
    python3 "$ROOT/kernels/nvfp4/gptq_s4_dump.py"
    echo "dump_rc=$?"
  fi
  echo "=== gptqdb48 nt=$NT spin=$SPIN m=64 n=5120 k=5120 ==="
  "$BIN" --b-bin "$DUMP" --nt "$NT" --m 64 --n 5120 --k 5120 \
    --warmup 10 --iters 20 --card "$CARD" --spin "$SPIN" --mhz 2400
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
