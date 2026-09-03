#!/usr/bin/env bash
# GPTQ INT4 -> s4 hist + ESIMD s4 DPAS oracle.
# Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/esimd_dpas/bin/dpas_s4_ckpt"
DUMP="$ROOT/results/k6/gptq_s4_down0_256.bin"
OUT="$ROOT/results/k6/gptq_s4_card${CARD}.txt"
FREQ="$ROOT/results/k6/gptq_s4_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
set +u
source "$ONEAPI/setvars.sh" --force >/dev/null
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$CARD"
export ZES_ENABLE_SYSMAN=1
export GPTQ_DUMP="$DUMP"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dpas_s4_ckpt gptq-int4 g128 sym"
  echo "=== dump/hist (CPU) ==="
  python3 "$ROOT/kernels/nvfp4/gptq_s4_dump.py"
  echo "dump_rc=$?"
  echo "=== esimd s4 on dump ==="
  set +e
  "$BIN" --b-bin "$DUMP" --warmup 5 --iters 20
  echo "bin_rc=$?"
  set -e
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
