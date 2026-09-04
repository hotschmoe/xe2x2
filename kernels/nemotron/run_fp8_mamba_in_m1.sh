#!/usr/bin/env bash
# K8: dump oneDNN fp8_gemm_w8a16 Lightning mamba in_proj M=1 n=10304 k=2688.
# Official PTQ is FP8 here, not NVFP4. Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k8/fp8_mamba_in_m1_card${CARD}.txt"
FREQ="$ROOT/results/k8/fp8_mamba_in_m1_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k8"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=fp8_gemm_w8a16 lightning_mamba_in n=10304 k=2688"
  echo "Official PTQ FP8. s8 control still open. ABSENT is a RESULT."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e ZES_ENABLE_SYSMAN=1 \
    --entrypoint python3 \
    "$IMG" /work/kernels/nemotron/bench_fp8_lightning.py --m 1 --n 10304 --k 2688 --name lightning_mamba_in
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
