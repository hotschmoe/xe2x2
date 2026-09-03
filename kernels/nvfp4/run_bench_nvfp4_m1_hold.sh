#!/usr/bin/env bash
# Held-clock oneDNN nvfp4_gemm_w4a16 M=1 5120. Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k6/nvfp4_w4a16_m1hold_card${CARD}.txt"
FREQ="$ROOT/results/k6/nvfp4_w4a16_m1hold_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k6"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=nvfp4_gemm_w4a16 heat=M64 spin=2000 time=M1 n=5120"
  echo "unheld sprint 36.8/37.2 us clocks not 2800. s8 34. W8A8 44. LUT 134.8."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -v /mnt/vm_8tb/b70/nvfp4_kernel_v028:/opt/nvfp4:ro \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e B70_XPU_C_SO=/opt/nvfp4/_xpu_C.abi3.so \
    -e NVFP4_SPIN=2000 \
    --entrypoint python3 \
    "$IMG" /work/kernels/nvfp4/bench_nvfp4_m1_hold.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
