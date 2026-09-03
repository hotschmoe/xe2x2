#!/usr/bin/env bash
# Held-clock oneDNN nvfp4_gemm_w4a16 M=64 N=17408 K=5120.
# Usage: gpu-run --card N bash this.sh N [SPIN]
set -euo pipefail
CARD="${1:?card}"
SPIN="${2:-512}"
ROOT=/mnt/vm_8tb/github/xe2x2
IMG=b70-sglang-xpu-int8-runtime:20260826-mtp6
OUT="$ROOT/results/k6/nvfp4_w4a16_m64_n17408_hold_card${CARD}.txt"
FREQ="$ROOT/results/k6/nvfp4_w4a16_m64_n17408_hold_card${CARD}.freq"
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
  echo "CONFIG backend=pytorch-xpu on sycl+l0 card=$CARD op=nvfp4_gemm_w4a16 spin=$SPIN time=M64 n=17408 k=5120"
  echo "w4a16 M=64 square 37.1. M=1 N=17408 97. s8 338.9. s4 94.7. compose 326.9. LUT 880. Napkin 37.1*97/34.7 ~104."
  docker run --rm --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$ROOT:/work:ro" \
    -v /mnt/vm_8tb/b70/nvfp4_kernel_v028:/opt/nvfp4:ro \
    -e ZE_AFFINITY_MASK="$CARD" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    -e B70_XPU_C_SO=/opt/nvfp4/_xpu_C.abi3.so \
    -e NVFP4_SPIN="$SPIN" \
    -e NVFP4_N=17408 \
    -e NVFP4_K=5120 \
    --entrypoint python3 \
    "$IMG" /work/kernels/nvfp4/bench_nvfp4_m64_hold.py
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
