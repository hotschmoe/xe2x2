#!/usr/bin/env bash
# K8: dyadic s2/s4 planes hail-mary at Lightning routed-up M=1.
# Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/dyadic_s2"
OUT="$ROOT/results/k8/dyadic_s2_moe_up_m1_card${CARD}.txt"
FREQ="$ROOT/results/k8/dyadic_s2_moe_up_m1_card${CARD}.freq"
GT="/sys/class/drm/card${CARD}/device/tile0/gt0/freq0"
mkdir -p "$ROOT/results/k8"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=dyadic_s2 hail-mary m=1 n=1856 k=2688 lightning_moe_up"
  echo "Planes {0.5,1,2,4} plus residual {1.5,3,6}. vs two-term 15.518 s8 16.060 s8xs4 13.458 W8A8 44.285. STOP if us > 4x W8A8 ~177 with no new trick. P2P off."
  echo "s2xs2 one-plane control. 4-plane E2M1 is unfused sequential 4x this (K6). s2xs4 COMPILE_REFUSED. residual {1.5,3,6} are pairwise sums of {0.5,1,2,4}, not extra unique planes. Never bitcast."
  echo "=== dyadic s2 moe-up m=1 n=1856 k=2688 ==="
  set +e
  "$BIN" --m 1 --n 1856 --k 2688 --warmup 20 --iters 20 2>&1
  echo "dyadic_m1_rc=$?"
  set -e
  echo "=== padM=RC=8 (M=1 not aligned to RC=8) n=1856 k=2688 one s2xs2 plane ==="
  set +e
  pad_out="$("$BIN" --check-m 8 --check-n 16 --check-k 64 --m 8 --n 1856 --k 2688 --warmup 20 --iters 20 2>&1)"
  pad_rc=$?
  set -e
  printf '%s\n' "$pad_out"
  echo "dyadic_padm8_rc=$pad_rc"
  timed_us="$(printf '%s\n' "$pad_out" | awk -F, '/^timed,/{print $5}')"
  timed_ok="$(printf '%s\n' "$pad_out" | awk -F, '/^timed,/{print $8}')"
  timed_abs="$(printf '%s\n' "$pad_out" | awk -F, '/^timed,/{print $7}')"
  if [ -n "${timed_us:-}" ]; then
    python3 - "$timed_us" "$timed_abs" "$timed_ok" <<'PY'
import sys
us = float(sys.argv[1])
mx = int(sys.argv[2])
ok = int(sys.argv[3])
cos = 1.0 if ok == 1 and mx == 0 else 0.0
print("one_plane_event_us=%.3f four_plane_napkin_us=%.3f residual_pairwise_no_extra_unique_planes cosine=%.6f max_abs=%d ok=%d vs_STOP_177 four_vs_stop=%.3f" % (
    us, 4.0 * us, cos, mx, ok, (4.0 * us) / 177.0))
PY
  fi
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
