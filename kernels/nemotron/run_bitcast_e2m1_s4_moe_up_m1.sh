#!/usr/bin/env bash
# K8 hail-mary: bitcast E2M1 onto s4 DPAS at Lightning routed-up M=1.
# Expect cosine death. Explicit negative. Never promote as a floor.
# Usage: gpu-run --card N bash this.sh N
set -euo pipefail
CARD="${1:?card}"
ROOT=/mnt/vm_8tb/github/xe2x2
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi
BIN="$ROOT/kernels/nvfp4/bin/bitcast_e2m1_s4"
OUT="$ROOT/results/k8/bitcast_e2m1_s4_moe_up_m1_card${CARD}.txt"
FREQ="$ROOT/results/k8/bitcast_e2m1_s4_moe_up_m1_card${CARD}.freq"
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
  echo "CONFIG backend=sycl+l0 card=$CARD arm=bitcast_e2m1_s4 hail-mary m=1 n=1856 k=2688 lightning_moe_up"
  echo "Expect cosine death vs E2M1 dequant oracle. Label hail-mary. Never promote as a floor."
  echo "=== bitcast e2m1-as-s4 moe-up m=1 n=1856 k=2688 ==="
  set +e
  "$BIN" --m 1 --n 1856 --k 2688 --warmup 20 --iters 20 2>&1
  echo "bitcast_m1_rc=$?"
  set -e
  echo "=== padM=RC=8 (M=1 not aligned to RC=8) n=1856 k=2688 ==="
  set +e
  "$BIN" --check-m 8 --check-n 16 --check-k 64 --m 8 --n 1856 --k 2688 --warmup 20 --iters 20 2>&1
  echo "bitcast_padm8_rc=$?"
  set -e
  echo "=== host cosine vs E2M1 dequant oracle (same fill as bitcast_e2m1_s4.cpp) ==="
  python3 - <<'PY'
import numpy as np
kMag2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int16)

def e2m1_q(nib):
    q = kMag2[nib & 7]
    return np.where((nib & 8) != 0, -q, q).astype(np.int8)

def bitcast_nibble_s4(nib):
    v = (nib & 15).astype(np.int16)
    return np.where(v >= 8, v - 16, v).astype(np.int8)

def fill_s4(n, seed):
    i = np.arange(n, dtype=np.uint64)
    return (((i * np.uint64(17) + np.uint64(seed)) % np.uint64(16)).astype(np.int16) - 8).astype(np.int8)

def fill_e2m1_bitcast(n, seed):
    i = np.arange(n, dtype=np.uint64)
    nib = ((i * np.uint64(13) + np.uint64(seed)) & np.uint64(15)).astype(np.uint8)
    return e2m1_q(nib), bitcast_nibble_s4(nib)

def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def run(m, n, k, label):
    ha = fill_s4(m * k, 1).reshape(m, k)
    hq, hb = fill_e2m1_bitcast(k * n, 9)
    hq = hq.reshape(k, n)
    hb = hb.reshape(k, n)
    cref = ha.astype(np.int32) @ hq.astype(np.int32)
    cgot = ha.astype(np.int32) @ hb.astype(np.int32)
    mx = int(np.max(np.abs(cgot - cref))) if cgot.size else 0
    cos = cosine(cgot, cref)
    print("host_oracle %s m=%d n=%d k=%d cosine=%.6f max_abs=%d ok=%d" % (
        label, m, n, k, cos, mx, int(mx == 0)))

run(8, 16, 64, "check")
run(8, 1856, 2688, "timed_padM8")
PY
  echo "=== clocks end ==="
  bash "$ROOT/scripts/clocks.sh" "$CARD"
} | tee "$OUT"
echo "wrote $OUT $FREQ"
