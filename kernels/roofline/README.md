# K0 -- roofline floor

Question: what fraction of 608 GB/s and 367 INT8 TOPS can this host
actually reach with boring kernels, on each card, with the current
UMD / IGC?

Open. Do not lock the copy kernel, the GEMM size, or the timer.

## Why

Every later "we lit XMX" or "we are bandwidth bound" claim needs a
local floor. Intel datasheet numbers are not a FINDING until a kernel
here reports percent-of-roof.

## Suggested arms (pick, replace, add)

- Host-to-device, device-to-host, device-to-device copy. Several sizes
  around L1 / LLC / HBM.
- s8x s8 square GEMM at large M=N=K (compute roof candidate).
- M=1 GEMV at Qwen-like K,N (bandwidth roof candidate).
- Optional: bf16 copy, s4 packed copy, E2M1 packed copy (bytes, not
  math).
- Optional: oneDNN GEMM vs ESIMD GEMM as two labeled backends.

Card0 || card1 on independent `gpu-run --card N`. Same binary first
(identity), then split arms across cards and swap.

## Record

Backend, card, IGC/NEO identity, health, warmup discarded, first-run
compile vs steady. us first, then GB/s and TOPS as diagnostics. Do
not quote a single blended score. Serving-shaped arms rank by wall
time.

## First binaries (2026-09-02g)

Standalone `icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31`:

- `copy_roof.cpp` -- USM H2D / D2H / D2D, event-profiled us and GB/s
- `s8_square_gemm.cpp` -- 16x16 local-memory s8xs8->s32 tile (XVE,
  not DPAS). CONFIG prior: will not approach 367 INT8 TOPS.

Host has no g++. Compile in a CPU docker with g++ plus the host
oneAPI 2026.1.1 bind-mount. Run:

```
source scripts/oneapi-env.sh
gpu-run --card N env ZE_AFFINITY_MASK=N ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  kernels/roofline/bin/copy_roof
gpu-run --card N env ZE_AFFINITY_MASK=N ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  kernels/roofline/bin/s8_square_gemm
```

Pair 1: copy on card0, s8 GEMM on card1, then swap.

## Exit

A `results/` JSON (or JOURNAL entry with the JSON path) per arm+card.
Promote a durable percent-of-roof to FINDINGS only after both cards
agree or the disagreement is explained.
