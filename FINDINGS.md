# FINDINGS.md -- current evidence ledger

Updated 2026-09-02. This file keeps only findings that should survive the
next kernel, driver, or backend refresh. Commands and raw evidence live
in JOURNAL.md and docs/.

## Host and topology

CONFIG -> Linux 7.1.0-070100, two Intel Arc Pro B70 cards, KMD xe,
  Compute Runtime / NEO 26.22.38646.4.

RESULT -> Kernel 7.1 aligned the GuC firmware requirement and cured the
  former BCS copy-engine / device-lost hardware wedge. The host is
  headless. Neither card is display-held. The two B70s sit on separate
  Intel PCI bridges (09:00.0 and 42:00.0), not as siblings on one
  switch.

VERDICT -> Keep kernel 7.1 as the host baseline. Treat cross-card P2P
  as a measured property of this PCI tree, not a given.

Evidence: docs/HOST.md and b70_ai_things docs/P2P_GPU.md.

## TP=2 software boundary

CONFIG -> Historical vLLM multiprocess TP=2, oneCCL, graph capture,
  and direct P2P experiments on this host (recorded in b70_ai_things).

RESULT -> Raw peer DMA and standalone oneCCL P2P can work. The dangerous
  failure is process / queue handoff: repeated worker-init or graph-capture
  failures can poison later collectives and sometimes both cards.

VERDICT -> Do not enable arbitrary P2P for TP=2. Require per-card health,
  two-rank collective health, matched rank evidence, and teardown
  re-health around every risky TP=2 or PP=2 attempt.

## SYCL adapter (P0)

CONFIG -> host oneAPI 2026.1.1, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`,
  `ZE_AFFINITY_MASK` 0 then 1, `sycl-ls --verbose`. Backend `sycl+l0`.

RESULT -> Live adapter is Intel oneAPI Unified Runtime over Level-Zero
  **V2** on both cards. Architecture name `intel_gpu_bmg_g31`. OpenCL
  is present as a second platform (NEO 26.22.38646.4) and is the
  labeled control. Aspects include `ext_intel_esimd` and
  `ext_intel_matrix`.

VERDICT -> Default kernel path is `sycl+l0` on L0 V2 (immediate lists).
  Do not copy V1 queued-list notes onto this host without a labeled
  control. Flashnext's `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LISTS=0`
  is not the default here.

Evidence: `results/p0/SUMMARY.md`, `docs/HOST.md` P0 freeze.

## Local copy roof (K0)

CONFIG -> backend `sycl+l0`, standalone icpx 2026.1.1 AOT
  `intel_gpu_bmg_g31`, USM memcpy, event profiling, 20 timed iters,
  GT0 cur=2800 MHz, throttle 0, both cards.

RESULT -> 256 MiB D2D is 551-553 GB/s (card0) and 550 GB/s (card1),
  about 90% of the 608 GB/s datasheet. 16 MiB D2D is 622-637 GB/s
  and is *not* the HBM number (working set at/under a cache-sized
  region). 32-128 MiB sit 561-569 GB/s. H2D at 256 MiB is 13.5-14.2
  GB/s. D2H at 256 MiB is 4.66-4.67 GB/s on both cards.

VERDICT -> Quote ~550 GB/s D2D at 256 MiB as the local HBM copy
  floor. Do not quote 16 MiB 622 GB/s as beating 608. Host D2H is
  the slow PCIe direction on this box (~3x slower than H2D).

Evidence: `results/k0/SUMMARY.md`.

## Boring s8 GEMM is not the XMX roof (K0)

CONFIG -> backend `sycl+l0`, 16x16 local-memory s8xs8->s32 tile,
  no DPAS. Prior: XVE-class TOPS, far under 367.

RESULT -> n=2048 square: 9049 us / 1.899 TOPS card0, 9053 us /
  1.898 TOPS card1 (0.52% of 367). Host oracle max_abs=0 at n=64
  on both cards. n=1024 ~2.2 ms / ~0.95-0.98 TOPS. Small n (64,
  256) us disagree across cards.

VERDICT -> This is the XVE floor. A real XMX kernel is the one
  that beats ~9050 us at this shape, not the one that cites 367
  TOPS. K2 DPAS / oneDNN are the next dumps.

Evidence: `results/k0/SUMMARY.md`.

## Decode-shaped s8 GEMV is a millisecond, not a microsecond (K0)

CONFIG -> backend `sycl+l0`, M=1 naive per-column loop, Qwen3.8-ish
  K=5120 N=5120 and N=17408, both cards, GT0 cur=2800 MHz.

RESULT -> 5120x5120: 989-990 us, 26.5 GB/s, 0.053 TOPS, max_abs=0.
  5120x17408: 2235-2241 us, 39.8-39.9 GB/s, 0.080 TOPS, max_abs=0.
  A 550 GB/s copy of the 85 MiB weight would be ~155 us; this kernel
  is ~14x slower than that copy roof.

VERDICT -> Decode GEMV ranks in us. This naive path is 1-2 ms per
  projection and ~5-7% of the local HBM copy floor. Serving-shaped
  kernels have to beat these microseconds, not 367 TOPS. Extra
  launches of this size would dominate a decode step.

Evidence: `results/k0/SUMMARY.md`.

## ESIMD DPAS lights s8, s4, s2, and s2xs8 (K2)

CONFIG -> backend `sycl+l0`, standalone icpx 2026.1.1 AOT
  `intel_gpu_bmg_g31` (issue 21741: not a fat SYCL tree).
  `xmx::dpas` + `lsc_load_2d` Transformed=true. 1024^3, both cards.
  s2 range packed as IGC [-2,1]. Never E2M1 bitcast.

RESULT -> All four arms compile and are host-oracle closed
  (max_abs=0) on both cards, including the literature mix s2xs8
  (arXiv 2508.06753). At matched ~583 MHz: s8 374 us / 5.75 TOPS,
  s4 250 us / 8.57 TOPS (1.49x s8, not 2x), s2xs8 278 us / 7.73 TOPS.
  s2xs2 numeric-closed on both cards but us/TOPS follow clock
  (card1 223 us at 583 MHz, card0 89 us at 1950 MHz on a repeat).
  Versus K0 XVE s8 at the same 1024^3 (~2200 us), DPAS s8 is ~6x
  in wall time. Short kernels do not hold 2800 MHz.

VERDICT -> INT2 silicon exists on these B70s and is usable from
  standalone ESIMD. Matching Intel's 2x s4 story is not required;
  we measured 1.49x at this tile. Do not quote TOPS% of 367 until
  clocks are held. Rank us. Issue 21741 standalone rule stands.

Evidence: `results/k2/SUMMARY.md`.

## Incumbent oneDNN floors (K1)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`, synthetic, gpu-run
  per card. fp8 image
  `b70-local/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50-s01`.
  int8 image `b70-sglang-xpu-int8-runtime:20260826-mtp6`.

RESULT -> `int8_gemm_w8a16` is missing from stock int8 images.
  It exists in `b70-sglang-xpu-int8-w8a16:20260828-2dd55f3` but
  IGC dumps OpenCL `ref_matmul` (scalar mad, no dpas) and M=1
  5120 is 2027 us on both cards. `fp8_gemm_w8a16` M=1 5120 is
  56-58 us. `int8_gemm_w8a8` GEMM-only is 45.0-45.5 us. M=64
  5120: fp8 92-94 us, w8a8 61-62 us. IGC on the fp8 call dumped
  `gemm_zero_fill`, not the GEMM mainloop.

VERDICT -> The live INT8 XMX incumbent is W8A8 at 45 us, not the
  W8A16 symbol. That W8A16 op in the tagged image is a reference
  kernel (~2 ms), 45x slower than W8A8 and 35x slower than FP8
  W8A16. Hand ESIMD must beat 45 us at M=1 5120. Do not read the
  fp8 fill dump as "fp8 has no DPAS."

## fp8_gemm_w8a16 is bf16 DPAS plus E4M3 decompress (K1)

CONFIG -> same fp8 image, `ONEDNN_JIT_DUMP=1`, iga Xe2 disasm of
  ngen `gpu,matmul,jit:gemm:any`. Card0. oneDNN 3.13.0, IGC 2.38.2
  in-container (not host 2.36.3).

RESULT -> Mainloop is `dpas.8x8 (16|M0) rD:f rAcc:f rW:bf rA:bf`.
  Weights: `shl :ub << 8` then `mul :bf * 1.329228e+36` (E4M3
  unpack), then bf16 systolic. `xe_hp_systolic` skipped (M=1
  nocopy; M=64 f8 unsupported). IGC OpenCL dump is only
  `gemm_zero_fill`.

VERDICT -> Xe2 has no native FP8 XMX in this op. The 56 us M=1
  floor is emulated FP8: decompress + bf16 DPAS. A real s8 kernel
  that returns faster than 56 us without that unpack is a beat.
  Confirmed on the timed image; JIT dump is card0.

Evidence: `results/k1/fp8_card0_isa.md`,
  `results/k1/igc_card0_fp8_jit/*.xe2.asm`.

Evidence: `results/k1/SUMMARY.md`.

## Compose-of-s8 did not lose (K3)

CONFIG -> backend `sycl+l0`, standalone ESIMD, same tile as K2.
  Prior: four schoolbook s4 terms cost ~2.7x native s8 if s4 is
  1.49x. Karatsuba three terms ~2.0x. Measure.

RESULT -> Schoolbook `u4_lo+s4_hi` vs native s8 in one binary,
  1024^3, max_abs=0 both cards. Within-run wall time: card0
  207 us compose vs 272 us native (compose faster); card1 112 vs
  115 us (tie). E2M1 two-term `w_lo+8*w_hi` vs s8 LUT also closed
  and faster in-binary (222 vs 374 us card1; 83 vs 94 us card0).
  Karatsuba three s4 DPAS refused: digit sums leave s4.

VERDICT -> The napkin "compose loses" is false on this tile.
  That is not a serving win and not a 367-TOPS claim; clocks still
  swing absolute us. It is enough to keep compose as a real arm,
  especially E2M1 two-term. Do not skip K3 because Intel has
  native s8.

Evidence: `results/k3/SUMMARY.md`.

Open campaign (questions, not findings): docs/KERNEL_CAMPAIGN.md.
Sibling-lab claims to reproduce before they can enter this file.
Now local (K2): s4 DPAS exists but was 1.49x s8 at 1024^3 / ~583 MHz,
not 2x; INT2 DPAS exists (s2xs2 and s2xs8 closed). Remaining:

- ESIMD `dpas<s4,s4>` ~2x s8 MAC rate -- 1.49x at this tile; 2x still open.
- Xe2 has no native FP8 XMX -- reproduced: fp8_gemm_w8a16 is
  E4M3 unpack + bf16 dpas.8x8 (K1 JIT dump).
- NVFP4 E2M1 x2 is exact int8; +-12 overflows s4; bitcast is wrong.
- Load-time s8 NVFP4 spoof fit 8B and not 27B on one 30.3 GiB card.
- `nvfp4_gemm_w4a16` is 4-bit resident decompress, not INT4 XMX.
- M=1 decode is tens to hundreds of times under the compute roof.
- W8A8 decode paid ~160 activation-quant launches that W8A16 skips.
- Transformed LSC VNNI loads were bit-exact; flat prepack was not.
- Push all-reduce is a fabric prototype worth a TP=2 arm.

Until an xe2x2 run with named backend, card pin, compiler identity,
and health repeats a bullet, it stays a hypothesis.

Napkin math: compose-of-s8 loses is now measured false on the K3
tile (see above). Remaining hypotheses: decode cannot use INT2,
PP=2 cannot win decode, we cannot beat XeTLA.
Serving-shaped work ranks by us, not TOPS%. Four B70s are
evidence-gated. Model shelf after the math floor: docs/MODELS.md.

## PP=2

No xe2x2-owned pipeline-parallel finding yet.

## Mixed 2x2

Blocked until TP=2 and PP=2 each have a passing correctness + health
run in this repo.
