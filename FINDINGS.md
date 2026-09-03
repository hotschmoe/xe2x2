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

## ESIMD DPAS encodings are dpas.8x8 with typed operands (K2)

CONFIG -> same standalone AOT binaries. Runtime
  `IGC_ShaderDumpEnable` on both cards dumped only SIP/caps
  (AOT zebin already in the ELF). Encoding from `ocloc disasm`
  of unbundled `sycl-spir64_gen` zebin, IGA Xe2. Backend
  `sycl+l0`.

RESULT -> Inner loop is `dpas.8x8 (16|M0)` acc `:d` on every arm.
  s8: `rW:b rA:b`. s4: `rW:s4 rA:s4`. s2: `rW:s2 rA:s2`.
  s2xs8: `rW:s2 rA:b` (literature mix). IGA prints s8 as `:b`,
  not `:s8`. `has_dpas: true`, GRF 128. B is `load_block2d.ugm.d8v`.

VERDICT -> s4/s2 are operand types on the same `dpas.8x8`, not a
  different opcode. s2xs8 matches the paper GEN dump. Do not wait
  on runtime IGC dumps of AOT ESIMD; disasm the zebin.

Evidence: `results/k2/igc_isa.md`.

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

Evidence: `results/k1/SUMMARY.md`, `results/k1/fp8_card0_isa.md`.

## W8A8 ngen is s8 DPAS with RC=4 at M=1 and GRF256 at M=64 (K1)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`, sglang int8 mtp6,
  `ONEDNN_JIT_DUMP=1`, IGA Xe2 of ngen bins. Both cards, bins
  md5-identical. oneDNN 3.12.0.

RESULT -> xe_hp_systolic skipped. M=1 5120: 64x `dpas.8x4`
  rW:b rA:b acc:d, wg 8x2, k64, some SLM. M=64: 64x `dpas.8x8`,
  wg 4x2x4, grf256, k64, 53 SLM. M=256: 384x `dpas.8x8`,
  grf256, k128, no SLM.

VERDICT -> The 45 us floor is native s8 XMX. Steal RC=4 for
  decode and GRF256+SLM for M=64. Our hand tile used RC=8 /
  GRF128 / no SLM. Do not copy fp8's bf16 dpas onto this op.

Evidence: `results/k1/int8a8_ngen_isa.md`.

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

## INT8 W8A8 GEMM beats FP8 W8A16 as kernels (K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`. fp8 image r50-s01
  per-tensor scale=1. int8 sglang mtp6 GEMM-only (no act quant).
  Qwen3.8-ish 5120, both cards, cosine vs host ref.

RESULT -> M=1 5120: INT8 42-46 us vs FP8 70-72 us. M=64: INT8
  46-49 us vs FP8 382-430 us. M=1024: INT8 230-256 us (~210-234
  TOPS) vs FP8 1111-1126 us (~48 TOPS). Cosine ~1.0 both arms.

VERDICT -> Unmixed, GEMM-only INT8 W8A8 is the faster kernel at
  decode and prefill shapes. Serving still mixed in ~160 quant
  launches; that is K5. Do not treat tok/s as the kernel result.

Evidence: `results/k4/SUMMARY.md`.

## Naive RMSNorm-epilogue fusion drops one launch (K5)

CONFIG -> backend `sycl+l0`, standalone icpx 2026.1.1 AOT
  `intel_gpu_bmg_g31`, one WI per row. RMSNorm gamma=1 eps=1e-6,
  symmetric s8 qmax=127, per-row absmax. Both cards.

RESULT -> Fused (1 launch) vs rmsnorm-then-quant (2 launches).
  M=1 K=5120: 830 us fused vs 1252-1361 us two-launch, max_abs=1.
  M=1 K=17408: 2820 vs 3769 us. M=64 matches across cards.
  Fusion is ~30-40% faster by dropping a launch. Absolute us is
  still hundreds to thousands on this naive loop.

VERDICT -> Fusion removes N=1 launch and is closed to max_abs<=1
  vs the two-kernel path. It is not a serving epilogue: 830 us
  already dwarfs the 45 us W8A8 GEMM. A bandwidth-capable
  producer epilogue is the next K5 arm, not this us as a floor
  to cite in a serve.

Evidence: `results/k5/SUMMARY.md`.

## WG-256 fused RMSNorm-quant is tens of us, not 830 (K5)

CONFIG -> same contract as the naive K5 micro, backend `sycl+l0`,
  one work-group per row, WG=256, `reduce_over_group`. Both cards,
  GT0 cur=2800 MHz.

RESULT -> Fused M=1 K=5120: 36 us (card0 first), 13 us (card0
  repeat), 7 us (card1). All max_abs<=1. Naive fused was 830 us
  on both cards. Two-launch WG path is ~1.5x the fused us.
  Short kernels still swing; M=64 K=17408 is ~20-38 us.

VERDICT -> The 830 us was the 1-WI loop, not RMSNorm+quant as
  an op. A producer epilogue at ~7-36 us sits next to the 45 us
  W8A8 GEMM, not 18x above it. Do not freeze 7 us. 160 extra
  launches would still lose; 1 fused write per residual is the
  remaining launch question.

Evidence: `results/k5/SUMMARY.md`.

## NVFP4 nibble LUT to s8 DPAS is numerically closed (K6)

CONFIG -> backend `sycl+l0`, standalone ESIMD, same s8 DPAS tile
  as K2. Packed E2M1 nibbles in HBM, LUT to s8 `{0,+-1,+-2,+-3,
  +-4,+-6,+-8,+-12}`. Never bitcast onto s4. Both cards, 1024^3.

RESULT -> Host LUT + s8 DPAS and device unpack + s8 DPAS both
  max_abs=0. Unpack tax ~12% of the s8 DPAS (33/271 us card0 at
  550 MHz; 9/75 us card1). Absolute us follows clocks.

VERDICT -> The nibble-LUT spoof is a real arm, not a bitcast.
  Two-launch unpack is a small tax at this tile. Do not quote a
  single us without the clock column.

Evidence: `results/k6/SUMMARY.md`.

## In-register E2M1 LUT lights VNNI4 and loses in us (K6)

CONFIG -> backend `sycl+l0`, packed E2M1 load, GRF nibble LUT,
  then s8 DPAS. Packs tried: raw, VNNI4 (4 along K), k-major.
  Never bitcast s4. Both cards, 1024^3, GT0 cur=2800.

RESULT -> VNNI4 max_abs=0 on the 8x16x32 check and 1024^3.
  Raw 6959 / 124224. K-major 8240 / 107824. Timed VNNI4
  2316-2317 us both cards vs two-launch unpack+DPAS 84-305 us.

VERDICT -> VNNI4 is the s8 B pack that matches Transformed LSC
  on these cards (host-prepack landmine was a different layout).
  This scalar LUT in the DPAS loop is not a latency win. Keep
  two-launch unpack as the fast closed spoof; vectorize the
  LUT before claiming an in-register beat.

Evidence: `results/k6/SUMMARY.md`.

## Serving-shaped NVFP4 nibble LUT is 158 us at 2800 (K6)

CONFIG -> backend `sycl+l0`, standalone `nibble_lut_sc`.
  Packed E2M1 B in HBM (2/byte along K), simd nibble LUT,
  VNNI4, then the K2 RC=4 8x2-N s8 scale-to-f16 tile.
  Never bitcast onto s4. Both cards, NT=2, spin=4000.
  M=1 and M=4, N=K=5120. Prior: s8 34 us, W8A8 44 us,
  scalar LUT 2316 us.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 158.17/158.18 vs s8 34 vs
  W8A8 44. M=4 tracks. Packed-B 83 GB/s. Spread ~0.004%.

VERDICT -> First serving-shaped NVFP4 in-register spoof
  is numerically closed. 4-bit B stays in HBM. Not a us
  beat of s8 (~4.65x) or W8A8. LUT tax, not HBM (83 vs
  608 GB/s). Closed-form decode later cut this to 134.8
  us (03ad). "NVFP4 cannot feed XMX" is false; "as fast
  as s8" is false. Rank us.

Evidence: `results/k6/sc_n2_s4000_card0.txt`,
  `results/k6/sc_n2_s4000_card1.txt`.

## 16-entry iselect table LUT loses to merge LUT (K6)

CONFIG -> backend `sycl+l0`, standalone `nibble_lut_sct`.
  Same packed E2M1 RC=4 8x2-N tile as `nibble_lut_sc`,
  but decode is a 16-entry GRF table + `iselect`.
  Never bitcast s4. Both cards, NT=2, spin=4000.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 1021.73/1021.88 vs merge
  LUT 158 vs s8 34. M=4 tracks. Spread ~0.01%.

VERDICT -> iselect table is ~6.46x the merge LUT.
  Numeric closed, us lost. Stop GRF gather tables
  on this tile. Keep the merge-chain simd LUT.

Evidence: `results/k6/sct_n2_s4000_card0.txt`,
  `results/k6/sct_n2_s4000_card1.txt`.

## Two-launch scalar unpack loses to 158 us LUT (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_unpack_sc`. Packed E2M1 unpack to s8
  each iter (1 byte / WI), then Transformed s8
  GEMM on the RC=4 8x2-N tile. Never bitcast s4.
  Both cards, NT=2, spin=4000.

RESULT -> cosine=1.0 max_abs=0. timed cur=2800
  throttle=1. M=1 two-launch pipe_host
  266.10/263.31 vs fused LUT 158 vs s8ctrl
  34.55/35.24. M=4 tracks. Spread ~1%.

VERDICT -> Naive two-launch unpack is ~1.67x
  the 158 us in-register LUT. s8ctrl matches
  the 34 us s8 tile, so the tax is unpack, not
  DPAS. throttle=1 is part of this control.
  Vectorized unpack is a loss (314.7 us
  card0). Rank pipe.

Evidence: `results/k6/unpack_n2_s4000_card0.txt`,
  `results/k6/unpack_n2_s4000_card1.txt`.

## One packed load per k64 loses to two k32 loads (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_lut_sck`. Same merge LUT as
  `nibble_lut_sc`, one height-32 packed load
  per k64. Never bitcast s4. Both cards, NT=2,
  spin=4000.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1 pipe_host
  169.02/169.14 vs two-k32 158. M=4 tracks.
  Spread ~0.07%.

VERDICT -> Combining the two packed loads is a
  small loss (~1.07x). Keep two k32 loads.
  Floor stays 158 us.

Evidence: `results/k6/sck_n2_s4000_card0.txt`,
  `results/k6/sck_n2_s4000_card1.txt`.

## Vectorized two-launch unpack loses to scalar unpack (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_unpack_scv`. Same two-launch as
  `nibble_unpack_sc`, ESIMD 16-wide simd nibble
  decode. Both cards, NT=2, spin=4000.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1 pipe_host
  314.72/314.44 vs scalar 265 vs fused LUT
  158 vs s8ctrl 34.3/34.1. M=4 tracks.
  Spread ~0.09%.

VERDICT -> Vectorizing the unpack kernel is a
  loss (~1.19x scalar, ~2.0x fused LUT) both
  cards. Stop this unpack path. Keep fused
  158 us LUT as the NVFP4 s8-A spoof floor.

Evidence: `results/k6/unpackv_n2_s4000_card0.txt`,
  `results/k6/unpackv_n2_s4000_card1.txt`.

## E2M1 two-term s4 decode is 28.5 us at 2800 (K3/K6)

CONFIG -> backend `sycl+l0`, standalone
  `compose_e2m1_sc`. RC=4 8x2-N tile, A is s4,
  B is E2M1 split to two s4 planes, acc =
  acc_lo + 8*acc_hi. Never bitcast. Both cards,
  NT=2, spin=4000. Prior: 2x s4 16.5 ~33 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1 pipe_host
  28.52/28.54 vs s4 16.5 vs s8 34 vs W8A8 44
  vs LUT 158. M=4 tracks. Spread ~0.08%.

VERDICT -> New overflow-split E2M1 floor 28.5
  us at 2800 both cards. ~1.73x native s4,
  under s8 and W8A8. A is s4, not the s8-A
  LUT contract. Rank us.

Evidence: `results/k6/e2m1sc_n2_s4000_card0.txt`,
  `results/k6/e2m1sc_n2_s4000_card1.txt`.

## E2M1 two-term N=17408 is 103.5 us at 2800 (K3/K6)

CONFIG -> backend `sycl+l0`, standalone
  `compose_e2m1_sc`. Same tile, M=1 N=17408
  K=5120. Both cards, NT=2, spin=4000.
  Prior: N-linear ~97 us; s4 29.5.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1 pipe_host
  102.73/104.35 vs 5120 28.5 vs s4 29.5 vs
  s8 141.6. M=4 tracks. Spread ~1.6%.

VERDICT -> New E2M1 two-term wide-N floor
  103.5 us at 2800 both cards. ~3.63x square,
  near linear, not s4's 1.80x. Native s4 29.5
  still wins this shape. Rank us.

Evidence: `results/k6/e2m1sc_n17408_n2_s4000_card0.txt`,
  `results/k6/e2m1sc_n17408_n2_s4000_card1.txt`.

## E2M1 two-term K=17408 is 193.6 us at 2800 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_sc`. M=1 N=5120 K=17408.
  Both cards, NT=2, spin=4000. Prior:
  K-linear ~97 us; s4 53.4.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1 pipe_host
  194.02/193.10 vs 5120 28.5 vs N=17408
  103.5 vs s4 53.4 vs W8A8 155.3. M=4
  tracks. Spread ~0.5%.

VERDICT -> New E2M1 two-term wide-K floor
  193.6 us at 2800 both cards. ~6.79x square,
  K-hostile. Native s4 53.4 and oneDNN W8A8
  155 both beat this at FFN-down. Qwen FFN
  compose decode map is closed. Rank us.

Evidence: `results/k6/e2m1sc_k17408_n2_s4000_card0.txt`,
  `results/k6/e2m1sc_k17408_n2_s4000_card1.txt`.

## E2M1 two-term 8x2-N loses at M=64 (K3/K6)

CONFIG -> backend `sycl+l0`, standalone
  `compose_e2m1_sc`. Same RC=4 8x2-N tile,
  M=64 N=K=5120. Card0 only, NT=2, spin=512.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  217.92 vs M=1 28.5 vs s4 4x8 33.6 vs s8
  75 vs W8A8 46.

VERDICT -> Decode-tile compose is ~6.5x s4
  4x8 at M=64. Not a prefill floor. One-card.

Evidence: `results/k6/e2m1sc_m64_n2_s512_card0.txt`.

## E2M1 two-term 8x2-N loses at M=256 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_sc`. M=256 N=K=5120. Both
  cards, NT=2, spin=512.

RESULT -> cosine=1.0 max_abs=0. timed
  act=2667/2683 cur=2800 throttle=1. M=256
  pipe_host 612.68/601.18 vs M=1 28.5 vs s4
  4-acc 48.6 vs s8 128 vs W8A8 75. Spread
  ~1.9%.

VERDICT -> ~12.4x s4 4-acc, ~8.0x W8A8.
  throttle=1 both cards. Stop 8x2-N compose
  at prefill. Keep 28.5 us as decode-only.

Evidence: `results/k6/e2m1sc_m256_n2_s512_card0.txt`,
  `results/k6/e2m1sc_m256_n2_s512_card1.txt`.

## NVFP4 merge LUT loses at M=64 on 8x2-N (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_lut_sc`. Packed E2M1, simd LUT,
  VNNI4, s8 DPAS. M=64 N=K=5120. Both
  cards, NT=2, spin=512. Never bitcast.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  665.56/645.63 vs M=1 158 vs s8 4x8 75
  vs W8A8 46. Spread ~3.1%.

VERDICT -> Decode-tile LUT is ~8.6x s8 4x8
  at M=64 both cards. Not a prefill floor.
  Stop 8x2-N LUT at prefill.

Evidence: `results/k6/sc_m64_n2_s512_card0.txt`,
  `results/k6/sc_m64_n2_s512_card1.txt`.

## E2M1 two-term 4x8 A-db is 68.7 us at M=64 (K3/K6)

CONFIG -> backend `sycl+l0`, standalone
  `compose_e2m1_db48`. RC=8 wg 4x8 A-db,
  two s4 DPAS per k64, acc_lo+8*acc_hi.
  M=64 N=K=5120. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 2x s4 ~67 us.

RESULT -> ocloc 64x `dpas.8x8` rW:s4 rA:s4,
  grf 128, no SLM. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0. M=64
  pipe_host 68.73/68.68 vs 8x2-N 217.9 vs
  s4 33.6 vs s8 75 vs W8A8 46. Spread
  ~0.07%.

VERDICT -> New E2M1 two-term 4x8 A-db floor
  68.7 us at 2800 both cards. ~3.17x the
  decode tile and ~2.04x native s4. Beats
  s8 75, loses to W8A8 46. A is s4. Rank us.

Evidence: `results/k6/e2m1db48_m64_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m64_n2_s512_card1.txt`,
  `results/k6/e2m1db48_dpas_lines.txt`.

## E2M1 two-term 4x8 A-db at M=256 is 194.9 us (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_db48`. M=256 N=K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=256 pipe_host
  194.82/195.03 vs M=64 68.7 vs 8x2-N 607
  vs s4 4-acc 48.6 vs s8 128 vs W8A8 75.
  Spread ~0.1%.

VERDICT -> New E2M1 two-term 4x8 A-db
  M=256 floor 194.9 us at 2800 both cards.
  ~3.12x 8x2-N 607, ~4.0x native s4 4-acc.
  Beats s8 128, loses to W8A8 75. A is s4.
  Rank us.

Evidence: `results/k6/e2m1db48_m256_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m256_n2_s512_card1.txt`.

## E2M1 two-term 4-acc loses at M=256 (K3/K6)

CONFIG -> backend `sycl+l0`, standalone
  `compose_e2m1_w48m4`. RC=8 4-acc wg 4x8
  k128, two s4 DPAS per k64. M=256 N=K=5120.
  Card0 only, NT=2, spin=512. Never
  bitcast. Prior: 2x s4 ~97 us.

RESULT -> ocloc 256x `dpas.8x8` rW:s4 rA:s4,
  grf 128, no SLM. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0. M=256
  pipe_host 411.303 vs 4x8 compose 194.9
  vs s4 48.6 vs s8 128 vs W8A8 75.

VERDICT -> 4-acc compose is ~2.11x the 4x8
  A-db compose tile and ~8.46x native s4.
  Napkin 2x died. One-card. Do not freeze
  411 us until card1.

Evidence: `results/k6/e2m1w48m4_m256_n2_s512_card0.txt`,
  `results/k6/e2m1w48m4_dpas_lines.txt`.

## NVFP4 merge LUT on 4x8 A-db is 392.4 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_lut_db48`. Packed E2M1, simd LUT,
  VNNI4, s8 DPAS on RC=8 wg 4x8 A-db.
  M=64 N=K=5120. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 4.65x * 75 ~349 us.

RESULT -> ocloc 64x `dpas.8x8` rW:b rA:b,
  packed B not Transformed, grf 128, no SLM.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 392.43/392.44
  vs 8x2-N 656 vs s8 75 vs W8A8 46 vs
  compose 68.7. Spread ~0.004%.

VERDICT -> New 4x8 A-db LUT floor 392.4 us
  at 2800 both cards. ~1.67x the decode
  tile. Still ~5.23x s8 75. Packed E2M1
  stays in HBM. Rank us.

Evidence: `results/k6/lutdb48_m64_n2_s512_card0.txt`,
  `results/k6/lutdb48_m64_n2_s512_card1.txt`,
  `results/k6/lutdb48_dpas_lines.txt`.

## NVFP4 merge LUT 4x8 A-db N=17408 is 1032 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_db48`. M=64 N=17408 K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  1027.42/1037.01 vs 5120 392.4 vs s8
  338.9 vs s4 94.7 vs compose 326.9.
  Spread ~0.9%.

VERDICT -> New 4x8 A-db LUT wide-N floor
  1032 us at 2800 both cards. ~2.63x
  square, ~3.05x s8 338.9. Rank us.

Evidence: `results/k6/lutdb48_m64_n17408_n2_s512_card0.txt`,
  `results/k6/lutdb48_m64_n17408_n2_s512_card1.txt`.

## NVFP4 merge LUT 4x8 A-db K=17408 is 1333 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_db48`. M=64 N=5120 K=17408.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: K-linear ~1334 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  1332.41/1332.67 vs 5120 392.4 vs N=17408
  1032 vs s8 374.7 vs s4 106.0 vs compose
  403.4. Spread ~0.02%.

VERDICT -> New 4x8 A-db LUT wide-K floor
  1333 us at 2800 both cards. K-linear
  ~3.40x, ~3.56x s8 374.7. Qwen FFN LUT
  M=64 map is closed. Rank us.

Evidence: `results/k6/lutdb48_m64_k17408_n2_s512_card0.txt`,
  `results/k6/lutdb48_m64_k17408_n2_s512_card1.txt`.

## NVFP4 merge LUT 4x8 A-db is 1203 us at M=256 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_db48`. M=256 N=K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: M-linear ~1570 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=256 pipe_host
  1198.44/1207.28 vs M=64 392.4 vs s8 128
  vs compose 194.9 vs W8A8 75. Spread ~0.7%.

VERDICT -> New 4x8 A-db LUT M=256 floor
  1203 us at 2800 both cards. ~3.07x M=64,
  ~9.4x s8 128. Rank us.

Evidence: `results/k6/lutdb48_m256_n2_s512_card0.txt`,
  `results/k6/lutdb48_m256_n2_s512_card1.txt`.

## Closed-form LUT on 4x8 A-db is 331.6 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, standalone
  `nibble_lut_scf_db48`. Packed E2M1,
  exp/mant shift, VNNI4, s8 DPAS on RC=8
  wg 4x8 A-db. M=64 N=K=5120. Both cards,
  NT=2, spin=512. Never bitcast. Prior:
  392.4*134.8/158 ~335 us.

RESULT -> ocloc 64x `dpas.8x8` rW:b rA:b,
  grf 128, no SLM. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0. M=64
  pipe_host 331.665/331.554 vs merge 392.4
  vs scf decode 134.8 vs s8 75 vs W8A8 46.
  Spread ~0.03%.

VERDICT -> New 4x8 A-db closed-form LUT
  floor 331.6 us at 2800 both cards.
  ~1.18x merge LUT, napkin held. Still
  ~4.42x s8 75. Rank us.

Evidence: `results/k6/lutscfdb48_m64_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m64_n2_s512_card1.txt`,
  `results/k6/lutscfdb48_dpas_lines.txt`.

## Closed-form LUT on 4x8 A-db is 1083 us at M=256 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_scf_db48`. M=256 N=K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: 331.6*3.08 ~1021 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=256 pipe_host
  1077.15/1089.13 vs M=64 331.6 vs merge
  1203 vs s8 128 vs compose 194.9 vs W8A8
  75. Spread ~1.1%.

VERDICT -> New 4x8 A-db closed-form LUT
  M=256 floor 1083 us at 2800 both cards.
  ~3.27x M=64, ~1.11x merge 1203, ~8.46x
  s8 128. Rank us.

Evidence: `results/k6/lutscfdb48_m256_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m256_n2_s512_card1.txt`.

## Closed-form LUT 4x8 A-db N=17408 is 880 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_scf_db48`. M=64 N=17408
  K=5120. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 331.6*1032/392.4
  ~872 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=2767/2783 cur=2800 throttle=1.
  M=64 pipe_host 882.54/877.32 vs 5120
  331.6 vs merge 1032 vs s8 338.9 vs s4
  94.7 vs compose 326.9 vs napkin 872.
  Spread ~0.6%.

VERDICT -> New 4x8 A-db closed-form LUT
  wide-N floor 880 us at ~2770/2800 both
  cards, throttle=1. ~2.65x square,
  ~1.17x merge 1032, ~2.60x s8 338.9.
  Napkin held. Rank us.

Evidence: `results/k6/lutscfdb48_m64_n17408_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m64_n17408_n2_s512_card1.txt`.

## Closed-form LUT 4x8 A-db K=17408 is 1125 us at M=64 (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_scf_db48`. M=64 N=5120
  K=17408. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 331.6*1333/392.4
  ~1127 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  1125.90/1123.39 vs 5120 331.6 vs N=17408
  880 vs merge 1333 vs s8 374.7 vs s4
  106.0 vs compose 403.4 vs napkin 1127.
  Spread ~0.2%.

VERDICT -> New 4x8 A-db closed-form LUT
  wide-K floor 1125 us at 2800 both cards.
  K-linear ~3.39x, ~1.18x merge 1333,
  ~3.00x s8 374.7. Qwen FFN closed-form
  LUT M=64 map is closed. Rank us.

Evidence: `results/k6/lutscfdb48_m64_k17408_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m64_k17408_n2_s512_card1.txt`.

## Closed-form LUT 4x8 A-db M=256 N=17408 is 3138 us (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_scf_db48`. M=256 N=17408
  K=5120. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 1083*880/331.6
  ~2874 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=2650/2683 cur=2800 throttle=1.
  M=256 pipe_host 3163.04/3113.86 vs 5120
  1083 vs M=64 N=17408 880 vs s8 469.8
  vs s4 140 vs compose 984.3 vs napkin
  2874. Spread ~1.6%.

VERDICT -> New 4x8 A-db closed-form LUT
  M=256 wide-N floor 3138 us at ~2660/2800
  both cards, throttle=1. ~2.90x square,
  ~6.68x s8 469.8. Rank us.

Evidence: `results/k6/lutscfdb48_m256_n17408_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m256_n17408_n2_s512_card1.txt`.

## Closed-form LUT 4x8 A-db M=256 K=17408 is 3428 us (K6)

CONFIG -> backend `sycl+l0`, same
  `nibble_lut_scf_db48`. M=256 N=5120
  K=17408. Both cards, NT=2, spin=512.
  Never bitcast. Prior: 1083*1125/331.6
  ~3675 us.

RESULT -> cosine=1.0 max_abs=0. timed
  act=2750/2783 cur=2800 throttle=1.
  M=256 pipe_host 3444.61/3412.24 vs 5120
  1083 vs M=64 K=17408 1125 vs s8 477.4
  vs s4 149 vs compose 968.7 vs napkin
  3675. Spread ~0.9%.

VERDICT -> New 4x8 A-db closed-form LUT
  M=256 wide-K floor 3428 us at ~2760/2800
  both cards, throttle=1. ~3.17x square,
  ~7.18x s8 477.4. Qwen FFN closed-form
  LUT M=256 map is closed. 4x8 LUT loses
  to s8/s4/compose at FFN prefill. Rank us.

Evidence: `results/k6/lutscfdb48_m256_k17408_n2_s512_card0.txt`,
  `results/k6/lutscfdb48_m256_k17408_n2_s512_card1.txt`.

## E2M1 two-term 4x8 A-db N=17408 is 326.9 us at M=64 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_db48`. M=64 N=17408 K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: N-linear ~233 us; s4 94.7.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  328.03/325.80 vs 5120 68.7 vs s4 94.7 vs
  s8 338.9. Spread ~0.7%.

VERDICT -> New E2M1 two-term 4x8 wide-N
  floor 326.9 us at 2800 both cards. ~4.76x
  square, ~3.45x native s4 94.7. More
  N-hostile than s4's 2.81x. Barely under
  s8 338.9. Rank us.

Evidence: `results/k6/e2m1db48_m64_n17408_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m64_n17408_n2_s512_card1.txt`.

## E2M1 two-term 4x8 A-db K=17408 is 403.4 us at M=64 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_db48`. M=64 N=5120 K=17408.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: K-linear ~233 us; s4 106.0.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64 pipe_host
  403.60/403.19 vs 5120 68.7 vs N=17408
  326.9 vs s4 106.0 vs s8 374.7. Spread
  ~0.1%.

VERDICT -> New E2M1 two-term 4x8 wide-K
  floor 403.4 us at 2800 both cards. ~5.87x
  square, ~3.81x native s4 106, and loses
  to s8 374.7. Qwen FFN compose M=64 map
  is closed. Rank us.

Evidence: `results/k6/e2m1db48_m64_k17408_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m64_k17408_n2_s512_card1.txt`.

## E2M1 two-term 4x8 A-db N=17408 is 984.3 us at M=256 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_db48`. M=256 N=17408 K=5120.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: s4 140.0; s8 469.8.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=256 pipe_host
  985.64/982.88 vs 5120 194.9 vs s4 140.0
  vs s8 469.8 vs M=64 326.9. Spread ~0.3%.

VERDICT -> New E2M1 two-term 4x8 M=256
  wide-N floor 984.3 us at 2800 both cards.
  ~5.05x square, ~7.03x native s4 140,
  ~2.10x s8 469.8. throttle=0. Rank us.

Evidence: `results/k6/e2m1db48_m256_n17408_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m256_n17408_n2_s512_card1.txt`.

## E2M1 two-term 4x8 A-db K=17408 is 968.7 us at M=256 (K3/K6)

CONFIG -> backend `sycl+l0`, same
  `compose_e2m1_db48`. M=256 N=5120 K=17408.
  Both cards, NT=2, spin=512. Never
  bitcast. Prior: s4 149.0; s8 477.4.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=256 pipe_host
  964.29/973.11 vs 5120 194.9 vs N=17408
  984.3 vs s4 149.0 vs s8 477.4 vs M=64
  403.4. Spread ~0.9%.

VERDICT -> New E2M1 two-term 4x8 M=256
  wide-K floor 968.7 us at 2800 both cards.
  ~4.97x square, ~6.50x native s4 149,
  ~2.03x s8 477.4. Qwen FFN compose M=256
  map is closed. Rank us.

Evidence: `results/k6/e2m1db48_m256_k17408_n2_s512_card0.txt`,
  `results/k6/e2m1db48_m256_k17408_n2_s512_card1.txt`.

## Closed-form E2M1 nibble decode is 134.8 us (K6)

CONFIG -> backend `sycl+l0`, standalone `nibble_lut_scf`.
  Same packed-E2M1 RC=4 8x2-N s8 scale-to-f16 tile as
  `nibble_lut_sc`, but mag = (e==0)? m : (2+m)<<(e-1)
  instead of the 3-merge LUT. Never bitcast s4. Both
  cards, NT=2, spin=4000, M=1 5120. Prior: merge 158.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. pipe_host 134.756/134.783 vs merge 158
  vs s8 34 vs W8A8 44. Packed-B 97.5 GB/s. Spread
  ~0.02%.

VERDICT -> New Family-A s8-A spoof floor 134.8 us
  at 2800 both cards. ~1.17x the merge LUT. Still
  ~4.0x s8 and ~3.06x W8A8. Keep packed E2M1 in
  HBM. Rank us.

Evidence: `results/k6/scf_n2_s4000_card0.txt`,
  `results/k6/scf_n2_s4000_card1.txt`.

## E2M1 bitcast onto s4 DPAS is an explicit negative (K6)

CONFIG -> backend `sycl+l0`, standalone `bitcast_e2m1_s4`.
  Feed E2M1 nibbles as s4 two's complement into
  `dpas<s4,s4>`. Oracle is E2M1 q = {0,+-1,+-2,+-3,
  +-4,+-6,+-8,+-12}. Both cards.

RESULT -> check 8x16x64 max_abs=352 ok=0. timed
  256^3 max_abs=1408 ok=0. Both cards agree.

VERDICT -> Do not bitcast NVFP4 E2M1 onto s4 DPAS.
  +-12 is outside s4 [-8,7]. Compile was not the
  failure; the numeric is.

Evidence: `results/k6/sprint_mix_lo_bitcast_card0.txt`,
  `results/k6/sprint_lo_bitcast_card1.txt`.

## Sparse-hi / lo-only compose is cheap and wrong (K6)

CONFIG -> backend `sycl+l0`, `compose_e2m1_loonly`
  (drop hi s4 DPAS). Both cards, NT=2, spin=4000,
  M=1 5120. Real checkpoint hist on qwen3.8-27b
  nvfp4-radixark FFN U8 weights.

RESULT -> 8 FFN tensors (layers 0,1,10 gate/up/down),
  89e6 nibbles each: ov_frac 0.2464-0.2505, zeros
  ~3.45% per sign, nearly uniform. lo-only pipe
  16.342/16.352 us at held 2800, cosine=0.760548
  max_abs=8.1953 ok=0.

VERDICT -> Hi plane is dense ~25% on this ckpt, same
  as uniform 4/16. Skip-hi is s4-cheap and E2M1-wrong.
  Stop sparse correction on codes 8 and 12.

Evidence: `results/k6/hist_nvfp4.txt`,
  `results/k6/sprint_mix_lo_bitcast_card0.txt`,
  `results/k6/sprint_lo_bitcast_card1.txt`.

## Mixed s8xs4 DPAS lights; s2xs4 does not compile (K6)

CONFIG -> backend `sycl+l0`, `sprint_dpas_mix` check
  tile 8x16x32. Both cards. `g16_k16_dpas` K=16
  probe. `dyadic_s2` s2xs2 control.

RESULT -> MIX_OK s8A_s4B and s4A_s8B both cards
  (runtime; no host s32 oracle this sprint).
  s2xs4 COMPILE_REFUSED (`A bits 8*8*4=256` vs
  operand 128). s8 `dpas<4,4>` K=16 COMPILE_REFUSED
  (`Systolic depth must be equal to 8`). dyadic_s2
  256^3 max_abs=0 ok=1 card1.

VERDICT -> Mixed s8/s4 DPAS exists on this IGC.
  Mixed s2/s4 does not compile. Hand s8 DPAS cannot
  isolate NVFP4 group-16: depth is fixed at 8, so
  s8 K is 32 = two groups.

Evidence: `results/k6/sprint_mix_prod_card1.txt`,
  `results/k6/sprint_mix_lo_bitcast_card0.txt`,
  `results/k6/g16_k16_dpas.compile.log`,
  `results/k6/g16_scale_landmine.txt`.

## Mixed s8xs4 is host-s32 closed both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8xs4`. AOT `intel_gpu_bmg_g31`.
  K=32 (OPC=4), s4 [-8,7], Transformed
  B. Host unpacked s8*s4 s32. Both
  cards. Never E2M1 bitcast.

RESULT -> max_abs=0 ok=1 on s8A_s4B
  and s4A_s8B at 8x16x32, 32x32x128,
  and 256^3 both cards. bin_rc=0.
  Timed 256^3 clocks not held.

VERDICT -> Mixed s8/s4 is numerically
  the integer product, not a dummy
  MIX_OK. K=32 mix, not s4xs4 K=64.
  Both cards. Do not quote 256^3 us.

Evidence: `results/k2/s8xs4_oracle_card0.txt`,
  `results/k2/s8xs4_oracle_card1.txt`.

## GPTQ INT4 codes feed ESIMD s4 both cards (K6)

CONFIG -> backend `sycl+l0`, Qwen3.8-27B
  `gptq-int4-mtp-bf16-9d189a60` g128
  `sym=true` pack i32. CPU unpack
  LSB-first along K, s4 = nibble-8.
  `dpas_s4_ckpt` synthetic s4 A, real
  B. Both cards. Never E2M1 bitcast.

RESULT -> 6/6 layer0/1 FFN tensors both
  cards: s4 in [-8,7], s4_ov=0,
  g_idx = i/128. Stored qzeros all 7.
  down_proj 256x256 dump. check 8x16x64
  and tile 8x256x256 max_abs=0 ok=1
  both cards.

VERDICT -> This GPTQ INT4 checkpoint
  is integer s4, and ESIMD `dpas<s4,s4>`
  matches host s32 on the dumped tile
  both cards. Stored zp nibble is 7,
  not 8. Group scales not applied.

Evidence: `results/k6/gptq_s4_card0.txt`,
  `results/k6/gptq_s4_card1.txt`.

## ESIMD s8xs4 decode is 22.1 us both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8xs4_sc`. RC=4 NT=2 unroll=16
  packB=2 A=s8 B=s4 K=32 dpas. M=1 and
  M=4 5120. Both cards. Named clock
  2800. Never E2M1 bitcast.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1
  pipe_host 21.961/22.149 vs s2xs8 14.1
  vs s4 16.5 vs s8 34 vs W8A8 44. M=4
  tracks. Spread ~0.9%.

VERDICT -> New s8xs4 decode floor 22.1
  us pipe_host at 2800 both cards.
  Beats s8 34 (~1.53x), loses to s4
  16.5 and s2xs8 14.1. Numeric closed.
  Rank pipe_host.

Evidence: `results/k2/s8xs4sc_n2_s4000_card0.txt`,
  `results/k2/s8xs4sc_n2_s4000_card1.txt`.

## GPTQ s4 group-scale f16 is closed both cards (K6)

CONFIG -> backend `sycl+l0`, `dpas_s4_gptq`.
  Real GPTQ s4 B + g128 f16 scales from
  Qwen3.8-27B down_proj 256x256.
  Synthetic s4 A * 0.02. Partial s32
  per group then * scale. Both cards.

RESULT -> scales 0.0028-0.0102. check
  8x16x128 cosine=1.0 max_abs=7.6e-6
  ok=1 both. tile 8x256x256 cosine=1.0
  max_abs=0 ok=1 both.

VERDICT -> Group-scale epilogue matches
  host s32*scale both cards. Integer
  GPTQ path includes the f16 scales.
  Do not rank us.

Evidence: `results/k6/gptq_s4_sc_card0.txt`,
  `results/k6/gptq_s4_sc_card1.txt`.

## ESIMD s8xs4 N=17408 is 38.6 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_sc`. M=1 N=17408 K=5120.
  Both cards. Named clock 2800. NT=2
  spin=4000. Prior: square 22.1;
  N-linear ~75; s4 29.5; s8 141.6.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1
  pipe_host 38.637/38.554 vs square
  22.1 vs s4 29.5 vs s8 141.6 vs W8A8
  158.1 vs napkin 75. M=4 tracks.
  Spread ~0.2%. ~1.74x square.

VERDICT -> New s8xs4 N=17408 floor
  38.6 us pipe_host at 2800 both cards.
  Under linear like s4 (1.80x), not s8
  (4.16x). Beats s8 and W8A8, loses to
  s4 (~1.31x). Rank pipe_host.

Evidence: `results/k2/s8xs4sc_n17408_n2_s4000_card0.txt`,
  `results/k2/s8xs4sc_n17408_n2_s4000_card1.txt`.

## ESIMD s8xs4 K=17408 is 73.2 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_sc`. M=1 N=5120 K=17408.
  Both cards. Named clock 2800. NT=2
  spin=4000. Prior: square 22.1;
  N-wide 38.6; K-linear ~75; s4 53.4;
  s8 261.6.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=1
  pipe_host 73.188/73.172 vs square
  22.1 vs N-wide 38.6 vs s4 53.4 vs
  s8 261.6 vs W8A8 155.3 vs napkin 75.
  M=4 tracks. Spread ~0.02%. ~3.31x
  square.

VERDICT -> New s8xs4 K=17408 floor
  73.2 us pipe_host at 2800 both cards.
  Near K-linear. Qwen FFN s8xs4 decode
  map is closed (22.1 / 38.6 / 73.2).
  Beats s8 and W8A8, loses to s4.
  Rank pipe_host.

Evidence: `results/k2/s8xs4sc_k17408_n2_s4000_card0.txt`,
  `results/k2/s8xs4sc_k17408_n2_s4000_card1.txt`.

## GPTQ s4 RC=4 decode is 29.9 us both cards (K6)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s4_gptq_sc`. RC=4 NT=2 gs=128.
  Real GPTQ down_proj 5120x5120 s4 +
  g128 f16 scales. M=1 and M=4 5120.
  Both cards. Named clock 2800. Never
  E2M1.

RESULT -> cosine=1.0 max_abs=6e-8.
  timed act=cur=2800 throttle=0. M=1
  pipe_host 29.955/29.850 vs s4 16.5
  vs s8xs4 22.1 vs s8 34 vs W8A8 44.
  M=4 tracks. Spread ~0.4%. ~1.81x
  native s4.

VERDICT -> New GPTQ s4 decode floor
  29.9 us pipe_host at 2800 both cards.
  Beats s8 and W8A8, loses to s4 16.5.
  Scale tax ~13 us. Rank pipe_host.

Evidence: `results/k6/gptq_s4sc_n2_s4000_card0.txt`,
  `results/k6/gptq_s4sc_n2_s4000_card1.txt`.

## GPTQ s4 N=17408 is 100 us both cards (K6)

CONFIG -> backend `sycl+l0`, same
  `dpas_s4_gptq_sc`. M=1 N=17408 K=5120
  gate_proj dump. Both cards. Named
  clock 2800. NT=2 spin=4000. Prior:
  square 29.9; N-linear ~102; s4 29.5.

RESULT -> cosine=1.0 max_abs=6e-8.
  timed act=cur=2800 throttle=0. M=1
  pipe_host 100.028/100.068 vs square
  29.9 vs s4 29.5 vs s8xs4 38.6 vs s8
  141.6 vs W8A8 158.1 vs napkin 102.
  M=4 tracks. Spread ~0.04%. ~3.35x
  square.

VERDICT -> New GPTQ s4 N=17408 floor
  100 us pipe_host at 2800 both cards.
  Near linear. Beats s8 and W8A8,
  loses to s4 and s8xs4. Rank
  pipe_host.

Evidence: `results/k6/gptq_s4sc_n17408_n2_s4000_card0.txt`,
  `results/k6/gptq_s4sc_n17408_n2_s4000_card1.txt`.

## GPTQ s4 K=17408 is 174.6 us both cards (K6)

CONFIG -> backend `sycl+l0`, same
  `dpas_s4_gptq_sc`. M=1 N=5120 K=17408
  down_proj dump. Both cards. Named
  clock 2800. NT=2 spin=4000. Prior:
  square 29.9; N-wide 100; K-linear
  ~102; s4 53.4; W8A8 155.3.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0. M=1
  pipe_host 174.629/174.509 vs square
  29.9 vs N-wide 100 vs s4 53.4 vs
  s8xs4 73.2 vs s8 261.6 vs W8A8
  155.3 vs napkin 102. M=4 tracks.
  Spread ~0.07%. ~5.84x square.

VERDICT -> New GPTQ s4 K=17408 floor
  174.6 us pipe_host at 2800 both
  cards. K-hostile. Qwen FFN GPTQ
  decode map is closed (29.9 / 100 /
  174.6). Beats s8, loses to s4,
  s8xs4, and W8A8 (~1.12x). Rank
  pipe_host.

Evidence: `results/k6/gptq_s4sc_k17408_n2_s4000_card0.txt`,
  `results/k6/gptq_s4sc_k17408_n2_s4000_card1.txt`.

## GPTQ 8x2-N loses at M=64 (K6)

CONFIG -> backend `sycl+l0`, same
  `dpas_s4_gptq_sc`. RC=4 NT=2 gs=128.
  M=64 N=K=5120. Card0. spin=512.
  Named clock 2800. Prior: decode
  29.9; s4 4x8 33.6; W8A8 46.

RESULT -> cosine=1.0 max_abs=3e-5.
  timed act=2767 cur=2800 throttle=1.
  M=64 pipe_host 123.528 vs decode
  29.9 vs s4 33.6 vs mix 43.3 vs s8
  75 vs W8A8 46. ~4.13x decode.

VERDICT -> Decode-tile GPTQ at M=64
  is not a prefill floor. Loses to
  s4, mix, W8A8 (~2.68x), and s8.
  Stop 8x2-N GPTQ at M=64 prefill.
  One-card. Rank pipe_host.

Evidence: `results/k6/gptq_s4sc_m64_n2_s512_card0.txt`.

## GPTQ 8x2-N loses at M=256 (K6)

CONFIG -> backend `sycl+l0`, same
  `dpas_s4_gptq_sc`. M=256 N=K=5120.
  Card1. NT=2 spin=512. Named clock
  2800. Prior: s4 4-acc 48.6; W8A8
  75.

RESULT -> cosine=1.0 max_abs=3e-5.
  timed act=2550 cur=2800 throttle=1.
  M=256 pipe_host 354.611 vs s4 48.6
  vs s8 128 vs W8A8 75 vs compose
  607. ~2.87x M=64.

VERDICT -> Decode-tile GPTQ at M=256
  loses to s4, s8, and W8A8 (~4.73x).
  Stop 8x2-N GPTQ at M=256 prefill.
  One-card. Rank pipe_host.

Evidence: `results/k6/gptq_s4sc_m256_n2_s512_card1.txt`.

## GPTQ 4x8 A-db loses at M=64 (K6)

CONFIG -> backend `sycl+l0`,
  `dpas_s4_gptq_db48`. RC=8 wg 4x8
  A-db gs=128. M=64 N=K=5120. Card0.
  NT=2 spin=512. Named clock 2800.
  Prior: 8x2-N 123.5; s4 33.6; mix
  43.3; W8A8 46; napkin ~61.

RESULT -> cosine=1.0 max_abs=3e-5.
  timed act=cur=2800 throttle=0. M=64
  pipe_host 102.936 vs 8x2-N 123.5 vs
  s4 33.6 vs mix 43.3 vs W8A8 46 vs
  napkin 61. ~3.06x s4.

VERDICT -> GPTQ 4x8 beats 8x2-N
  (~1.20x) but is not a prefill
  floor. Loses to s4, mix, and W8A8
  (~2.24x). Napkin 61 miss. One-card.
  Rank pipe_host.

Evidence: `results/k6/gptq_s4db48_m64_n2_s512_card0.txt`.

## GPTQ 4x8 A-db loses at M=256 (K6)

CONFIG -> backend `sycl+l0`, same
  `dpas_s4_gptq_db48`. M=256 N=K=5120.
  Card1. NT=2 spin=512. Named clock
  2800. Prior: 8x2-N 355; s4 48.6;
  mix 123; W8A8 75.

RESULT -> cosine=1.0 max_abs=3e-5.
  timed act=cur=2800 throttle=0. M=256
  pipe_host 302.722 vs 8x2-N 355 vs
  s4 48.6 vs mix 123 vs W8A8 75.
  ~2.94x M=64.

VERDICT -> GPTQ 4x8 at M=256 beats
  8x2-N (~1.17x), loses to s4, mix,
  and W8A8 (~4.04x). Stop GPTQ 4x8
  at prefill vs W8A8. One-card. Rank
  pipe_host.

Evidence: `results/k6/gptq_s4db48_m256_n2_s512_card1.txt`.

## s8xs4 4x8 A-db M=64 is 43.3 us both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8xs4_db48`. RC=8 wg 4x8 A-db.
  A=s8 B=s4 pack=2. M=64 N=K=5120.
  Both cards. NT=2 spin=512. Named
  clock 2800.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 43.431/43.286 vs 8x2-N 114
  vs s4 33.6 vs s8 75 vs W8A8 46.
  Spread ~0.33%.

VERDICT -> New s8xs4 4x8 A-db M=64
  floor 43.3 us pipe_host at 2800 both
  cards. Mix 4x8 is ~2.64x the decode
  tile at M=64. Beats s8 and W8A8,
  loses to s4 (~1.29x). Rank
  pipe_host.

Evidence: `results/k2/s8xs4db48_m64_n2_s512_card0.txt`,
  `results/k2/s8xs4db48_m64_n2_s512_card1.txt`.

## s8xs4 4x8 A-db M=256 loses both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_db48`. RC=8 wg 4x8 A-db.
  A=s8 B=s4 pack=2. M=256 N=K=5120.
  Both cards. NT=2 spin=512. Named
  clock 2800. Prior: M=64 43.3;
  napkin ~173; s4 4-acc 48.6; W8A8
  75.

RESULT -> cosine=1.0 max_abs=0. timed
  act=2767/2750 cur=2800 throttle=1.
  M=256 pipe_host 122.830/123.272 vs
  M=64 43.3 vs s4 48.6 vs s8 128 vs
  W8A8 75 vs compose 194.9. Spread
  ~0.36%. ~2.85x M=64.

VERDICT -> Mix 4x8 at M=256 is 123 us
  pipe_host both cards, throttle=1.
  Under linear but not a prefill
  floor. Beats s8 and compose, loses
  to s4 (~2.54x) and W8A8 (~1.64x).
  Stop 4x8 mix at M=256 prefill.
  Rank pipe_host.

Evidence: `results/k2/s8xs4db48_m256_n2_s512_card0.txt`,
  `results/k2/s8xs4db48_m256_n2_s512_card1.txt`.

## s8xs4 4x8 A-db M=64 N=17408 is 129 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_db48`. RC=8 wg 4x8 A-db.
  A=s8 B=s4 pack=2. M=64 N=17408
  K=5120. Both cards. NT=2 spin=512.
  Named clock 2800. Prior: square
  43.3; s4 94.7; W8A8 202; napkin
  ~147.

RESULT -> cosine=1.0 max_abs=0. M=64
  pipe_host 126.931/129.215 vs square
  43.3 vs s4 94.7 vs s8 338.9 vs
  W8A8 202. card0 act=2783 throttle=1.
  card1 act=cur=2800 throttle=0.
  Spread ~1.80%. ~2.98x square.

VERDICT -> New mix 4x8 M=64 N=17408
  floor 129 us pipe_host both cards
  (held-clock sibling). Beats W8A8
  (~1.56x) and s8, loses to s4
  (~1.36x). Rank pipe_host.

Evidence: `results/k2/s8xs4db48_m64_n17408_n2_s512_card0.txt`,
  `results/k2/s8xs4db48_m64_n17408_n2_s512_card1.txt`.

## s8xs4 4x8 A-db M=64 K=17408 is 144.7 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_db48`. M=64 N=5120
  K=17408. Both cards. NT=2 spin=512.
  Named clock 2800. Prior: square
  43.3; s4 106.0; W8A8 181; napkin
  ~147.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 144.261/144.684 vs square
  43.3 vs s4 106.0 vs s8 374.7 vs
  W8A8 181. Spread ~0.29%. ~3.34x
  square.

VERDICT -> New mix 4x8 M=64 K=17408
  floor 144.7 us pipe_host both cards
  at 2800. Beats W8A8 (~1.25x) and
  s8, loses to s4 (~1.37x). Qwen FFN
  mix M=64 map is closed (43.3 / 129
  / 144.7). Rank pipe_host.

Evidence: `results/k2/s8xs4db48_m64_k17408_n2_s512_card0.txt`,
  `results/k2/s8xs4db48_m64_k17408_n2_s512_card1.txt`.

## s8xs4 8x2-N loses at M=64 (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8xs4_sc`. M=64 N=K=5120.
  Card1. NT=2 spin=512. Named clock
  2800.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 114.146 vs M=1 22.1 vs s4
  4x8 33.6 vs s8 75 vs W8A8 46 vs
  compose 217.9. ~5.16x M=1.

VERDICT -> Decode-tile mix is ~3.4x
  s4 4x8 at M=64. Not a prefill floor.
  Stop 8x2-N s8xs4 at prefill. One-card.

Evidence: `results/k2/s8xs4sc_m64_n2_s512_card1.txt`.

## 256-entry product LUT GEMV is a numeric-closed loss (K6)

CONFIG -> backend `sycl+l0`, `prod_lut_gemv` W4A4
  16x16 E2M1 product table, M=1 N=K=5120. Both
  cards. Naive per-column scalar loop. Never
  bitcast s4.

RESULT -> max_abs=0 ok=1 both cards. card1 697.042
  us (end cur=1050). card0 1105.573 us (start 1983
  end 2800). Clocks unmatched.

VERDICT -> Numeric closed, us lost vs closed-form
  134.8 (~5-8x) and vs s8 34. Stop this hail-mary
  as a serving tile.

Evidence: `results/k6/sprint_mix_prod_card1.txt`,
  `results/k6/sprint_prod_card0.txt`.

## oneDNN nvfp4_gemm_w4a16 M=1 is ~37 us after M=64 heat (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Image `b70-sglang-xpu-int8-runtime:20260826-mtp6`
  does not export the op. Load
  `/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so`
  via `torch.ops.load_library` without importing
  the image `_xpu_C` first. B packed NT, stride(0)=1.
  A bf16, W uint8 [K/2,N], group 16. M=64 heat then
  us_bench M=1 5120. Both cards.

RESULT -> HAS nvfp4_gemm_w4a16 and f8scale. out
  bf16 [1,5120]. Folded bf16 scale 36.809/37.169
  us. f8scale (e4m3 group-16 + fp32 global)
  38.448/39.611 us. No E2M1 cosine this dump.
  Clocks not held 2800: card0 freq 2800 then 1583;
  card1 1750 then 1383, never 2800.

VERDICT -> The 27B-class incumbent lights and is
  in the same us class as W8A8 44, not Family-A
  LUT 135. A is bf16, not s8. f8scale is the real
  NVFP4 group-16 epilogue in oneDNN; hand s8 DPAS
  cannot match that isolation. Do not freeze 37 us
  against held-2800 s8 34. Rank us with clocks.

Evidence: `results/k6/nvfp4_w4a16_m1_card0.txt`,
  `results/k6/nvfp4_w4a16_m1_card1.txt`.

## Held-clock nvfp4_gemm_w4a16 M=1 is 34.7 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so, packed NT, g16. M=64 heat
  then M=1 spin=2000 then us_bench M=1
  5120. Both cards. Named clock 2800.

RESULT -> spin/timed act=cur=2800
  throttle=0. Folded bf16 scale
  34.964/34.395 us vs unheld 36.809/37.169
  vs s8 34 vs W8A8 44 vs LUT 134.8.
  Spread ~1.6%. f8scale 37.738/37.944.
  out bf16 [1,5120]. No E2M1 cosine.

VERDICT -> New held-clock folded w4a16
  M=1 floor 34.7 us at 2800 both cards.
  Same us class as s8 34, under W8A8 44,
  ~3.9x LUT 135. A is bf16, not s8. Do
  not call this a beat of s8. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m1hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m1hold_card1.txt`.

## nvfp4_gemm_w4a16 M=64 is 37.1 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. Card1 spin=1000, card0
  spin=2000 of M=64 then us_bench M=64
  5120. Both cards.

RESULT -> out bf16 [64,5120]. timed
  act=2150/2400 cur=2800 throttle=0.
  Folded 37.406/36.761 us vs M=1 34.7 vs
  W8A8 46 vs s8 75 vs compose 68.7 vs LUT
  331.6. Spread ~1.7%. f8scale
  39.153/39.047. More spin did not raise
  act.

VERDICT -> New w4a16 M=64 floor 37.1 us
  both cards, cur=2800, act 2150-2400.
  ~1.07x M=1, under W8A8 46, ~2.02x s8
  75. Not a 2800-act hold. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m64hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m64hold_card1.txt`.

## nvfp4_gemm_w4a16 M=256 is 118 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. spin=512 of M=256 then
  us_bench M=256 5120. Both cards. Card0
  try1 folded 140 us discarded (cold).

RESULT -> out bf16 [256,5120]. timed
  act=2350/2550 cur=2800 throttle=0.
  Folded 120.540/115.560 us vs M=64 37.1
  vs W8A8 75 vs s8 128 vs compose 194.9
  vs LUT 1083 vs s4 48.6. Spread ~4.2%.
  f8scale 116.389/114.151.

VERDICT -> New w4a16 M=256 floor 118 us
  both cards, cur=2800, act 2350-2550.
  ~3.18x M=64, loses to W8A8 75 (~1.57x)
  and s4 48.6, beats s8 128. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m256hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m256hold_card1.txt`.

## nvfp4_gemm_w4a16 M=1 N=17408 is 97 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. M=64 heat on same B,
  spin=2000 of M=1 then us_bench M=1
  N=17408 K=5120. Both cards.

RESULT -> out bf16 [1,17408]. timed
  act=2700/2717 cur=2800 throttle=1.
  Folded 97.878/97.050 us vs square 34.7
  vs s8 141.6 vs W8A8 158.1 vs s4 29.5
  vs compose 103.5 vs napkin 118. Spread
  ~0.8%. f8scale 89.449/89.023.

VERDICT -> New w4a16 M=1 wide-N floor 97
  us both cards at ~2700/2800, throttle=1.
  ~2.80x square, beats s8 141.6 and W8A8
  158.1, loses to s4 29.5. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m1_n17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m1_n17408_hold_card1.txt`.

## nvfp4_gemm_w4a16 M=1 K=17408 is 101 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. M=64 heat on same B,
  spin=2000 of M=1 then us_bench M=1
  N=5120 K=17408. Both cards.

RESULT -> out bf16 [1,5120]. timed
  act=2733/2750 cur=2800 throttle=1.
  Folded 101.909/101.270 us vs square 34.7
  vs N=17408 97 vs s8 261.6 vs W8A8 155.3
  vs s4 53.4 vs compose 193.6 vs napkin
  118. Spread ~0.6%. f8scale 97.361/97.931.

VERDICT -> New w4a16 M=1 wide-K floor 101
  us both cards at ~2740/2800, throttle=1.
  ~2.92x square, beats s8 261.6 and W8A8
  155.3, loses to s4 53.4. Qwen FFN
  w4a16 decode map is closed. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m1_k17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m1_k17408_hold_card1.txt`.

## nvfp4_gemm_w4a16 M=64 N=17408 is 142 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. spin=512 of M=64 then
  us_bench M=64 N=17408 K=5120. Both cards.

RESULT -> out bf16 [64,17408]. timed
  act=2050/2300 cur=2800 throttle=0.
  Folded 144.451/138.903 us vs square 37.1
  vs M=1 N=17408 97 vs s8 338.9 vs s4 94.7
  vs compose 326.9 vs LUT 880 vs napkin
  104. Spread ~4.0%. f8scale 141.560/138.595.

VERDICT -> New w4a16 M=64 wide-N floor 142
  us both cards, cur=2800, act 2050-2300.
  ~3.82x square (napkin 104 missed), beats
  s8 338.9, loses to s4 94.7. Act not 2800.
  Rank us.

Evidence: `results/k6/nvfp4_w4a16_m64_n17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m64_n17408_hold_card1.txt`.

## nvfp4_gemm_w4a16 M=64 K=17408 is 130 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. spin=512 of M=64 then
  us_bench M=64 N=5120 K=17408. Both cards.

RESULT -> out bf16 [64,5120]. timed
  act=2100-2400 cur=2800 throttle=0.
  Folded 132.966/127.793 us vs square 37.1
  vs M=1 K=17408 101 vs N-wide 142 vs s8
  374.7 vs s4 106.0 vs compose 403.4 vs
  LUT 1125 vs napkin 108. Spread ~4.0%.
  f8scale 139.400/129.001. ~K-linear
  (37.1*17408/5120 ~126).

VERDICT -> New w4a16 M=64 wide-K floor 130
  us both cards, cur=2800, act 2100-2400.
  ~3.51x square, K-linear held, beats s8
  374.7, loses to s4 106.0. Qwen FFN
  w4a16 M=64 map is closed. Act not 2800.
  Rank us.

Evidence: `results/k6/nvfp4_w4a16_m64_k17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m64_k17408_hold_card1.txt`.

## nvfp4_gemm_w4a16 M=256 N=17408 is 394 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. spin=512 of M=256 then
  us_bench M=256 N=17408 K=5120. Both cards.

RESULT -> out bf16 [256,17408]. timed
  act=2250-2300 cur=2800 throttle=1.
  Folded 390.997/397.434 us vs square 118
  vs M=64 N=17408 142 vs s8 469.8 vs s4
  140.0 vs compose 984.3 vs LUT 3138 vs
  napkin 330. Spread ~1.6%. f8scale
  384.683/390.800. ~N-linear
  (118*17408/5120 ~401).

VERDICT -> New w4a16 M=256 wide-N floor 394
  us both cards, cur=2800, act 2250-2300,
  throttle=1. ~3.34x square, N-linear held,
  beats s8 469.8, loses to s4 140.0. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m256_n17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m256_n17408_hold_card1.txt`.

## nvfp4_gemm_w4a16 M=256 K=17408 is 377 us both cards (K6)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  Same v028 so. spin=512 of M=256 then
  us_bench M=256 N=5120 K=17408. Both cards.

RESULT -> out bf16 [256,5120]. timed
  act=2250-2317 cur=2800 throttle=1.
  Folded 378.110/375.885 us vs square 118
  vs M=64 K=17408 130 vs N-wide 394 vs s8
  477.4 vs s4 149.0 vs compose 968.7 vs
  LUT 3428 vs napkin 343. Spread ~0.6%.
  f8scale 367.919/361.498. Under K-linear
  (118*17408/5120 ~401).

VERDICT -> New w4a16 M=256 wide-K floor 377
  us both cards, cur=2800, act 2250-2317,
  throttle=1. ~3.19x square, under K-linear,
  beats s8 477.4, loses to s4 149.0. Qwen
  FFN w4a16 M=256 map is closed. Rank us.

Evidence: `results/k6/nvfp4_w4a16_m256_k17408_hold_card0.txt`,
  `results/k6/nvfp4_w4a16_m256_k17408_hold_card1.txt`.

## oneDNN W8A8 M=256 N=17408 is 248 us both cards (K1/K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  mtp6 `int8_gemm_w8a8` GEMM-only. spin=512
  of M=256 then us_bench M=256 N=17408
  K=5120. Both cards. Oracle after timed.

RESULT -> out f16 [256,17408]. timed
  act=2467-2517 cur=2800 throttle=1.
  248.232/248.116 us vs square 75 vs
  w4a16 394 vs s8 469.8 vs s4 140.0 vs
  napkin 255. Spread ~0.05%. cosine=1.000
  max_abs=0.063. 359 GB/s.

VERDICT -> New W8A8 M=256 wide-N floor 248
  us both cards, cur=2800, act 2467-2517,
  throttle=1. ~3.31x square, N-linear held,
  beats w4a16 394 (~1.59x) and hand s8
  469.8, loses to s4 140.0. Numeric closed.
  Rank us.

Evidence: `results/k2/w8a8_m256_n17408_hold_card0.txt`,
  `results/k2/w8a8_m256_n17408_hold_card1.txt`.

## oneDNN W8A8 M=256 K=17408 is 226 us both cards (K1/K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  mtp6 `int8_gemm_w8a8` GEMM-only. spin=512
  of M=256 then us_bench M=256 N=5120
  K=17408. Both cards. Oracle after timed.

RESULT -> out f16 [256,5120]. timed
  act=2483-2567 cur=2800 throttle=1.
  223.594/228.094 us vs square 75 vs
  N-wide 248 vs w4a16 377 vs s8 477.4 vs
  s4 149.0 vs napkin 265. Spread ~2.0%.
  cosine=1.000 max_abs=0.125. 399/391 GB/s.
  Under K-linear (75*17408/5120 ~255).

VERDICT -> New W8A8 M=256 wide-K floor 226
  us both cards, cur=2800, act 2483-2567,
  throttle=1. ~3.01x square, under K-linear,
  beats w4a16 377 (~1.67x) and hand s8
  477.4, loses to s4 149.0. Qwen FFN W8A8
  M=256 map is closed. Numeric closed.
  Rank us.

Evidence: `results/k2/w8a8_m256_k17408_hold_card0.txt`,
  `results/k2/w8a8_m256_k17408_hold_card1.txt`.

## oneDNN W8A8 M=64 N=17408 is 202 us both cards (K1/K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  mtp6 `int8_gemm_w8a8` GEMM-only. spin=512
  of M=64 then us_bench M=64 N=17408
  K=5120. Both cards. Oracle after timed.

RESULT -> out f16 [64,17408]. timed
  act=2783 cur=2800 throttle=1.
  202.772/201.221 us vs square 46 vs
  w4a16 142 vs s8 338.9 vs s4 94.7 vs
  napkin 156. Spread ~0.8%. cosine=1.000
  max_abs=0.062. 440/443 GB/s. ~4.39x
  square, superlinear.

VERDICT -> New W8A8 M=64 wide-N floor 202
  us both cards at 2783/2800, throttle=1.
  Loses to w4a16 142 (~1.42x) both cards
  and s4 94.7, beats hand s8 338.9. Napkin
  156 missed. Crossover holds: w4a16 wins
  M=1 and M=64 FFN-up; W8A8 wins M=256.
  Numeric closed. Rank us.

Evidence: `results/k2/w8a8_m64_n17408_hold_card0.txt`,
  `results/k2/w8a8_m64_n17408_hold_card1.txt`.

## oneDNN W8A8 M=64 K=17408 is 181 us both cards (K1/K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`.
  mtp6 `int8_gemm_w8a8` GEMM-only. spin=512
  of M=64 then us_bench M=64 N=5120
  K=17408. Both cards. Oracle after timed.

RESULT -> out f16 [64,5120]. timed
  act=2717-2733 cur=2800 throttle=1.
  177.372/184.009 us vs square 46 vs
  N-wide 202 vs w4a16 130 vs s8 374.7 vs
  s4 106.0 vs napkin 162. Spread ~3.7%.
  cosine=1.000 max_abs=0.123. 502/484 GB/s.
  ~3.93x square, superlinear vs 156.

VERDICT -> New W8A8 M=64 wide-K floor 181
  us both cards at ~2725/2800, throttle=1.
  Loses to w4a16 130 (~1.39x) both cards
  and s4 106.0, beats hand s8 374.7. Qwen
  FFN W8A8 M=64 map is closed. Numeric
  closed. Rank us.

Evidence: `results/k2/w8a8_m64_k17408_hold_card0.txt`,
  `results/k2/w8a8_m64_k17408_hold_card1.txt`.

## ESIMD s2 decode tile is 11.5 us both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  icpx AOT `intel_gpu_bmg_g31`. `dpas_s2_sc`
  RC=4 8x2-N scale-to-f16, pack=4 along K,
  IGC s2 [-2,1]. NT=2 spin=4000. Both cards.
  Never E2M1 bitcast.

RESULT -> check cosine=1.0 max_abs=0.
  timed M=1 5120 act=cur=2800 throttle=0.
  pipe_host 11.468/11.474 us vs s4 16.5
  vs s8 34 vs W8A8 44. M=4 tracks.
  Spread ~0.05%. ~1.43x s4, ~2.96x s8.
  Napkin 8 (2x s4) missed.

VERDICT -> New s2 decode floor 11.5 us
  pipe_host at 2800 both cards. Numeric
  closed. Beats s4, not 2x s4. Rank
  pipe_host.

Evidence: `results/k2/s2sc_n2_s4000_card0.txt`,
  `results/k2/s2sc_n2_s4000_card1.txt`.

## ESIMD s2 4x8 A-db M=64 is 20 us both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s2_db48`. RC=8 wg 4x8 A-db
  pack=4 along K. IGC s2 [-2,1]. M=64
  N=K=5120. Both cards. NT=2 spin=512.
  Named clock 2800. Never E2M1.
  Prior: s2 decode 11.5; s4 4x8 33.6;
  W8A8 46.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 19.794/20.814 vs s2
  decode 11.5 vs s4 33.6 vs W8A8 46.
  event 19.318/19.667. pipe spread
  ~5.2%, event ~1.8%.

VERDICT -> New s2 4x8 A-db M=64 floor
  20 us pipe_host at 2800 both cards.
  Numeric closed. Beats s4 33.6
  (~1.68x) and W8A8 46 (~2.21x). New
  M=64 hand floor. Rank pipe_host.

Evidence: `results/k2/s2db48_m64_n2_s512_card0.txt`,
  `results/k2/s2db48_m64_n2_s512_card1.txt`.

## s2 4x8 A-db M=64 N=17408 is 53.1 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_db48`. M=64 N=17408 K=5120.
  Both cards. NT=2 spin=512. Named
  clock 2800. Prior: square 20; s4
  94.7; mix 129; W8A8 202; napkin
  ~68.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 53.079/52.859 vs square
  20 vs s4 94.7 vs mix 129 vs W8A8
  202. Spread ~0.41%. ~2.65x square.

VERDICT -> New s2 4x8 M=64 N=17408
  floor 53.1 us pipe_host at 2800
  both cards. Beats W8A8 (~3.81x),
  mix, and s4. Rank pipe_host.

Evidence: `results/k2/s2db48_m64_n17408_n2_s512_card0.txt`,
  `results/k2/s2db48_m64_n17408_n2_s512_card1.txt`.

## s2 4x8 A-db M=64 K=17408 is 64 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_db48`. M=64 N=5120 K=17408.
  Both cards. NT=2 spin=512. Named
  clock 2800. Prior: square 20; s4
  106.0; mix 144.7; W8A8 181; napkin
  ~68.

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0. M=64
  pipe_host 64.044/62.540 vs square
  20 vs s4 106.0 vs mix 144.7 vs
  W8A8 181. Spread ~2.4%. ~3.20x
  square.

VERDICT -> New s2 4x8 M=64 K=17408
  floor 64 us pipe_host at 2800 both
  cards. Beats W8A8 (~2.83x), mix,
  and s4. Qwen FFN s2 M=64 map is
  closed (20 / 53.1 / 64). Rank
  pipe_host.

Evidence: `results/k2/s2db48_m64_k17408_n2_s512_card0.txt`,
  `results/k2/s2db48_m64_k17408_n2_s512_card1.txt`.

## ESIMD s2 4x8 A-db M=256 is 55.5 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_db48`. RC=8 wg 4x8 A-db
  pack=4 along K. IGC s2 [-2,1].
  M=256 N=K=5120. Both cards. NT=2
  spin=512. Named clock 2800.
  Never E2M1. Prior: M=64 20; s4
  4-acc 48.6; mix 123 a loss;
  W8A8 75; napkin ~80.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 55.453/55.497 vs
  M=64 20 vs s4 48.6 vs mix 123 vs
  W8A8 75. Spread ~0.08%. ~2.77x
  M=64.

VERDICT -> New s2 4x8 A-db M=256
  floor 55.5 us pipe_host at 2800
  both cards. Numeric closed. Beats
  W8A8 75 (~1.35x) and mix 123.
  Loses to s4 4-acc 48.6 (~1.14x).
  Not the M=256 hand floor. Rank
  pipe_host.

Evidence: `results/k2/s2db48_m256_n2_s512_card0.txt`,
  `results/k2/s2db48_m256_n2_s512_card1.txt`.

## s2 4x8 A-db M=256 N=17408 is 171 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_db48`. M=256 N=17408
  K=5120. Both cards. NT=2
  spin=512. Named clock 2800.
  Prior: square 55.5; s4 140;
  W8A8 248; napkin ~189.

RESULT -> cosine=1.0 max_abs=0.
  timed act=2750/2767 cur=2800
  throttle=1 both. M=256 pipe_host
  170.943/170.446 vs square 55.5
  vs s4 140 vs W8A8 248. Spread
  ~0.29%. ~3.08x square.

VERDICT -> New s2 4x8 M=256 N=17408
  floor 171 us pipe_host at 2800
  both cards. throttle=1. Beats
  W8A8 248 (~1.45x). Loses to s4
  140 (~1.22x). Rank pipe_host.

Evidence: `results/k2/s2db48_m256_n17408_n2_s512_card0.txt`,
  `results/k2/s2db48_m256_n17408_n2_s512_card1.txt`.

## s2 4x8 A-db M=256 K=17408 is 201 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_db48`. M=256 N=5120
  K=17408. Both cards. NT=2
  spin=512. Named clock 2800.
  Prior: square 55.5; N-wide 171
  throttle=1; s4 149; W8A8 226;
  napkin ~189.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 201.019/199.070
  vs square 55.5 vs N-wide 171 vs
  s4 149 vs W8A8 226. Spread
  ~0.98%. ~3.62x square.

VERDICT -> New s2 4x8 M=256 K=17408
  floor 201 us pipe_host at 2800
  both cards. throttle=0. Beats
  W8A8 226 (~1.12x). Loses to s4
  149 (~1.35x). Qwen FFN s2 M=256
  map is closed (55.5 / 171 / 201).
  Rank pipe_host.

Evidence: `results/k2/s2db48_m256_k17408_n2_s512_card0.txt`,
  `results/k2/s2db48_m256_k17408_n2_s512_card1.txt`.

## ESIMD s2xs8 decode mix is 14.1 us both cards (K2)

CONFIG -> backend `sycl+l0`, standalone
  AOT `dpas_s2xs8_sc`. A=s8 B=s2 pack=4,
  dpas K=32 (OPC=4). RC=4 8x2-N scale-to-
  f16. NT=2 spin=4000. Both cards.
  Literature mix arXiv 2508.06753. Never
  E2M1 bitcast.

RESULT -> check cosine=1.0 max_abs=0.
  timed M=1 5120 act=cur=2800 throttle=0.
  pipe_host 13.971/14.140 us vs s2 11.5
  vs s4 16.5 vs s8 34 vs napkin 34. M=4
  tracks. Spread ~1.2%. ~2.41x s8.

VERDICT -> New s2xs8 decode floor 14.1
  us pipe_host at 2800 both cards.
  Numeric closed. Beats s8, loses to
  s2xs2 11.5. Paper same-rate napkin
  missed. Rank pipe_host.

Evidence: `results/k2/s2xs8sc_n2_s4000_card0.txt`,
  `results/k2/s2xs8sc_n2_s4000_card1.txt`.

## ESIMD s2xs8 4x8 A-db M=64 is 33.2 us both cards (K2)

CONFIG -> backend `sycl+l0`,
  standalone `dpas_s2xs8_db48`.
  A=s8 B=s2 pack=4, dpas K=32
  (OPC=4). RC=8 wg 4x8 A-db.
  M=64 N=K=5120. Both cards. NT=2
  spin=512. Named clock 2800.
  Literature mix arXiv 2508.06753.
  Never E2M1. Prior: s2 20; s8xs4
  43.3; s8 75; W8A8 46; decode
  14.1.

RESULT -> ocloc 64x `dpas.8x8`
  rW:s2 rA:b, grf 128, B d8v rd:2,
  no SLM. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64 pipe_host 33.152/33.200 vs
  s2 20 vs s8xs4 43.3 vs s8 75 vs
  W8A8 46. Spread ~0.14%.

VERDICT -> New s2xs8 4x8 A-db M=64
  floor 33.2 us pipe_host at 2800
  both cards. Numeric closed. Beats
  W8A8 46 (~1.39x) and s8xs4 43.3.
  Loses to native s2 20 (~1.66x).
  Rank pipe_host.

Evidence: `results/k2/s2xs8db48_m64_n2_s512_card0.txt`,
  `results/k2/s2xs8db48_m64_n2_s512_card1.txt`,
  `results/k2/s2xs8db48_dpas_lines.txt`.

## s2xs8 4x8 A-db M=64 N=17408 is 100.5 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2xs8_db48`. A=s8 B=s2
  pack=4. M=64 N=17408 K=5120.
  Both cards. NT=2 spin=512. Named
  clock 2800. Never E2M1. Prior:
  square 33.2; s2 53.1; mix 129;
  W8A8 202; napkin ~113.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64 pipe_host 100.137/100.528
  vs square 33.2 vs s2 53.1 vs mix
  129 vs W8A8 202. Spread ~0.39%.
  ~3.03x square.

VERDICT -> New s2xs8 4x8 M=64
  N=17408 floor 100.5 us pipe_host
  at 2800 both cards. Beats W8A8
  202 (~2.01x) and mix 129. Loses
  to s2 53.1 (~1.89x). Rank
  pipe_host.

Evidence: `results/k2/s2xs8db48_m64_n17408_n2_s512_card0.txt`,
  `results/k2/s2xs8db48_m64_n17408_n2_s512_card1.txt`.

## s2xs8 4x8 A-db M=64 K=17408 is 107 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2xs8_db48`. A=s8 B=s2
  pack=4. M=64 N=5120 K=17408.
  Both cards. NT=2 spin=512. Named
  clock 2800. Never E2M1. Prior:
  square 33.2; N-wide 100.5; s2 64;
  mix 144.7; W8A8 181; napkin ~113.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64 pipe_host 106.722/107.385
  vs square 33.2 vs N-wide 100.5 vs
  s2 64 vs mix 144.7 vs W8A8 181.
  Spread ~0.62%. ~3.23x square.

VERDICT -> New s2xs8 4x8 M=64
  K=17408 floor 107 us pipe_host
  at 2800 both cards. Beats W8A8
  181 (~1.69x) and mix 144.7.
  Loses to s2 64 (~1.67x). Qwen
  FFN s2xs8 M=64 map is closed
  (33.2 / 100.5 / 107). Rank
  pipe_host.

Evidence: `results/k2/s2xs8db48_m64_k17408_n2_s512_card0.txt`,
  `results/k2/s2xs8db48_m64_k17408_n2_s512_card1.txt`.

## s2xs8 4x8 A-db M=256 loses at 96 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2xs8_db48`. A=s8 B=s2
  pack=4. M=256 N=K=5120. Both
  cards. NT=2 spin=512. Named
  clock 2800. Never E2M1. Prior:
  M=64 33.2; s2 55.5; mix 123 a
  loss; W8A8 75; napkin ~133.

RESULT -> cosine=1.0 max_abs=0.
  timed act=2767/2800 throttle=1
  both. M=256 pipe_host
  95.536/95.735 vs M=64 33.2 vs
  s2 55.5 vs mix 123 vs W8A8 75.
  Spread ~0.21%. ~2.88x M=64.

VERDICT -> s2xs8 4x8 M=256 is 96 us
  pipe_host at 2800 both cards.
  throttle=1. Beats mix 123. Loses
  to s2 55.5 (~1.72x) and W8A8 75
  (~1.27x). Stop 4x8 mix at M=256
  prefill vs W8A8. Rank pipe_host.

Evidence: `results/k2/s2xs8db48_m256_n2_s512_card0.txt`,
  `results/k2/s2xs8db48_m256_n2_s512_card1.txt`.

## ESIMD s2 4-acc M=256 is 37.4 us both cards (K2)

CONFIG -> backend `sycl+l0`,
  standalone `dpas_s2_w48m4`. RC=8
  4-acc wg 4x8 k128 pack=4. IGC s2
  [-2,1]. M=256 N=K=5120. Both
  cards. NT=2 spin=512. Named
  clock 2800. Never E2M1. Prior:
  s4 4-acc 48.6; s2 4x8 55.5;
  W8A8 75; napkin ~29.

RESULT -> ocloc 128x `dpas.8x8`
  rW:s2 rA:s2, grf 128, B d8v rd:4,
  no SLM. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 37.409/37.405 vs
  s4 48.6 vs s2 4x8 55.5 vs W8A8
  75. Spread ~0.01%.

VERDICT -> New s2 4-acc M=256 floor
  37.4 us pipe_host at 2800 both
  cards. New M=256 hand floor.
  Numeric closed. Beats s4 48.6
  (~1.30x) and W8A8 75 (~2.01x).
  Napkin 29 miss. Rank pipe_host.

Evidence: `results/k2/s2w48m4_m256_n2_s512_card0.txt`,
  `results/k2/s2w48m4_m256_n2_s512_card1.txt`,
  `results/k2/s2w48m4_dpas_lines.txt`.

## s2 4-acc M=256 N=17408 is 110 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_w48m4`. M=256 N=17408
  K=5120. Both cards. NT=2
  spin=512. Named clock 2800.
  Prior: square 37.4; s4 140;
  W8A8 248; napkin ~127.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 108.604/109.947
  vs square 37.4 vs s4 140 vs
  W8A8 248. Spread ~1.24%. ~2.94x
  square.

VERDICT -> New s2 4-acc M=256
  N=17408 floor 110 us pipe_host
  at 2800 both cards. Beats s4 140
  (~1.27x) and W8A8 248 (~2.25x).
  Rank pipe_host.

Evidence: `results/k2/s2w48m4_m256_n17408_n2_s512_card0.txt`,
  `results/k2/s2w48m4_m256_n17408_n2_s512_card1.txt`.

## s2 4-acc M=256 K=17408 is 108 us both cards (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_w48m4`. M=256 N=5120
  K=17408. Both cards. NT=2
  spin=512. Named clock 2800.
  Prior: square 37.4; s4 149;
  W8A8 226; napkin ~127.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 108.216/108.414
  vs square 37.4 vs s4 149 vs
  W8A8 226. Spread ~0.18%. ~2.89x
  square.

VERDICT -> New s2 4-acc M=256
  K=17408 floor 108 us pipe_host
  at 2800 both cards. Beats s4 149
  (~1.38x) and W8A8 226 (~2.09x).
  Qwen FFN s2 4-acc M=256 map is
  closed (37.4 / 110 / 108). Rank
  pipe_host.

Evidence: `results/k2/s2w48m4_m256_k17408_n2_s512_card0.txt`,
  `results/k2/s2w48m4_m256_k17408_n2_s512_card1.txt`.

## s2 4-acc M=64 pads to M=256 us (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_w48m4`. M=64 N=K=5120.
  Card0. NT=2 spin=512. Named
  clock 2800. Occupancy check.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64 pipe_host 37.152 vs 4x8 20
  vs M=256 37.4 vs W8A8 46.

VERDICT -> 4-acc at M=64 is 37 us
  pipe_host at 2800 card0, same
  class as M=256. Loses to 4x8 20
  (~1.86x). Stop 4-acc at M=64.
  One-card.

Evidence: `results/k2/s2w48m4_m64_n2_s512_card0.txt`.

## s2 4-acc NT=4 loses at M=256 (K2)

CONFIG -> backend `sycl+l0`, same
  `dpas_s2_w48m4`. M=256 N=K=5120.
  Card1. NT=4 unroll=4 spin=512.
  Named clock 2800.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256 pipe_host 307.201 vs NT=2
  37.4 vs W8A8 75.

VERDICT -> NT=4 is 307 us pipe_host
  at 2800 card1, ~8.2x NT=2. Stop
  NT=4 on this tile. One-card.

Evidence: `results/k2/s2w48m4_m256_n4_s512_card1.txt`.

## s2 4-acc A-db is a wash at M=256 (K2)

CONFIG -> backend `sycl+l0`,
  standalone `dpas_s2_w48m4db`.
  RC=8 4-acc k64 A-db wg 4x8 k128
  pack=4. M=256 N=K=5120. Both
  cards. NT=2 spin=512. Named
  clock 2800. Never E2M1. Prior:
  no-db 37.4; s4 A-db 51.9 tax.

RESULT -> ocloc 128x `dpas.8x8`
  rW:s2 rA:s2, grf 128. cosine=1.0
  max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host
  37.138/37.274 vs no-db 37.4.
  Spread ~0.37%.

VERDICT -> A-db is 37.2 us pipe_host
  at 2800 both cards, a wash vs
  no-db 37.4 (not s4's tax). Stop
  A-db on s2 4-acc. Floor stays
  37.4. Rank pipe_host.

Evidence: `results/k2/s2w48m4db_m256_n2_s512_card0.txt`,
  `results/k2/s2w48m4db_m256_n2_s512_card1.txt`,
  `results/k2/s2w48m4db_dpas_lines.txt`.

## Qwen3.8 GDN conv1d is ~115 us launch-bound (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. Qwen3.8-27B dims:
  key_dim=2048 value_dim=6144
  conv_k=4. Depthwise conv1d bf16.
  Both cards. No serve. Short
  kernels, cur 550-2800,
  throttle=0.

RESULT -> T=1/64/256 all ~115 us.
  T=1 q 125.3/117.1, k 114.7/115.8,
  v 116.8/119.4. GB/s 0.2 at T=1
  vs 55 at v T=256.

VERDICT -> Eager GDN conv1d is a
  ~115 us launch, not a GEMM.
  Three of these (q,k,v) already
  dwarf s2 decode 11.5. Rank us.

Evidence: `results/k7/conv1d_card0.txt`,
  `results/k7/conv1d_card1.txt`,
  `results/k7/INVENTORY.md`.

## Qwen3.8 GDN delta recurrent is 308 us (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. 48 v-heads, S 128x128
  bf16 (1.5 MiB/layer, 72 MiB x48).
  Eager bmm fused-recurrent. Both
  cards. No serve.

RESULT -> 306.575/309.042 us.
  Spread ~0.8%. ~10 GB/s.

VERDICT -> Eager delta is 308 us
  both cards, ~7x W8A8 44. State
  footprint is small; this path
  is the decode leftover after
  INT8 projections. Rank us.

Evidence: `results/k7/delta_recurrent_card0.txt`,
  `results/k7/delta_recurrent_card1.txt`.

## GDN q/v proj W8A8 is ~46 us decode (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. M=1 k=5120. q n=2048,
  v n=6144. int8_gemm_w8a8. Heat
  M=64 spin=512. Both cards. No
  serve.

RESULT -> v-proj 46.080/46.306 us
  cosine=1. q-proj 45.344/58.429
  (spread ~29%, clocks). vs square
  44 vs conv 115 vs delta 308.

VERDICT -> GDN projections sit in
  the W8A8 44 us class, not 3x N.
  Mixer (conv 115 + delta 308) is
  the leftover, not qkvz GEMM.
  Do not freeze q 45. Rank us.

Evidence: `results/k7/proj_q_w8a8_card0.txt`,
  `results/k7/proj_q_w8a8_card1.txt`,
  `results/k7/proj_v_w8a8_card0.txt`,
  `results/k7/proj_v_w8a8_card1.txt`.

## ESIMD GDN conv1d is ~4.4 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_conv1d` AOT
  `intel_gpu_bmg_g31`. Depthwise
  K=4 T=1 f16 with K-1 state.
  VL=16 wg=16. Both cards.
  spin=4000. Prior: eager ~115 us.

RESULT -> cosine=1.0 max_abs=0.
  C=2048 pipe_host 4.350/4.500
  event 1.456/1.799. cur 1700/
  1400 throttle=0. C=6144
  4.799/5.000 at 2250/2167.
  Spread ~3.4%. vs eager 115.

VERDICT -> Fused ESIMD conv1d is
  ~4.4 us pipe_host both cards,
  ~26x eager. Short kernel,
  clocks 1400-1700 not 2800. Do
  not freeze 4.4 as a 2800 floor.
  Rank pipe_host.

Evidence: `results/k7/esimd_conv1d_s4000_card0.txt`,
  `results/k7/esimd_conv1d_s4000_card1.txt`.

## ESIMD GDN delta is 7.1 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta` AOT
  `intel_gpu_bmg_g31`. 48 heads,
  S 128x128 f16. VL=16 wg=16.
  Both cards. spin=4000. Prior:
  eager 308 us.

RESULT -> cosine=1.0 max_abs=
  0.015625 (1 ulp f16) cosine_o=1
  max_abs_o=0. timed act=cur=2800
  throttle=0. pipe_host 7.028/
  7.093. 455/450 GB/s vs copy
  550. Spread ~0.9%.

VERDICT -> Fused ESIMD delta is
  7.1 us pipe_host both cards at
  2800, ~43x eager 308, near HBM.
  Mixer 4.4+7.1 ~11.5 us sits
  under W8A8 46; leftover moves
  to qkvz. Rank pipe_host.

Evidence: `results/k7/esimd_delta_s4000_card0.txt`,
  `results/k7/esimd_delta_s4000_card1.txt`.

## GDN o-proj W8A8 is ~47 us decode (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. M=1 n=5120 k=6144
  (value_dim -> H). int8_gemm_w8a8.
  Heat M=64 spin=512. Both cards.
  No serve.

RESULT -> cosine=1 ok=1.
  46.293/47.133 us. Spread ~1.8%.
  vs v-proj 46 vs square 44.

VERDICT -> o-proj is 46-47 us
  both cards, same W8A8 44-class
  as v-proj, not K-linear. Rank
  us.

Evidence: `results/k7/proj_o_w8a8_card0.txt`,
  `results/k7/proj_o_w8a8_card1.txt`.

## Fused qkv conv1d is 4.4-4.9 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_conv1d_qkv` AOT
  `intel_gpu_bmg_g31`. Packed
  q+k+v C=10240 K=4 T=1 f16.
  VL=16 wg=16. Both cards.
  spin=4000. Prior: one-arm ~4.4,
  3x ~13.2.

RESULT -> fused cosine=1.0
  max_abs=0. pipe_host 4.856/
  4.437. cur 1183/2800. Spread
  ~9% (clocks). trio pipe_host
  14.233/13.449 at 2800 both.

VERDICT -> One launch covers
  q+k+v in the 4.4-4.9 us class
  both cards, ~3x the trio ~13.8.
  Do not freeze 4.44 as 2800.
  Launch bound. Rank pipe_host.

Evidence: `results/k7/esimd_conv1d_qkv_s4000_card0.txt`,
  `results/k7/esimd_conv1d_qkv_s4000_card1.txt`.

## Packed qkv W8A8 is 96 us (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. M=1 n=10240 k=5120
  (q+k+v). int8_gemm_w8a8. Heat
  M=64 spin=512. Both cards. No
  serve.

RESULT -> cosine=1 ok=1.
  95.783/95.481 us. Spread ~0.3%.
  vs 3x 46 ~138 vs v-proj 46.

VERDICT -> Packed qkv is 96 us
  both cards, ~1.44x three
  sequential GEMMs, ~2.08x
  v-proj. Not the 46 us launch
  class. Rank us.

Evidence: `results/k7/proj_qkv_w8a8_card0.txt`,
  `results/k7/proj_qkv_w8a8_card1.txt`.

## ESIMD mixer conv+delta is 8.2-8.7 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_mixer` AOT
  `intel_gpu_bmg_g31`. Packed
  conv C=10240 then 48-head
  delta, q/k repeat 16->48.
  Both cards. spin=4000. Prior:
  4.4+7.1 = 11.5.

RESULT -> cosine=1.0 max_abs=
  1.22e-4 cosine_o=1 max_abs_o=0.
  timed act=cur=2800 throttle=0.
  pipe_host 8.746/8.229. Spread
  ~6.3%.

VERDICT -> Mixer is 8.2-8.7 us
  pipe_host both cards at 2800,
  ~1.4x the 11.5 sum. Spread >5%.
  Do not freeze 8.23. Conv hides
  under delta. Rank pipe_host.

Evidence: `results/k7/esimd_mixer_s4000_card0.txt`,
  `results/k7/esimd_mixer_s4000_card1.txt`.

## Packed qkv W8A8 M=64 is 140 us (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. M=64 n=10240 k=5120.
  int8_gemm_w8a8. Heat M=64
  spin=512. Both cards. No serve.

RESULT -> cosine=1 ok=1.
  142.053/138.079 us. Spread
  ~2.9%. vs M=1 96 vs 3x 46 ~138.

VERDICT -> Packed qkv M=64 is
  138-142 us both cards, wash vs
  3x 46. Rank us.

Evidence: `results/k7/proj_qkv_w8a8_m64_card0.txt`,
  `results/k7/proj_qkv_w8a8_m64_card1.txt`.

## ESIMD conv1d T=64 is 10.1 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_conv1d_t` AOT
  `intel_gpu_bmg_g31`. C=2048
  T=64 K=4 f16 causal. Both
  cards. spin=4000. Prior: eager
  ~115, decode T=1 4.4.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 10.130/10.161. Spread
  ~0.3%.

VERDICT -> Prefill conv T=64 is
  10.1 us pipe_host both cards at
  2800, ~11x eager 115, ~2.3x
  decode T=1. Not T-linear. Rank
  pipe_host.

Evidence: `results/k7/esimd_conv1d_t64_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t64_s4000_card1.txt`.

## ESIMD conv1d T=256 is 37.7 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_conv1d_t`. C=2048
  T=256 K=4 f16. Both cards.
  spin=4000. Prior: T=64 10.1,
  eager ~115.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 37.607/37.811. Spread
  ~0.5%.

VERDICT -> Prefill conv T=256 is
  37.7 us pipe_host both cards at
  2800, ~3.72x T=64, ~3.1x eager
  115. Near T-linear. Rank
  pipe_host.

Evidence: `results/k7/esimd_conv1d_t256_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t256_s4000_card1.txt`.

## ESIMD conv1d T=256 C=6144 is 38.0 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_conv1d_t`. C=6144
  T=256 K=4 f16 (v-channels).
  Both cards. spin=4000. Prior:
  C=2048 37.7, napkin 3x ~113.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 38.010/38.032. Spread
  ~0.06%.

VERDICT -> v-conv T=256 C=6144 is
  38.0 us pipe_host both cards at
  2800, wash vs q/k C=2048 37.7
  not 3x. Occupancy. Rank
  pipe_host.

Evidence: `results/k7/esimd_conv1d_t256_c6144_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t256_c6144_s4000_card1.txt`.

## Packed qkv W8A8 M=256 is 164 us (K7)

CONFIG -> backend `pytorch-xpu` on
  `sycl+l0`. M=256 n=10240 k=5120.
  int8_gemm_w8a8. Heat M=64
  spin=512. Both cards. No serve.

RESULT -> cosine=1 ok=1.
  163.739/163.539 us. Spread
  ~0.12%. vs M=64 140 vs square
  75 vs 3x 75 ~225.

VERDICT -> Packed qkv M=256 is
  164 us both cards, ~1.17x M=64,
  ~1.37x 3 sequential 225. Rank
  us.

Evidence: `results/k7/proj_qkv_w8a8_m256_card0.txt`,
  `results/k7/proj_qkv_w8a8_m256_card1.txt`.

## ESIMD delta T=64 is 265-271 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_t` AOT
  `intel_gpu_bmg_g31`. T=64 nv=48
  dv=128 dk=128 f16. S float in
  GRF across T. L2-norm q/k.
  Both cards. spin=4000. Prior:
  decode T=1 7.1, napkin 64x ~454.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  271.249/264.906. Spread ~2.4%.
  timed act 2583/2633-2650
  cur=2800 throttle=1 both.

VERDICT -> Prefill delta T=64 is
  265-271 us pipe_host both cards,
  ~37x decode T=1 not 64x. Not HBM
  (24 GB/s). Do not freeze 265 as
  2800. Rank pipe_host.

Evidence: `results/k7/esimd_delta_t64_s4000_card0.txt`,
  `results/k7/esimd_delta_t64_s4000_card1.txt`.

## ESIMD delta T=256 is 1100-1109 us (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_t`. T=256 nv=48
  dv=128 dk=128 f16. Both cards.
  spin=4000. Prior: T=64 265-271,
  napkin 4x ~1060.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  1109.372/1099.419. Spread ~0.9%.
  timed act 2617/2650 cur=2800
  throttle=1 both.

VERDICT -> Prefill delta T=256 is
  1100-1109 us pipe_host both
  cards, ~4.1x T=64 near T-linear.
  Prefill GDN leftover vs packed
  qkv 164 and conv 38. Do not
  freeze 1100 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_t256_s4000_card0.txt`,
  `results/k7/esimd_delta_t256_s4000_card1.txt`.

## ESIMD mixer T=64 is 395-399 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_mixer_t` AOT
  `intel_gpu_bmg_g31`. T=64 packed
  conv C=10240 then delta, q/k
  repeat 16->48, L2-norm q/k.
  Both cards. spin=4000. Prior:
  decode mixer 8.2, sequential
  conv 10.1 + delta 265 ~275.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  399.311/394.542. Spread ~1.2%.
  timed act 2633/2667 cur=2800
  throttle=1 both.

VERDICT -> Prefill mixer T=64 is
  395-399 us pipe_host both cards,
  ~1.45x sequential ~275. Packed
  layout + L2 tax. Stop two-kernel
  packed mixer at prefill vs
  sequential conv+delta. Do not
  freeze 395 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_mixer_t64_s4000_card0.txt`,
  `results/k7/esimd_mixer_t64_s4000_card1.txt`.

## ESIMD conv1d T=64 C=6144 is 10.2-10.4 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_conv1d_t`. C=6144
  T=64 K=4 f16 (v-channels).
  Both cards. spin=4000. Prior:
  C=2048 10.1, napkin 3x ~30.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 10.449/10.231. Spread
  ~2.1%.

VERDICT -> v-conv T=64 C=6144 is
  10.2-10.4 us pipe_host both
  cards at 2800, wash vs q/k
  C=2048 10.1 not 3x. Occupancy.
  Rank pipe_host.

Evidence: `results/k7/esimd_conv1d_t64_c6144_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t64_c6144_s4000_card1.txt`.

## ESIMD conv1d T=64 C=10240 is 10.5-10.7 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_conv1d_t`. C=10240
  T=64 K=4 f16 (packed qkv).
  Both cards. spin=4000. Prior:
  C=2048 10.1, C=6144 10.2-10.4,
  napkin 5x ~50.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 10.667/10.535. Spread
  ~1.3%.

VERDICT -> Packed-qkv conv T=64
  C=10240 is 10.5-10.7 us
  pipe_host both cards at 2800,
  wash vs C=2048 10.1 not 5x.
  Occupancy. Rank pipe_host.

Evidence: `results/k7/esimd_conv1d_t64_c10240_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t64_c10240_s4000_card1.txt`.

## ESIMD conv1d T=256 C=10240 is 40.7-40.8 us at 2800 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_conv1d_t`. C=10240
  T=256 K=4 f16 (packed qkv).
  Both cards. spin=4000. Prior:
  C=2048 37.7, C=6144 38.0,
  napkin 5x ~189.

RESULT -> cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  pipe_host 40.797/40.742. Spread
  ~0.13%.

VERDICT -> Packed-qkv conv T=256
  C=10240 is 40.7-40.8 us
  pipe_host both cards at 2800,
  ~1.07x C=6144 38.0 not 5x.
  Occupancy. Rank pipe_host.

Evidence: `results/k7/esimd_conv1d_t256_c10240_s4000_card0.txt`,
  `results/k7/esimd_conv1d_t256_c10240_s4000_card1.txt`.

## ESIMD chunk/WY C=16 loses to fused delta T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_chunk` AOT
  `intel_gpu_bmg_g31`. T=256 C=16
  nv=48 dv=128 dk=128 f16. FLA
  KK^T + (I+L)^{-1} + w/u + h/o.
  Card1. spin=4000. Prior: fused
  1100-1109 throttle=1.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  3210.272 event 3199.857. timed
  act=cur=2800 throttle=0.

VERDICT -> Chunk/WY C=16 is 3210
  us pipe_host card1 at 2800,
  ~2.92x fused 1100. Numeric
  closed. Stop C=16 vs fused.
  Rank pipe_host.

Evidence: `results/k7/esimd_delta_chunk_t256_s4000_card1.txt`.

## ESIMD fused delta T=256 stays throttle=1 (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_t`. T=256 nv=48
  card1 hold retry. spin=4000.
  Prior: fv/fw 1100-1109
  throttle=1.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  1085.686. timed act=2683
  cur=2800 throttle=1.

VERDICT -> Hold retry still
  throttle=1. Fused T=256 is
  1086 us pipe_host card1, not
  2800. Prefill leftover. Do not
  freeze 1086 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_t256_hold_s4000_card1.txt`.

## ESIMD chunk/WY C=64 loses worse than C=16 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_chunk64`
  AOT `intel_gpu_bmg_g31`. T=256
  C=64. Card0. spin=0. Prior:
  C=16 3210, fused 1086.

RESULT -> cosine=1.0 max_abs
  3.1e-5 / 2.4e-4 ok=1. pipe_host
  95419.883. timed act=cur=2800
  throttle=0.

VERDICT -> Chunk/WY C=64 is
  95420 us pipe_host card0 at
  2800, ~88x fused 1086, ~30x
  C=16. Stop C=64. Stop this WY
  path vs fused. Rank pipe_host.

Evidence: `results/k7/esimd_delta_chunk64_t256_s0_card0.txt`.

## ESIMD fused delta T=256 SLM-K is 847-858 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmk` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  serial S, k/q in SLM. Both
  cards. card0 spin=0, card1
  spin=4000. Prior: fused 1086.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  847.280 / 858.215. Spread
  ~1.3%. throttle=1 both. act
  2767-2783 / 2700-2717.

VERDICT -> SLM-K T=256 is 847-858
  us pipe_host both cards, ~1.27x
  fused 1086. New leftover class.
  Do not freeze 847 as 2800.
  Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmk_t256_s0_card0.txt`,
  `results/k7/esimd_delta_slmk_t256_s4000_card1.txt`.

## ESIMD fused delta T=256 row-block rb=4 is 1034 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_rowb` AOT
  `intel_gpu_bmg_g31`. T=256 rb=4.
  Card1. spin=0. Prior: fused 1086,
  SLM-K 847.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  1034.092. timed act=cur=2800
  throttle=0.

VERDICT -> Row-block rb=4 is 1034
  us pipe_host card1 at 2800,
  ~1.05x fused 1086, loses to
  SLM-K 847. One-card. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_rowb_t256_s0_card1.txt`.

## ESIMD fused delta T=256 row-block rb=8 loses (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_rowb8` AOT
  `intel_gpu_bmg_g31`. T=256 rb=8.
  Card0. spin=0. Prior: rb=4 1034,
  SLM-K 847.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  2060.439. timed act=cur=2800
  throttle=0.

VERDICT -> Row-block rb=8 is 2060
  us pipe_host card0 at 2800,
  ~2x rb=4, ~2.4x SLM-K.
  Occupancy. Stop rb=8 vs rb=4
  and SLM-K. Rank pipe_host.

Evidence: `results/k7/esimd_delta_rowb8_t256_s0_card0.txt`.

## ESIMD SLM-K+rb=4 loses to SLM-K (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmk_rb4`
  AOT `intel_gpu_bmg_g31`. T=256
  blk=16 rb=4. Card0. spin=0.
  Prior: SLM-K 847, rb=4 1034.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  998.817. timed act=cur=2800
  throttle=0.

VERDICT -> SLM-K+rb=4 is 999 us
  pipe_host card0 at 2800, ~1.18x
  SLM-K 847. Occupancy. Stop
  combine vs SLM-K. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmk_rb4_t256_s0_card0.txt`.

## ESIMD SLM-K blk=32 is 832-862 us (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmk32` AOT
  `intel_gpu_bmg_g31`. T=256 blk=32.
  Both cards. card1 spin=0, card0
  spin=4000. Prior: blk=16 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  831.953 / 862.027. Spread
  ~3.6%. throttle=1 both. act
  2750-2783 / 2650.

VERDICT -> SLM-K blk=32 is 832-862
  us pipe_host both cards,
  throttle=1. Wash vs blk=16
  847-858 (clocks). Do not freeze
  832 as 2800. Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmk32_t256_s0_card1.txt`,
  `results/k7/esimd_delta_slmk32_t256_s4000_card0.txt`.

## ESIMD SLM-K blk=64 washes vs blk=32 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmk64` AOT
  `intel_gpu_bmg_g31`. T=256 blk=64.
  Card1. spin=0. Prior: blk=32
  832-862.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  834.985. timed act 2800-2733
  throttle=1 at end.

VERDICT -> SLM-K blk=64 is 835 us
  pipe_host card1, wash vs blk=32
  832. Stop larger blk vs 16/32.
  Do not freeze 835. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmk64_t256_s0_card1.txt`.

## ESIMD SLM-K T=64 is 214-218 us (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_slmk`. T=64
  blk=16. Both cards. card0
  spin=0, card1 spin=4000. Prior:
  fused T=64 265-271 throttle=1.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  214.083 / 217.843. Spread
  ~1.8%. card0 act=cur=2800
  throttle=0. card1 act=2750
  throttle=1.

VERDICT -> SLM-K T=64 is 214-218
  us pipe_host both cards, ~1.23x
  fused 265, T-linear vs 847.
  New T=64 leftover class. Do
  not freeze 214 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmk_t64_s0_card0.txt`,
  `results/k7/esimd_delta_slmk_t64_s4000_card1.txt`.

## ESIMD SLM a/b washes vs SLM-K T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmab` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  k/q/a/b in SLM. Card1. spin=0.
  Prior: SLM-K 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  853.663. timed act 2783-2750
  throttle=1.

VERDICT -> SLM a/b is 854 us
  pipe_host card1, wash vs SLM-K
  847-858. Stop a/b SLM vs
  k/q-only. Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmab_t256_s0_card1.txt`.

## ESIMD v-prefetch loses to SLM-K T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmv` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  k/q SLM, 16 v in GRF. Card0.
  spin=0. Prior: SLM-K 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  873.078. timed act 2783-2750
  throttle=1.

VERDICT -> v-prefetch is 873 us
  pipe_host card0, ~1.03x SLM-K
  847. Stop v-prefetch vs SLM-K.
  Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmv_t256_s0_card0.txt`.

## ESIMD SLM-K T=1 blk=1 loses to fused decode (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmk1` AOT
  `intel_gpu_bmg_g31`. T=1 blk=1
  k/q SLM, lid<1 fill. Card0.
  spin=4000. Prior: fused 7.1 at
  2800.

RESULT -> cosine=1.0 max_abs
  1.2e-4 / 2.4e-4 ok=1. pipe_host
  8.149 event 7.836. 392 GB/s.
  timed act=cur=2800 throttle=0.

VERDICT -> SLM-K T=1 is 8.15 us
  pipe_host card0 at 2800, ~1.16x
  fused 7.1. Event wash vs fused
  7.825. Barrier tax. Stop SLM-K
  vs fused at decode. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmk1_t1_s4000_card0.txt`.

## ESIMD SLM-K inner unroll washes vs T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmku` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  inner tt unrolled. Card1.
  spin=0. Prior: SLM-K 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  856.296. timed act 2800-2783
  throttle=1.

VERDICT -> Inner unroll T=256 is
  856 us pipe_host card1, wash vs
  SLM-K 847-858. Stop inner
  unroll vs SLM-K. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmku_t256_s0_card1.txt`.

## ESIMD SLM f32 k/q loses to SLM-K T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmf32` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  k/q converted to f32 in SLM.
  Card0. spin=0. Prior: SLM-K
  847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  867.995. timed act 2800-2767
  throttle=1.

VERDICT -> SLM f32 k/q is 868 us
  pipe_host card0, ~1.02x SLM-K
  847. Stop f32 SLM vs half.
  Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmf32_t256_s0_card0.txt`.

## ESIMD SLM double-buffer washes vs T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmdb` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  ping-pong k/q SLM. Card1.
  spin=0. Prior: SLM-K 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  842.973. timed act 2783-2733
  throttle=1.

VERDICT -> SLM db is 843 us
  pipe_host card1, throttle=1,
  wash vs SLM-K 847-858. Do not
  freeze 843 as 2800. Stop
  double-buffer vs SLM-K. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmdb_t256_s0_card1.txt`.

## ESIMD SLM-K tree hsum is 426-477 us T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmh` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  esimd::reduce hsum. Both cards.
  card0 spin=0, card1 spin=4000.
  Prior: SLM-K 847-858.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  425.689 / 477.332. Spread
  ~12%. throttle=1 both. act
  2800-2700 / 2417.

VERDICT -> Tree hsum T=256 is
  426-477 us pipe_host both
  cards, throttle=1. Clock
  spread, not a kernel split.
  ~1.99x SLM-K 847 at card0.
  New leftover class. Do not
  freeze 426 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmh_t256_s0_card0.txt`,
  `results/k7/esimd_delta_slmh_t256_s4000_card1.txt`.

## ESIMD SLM-K T=16 is 58 us both cards (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_slmk`. T=16
  blk=16. Both cards. card1
  spin=0, card0 spin=4000.
  Prior: T=256 847, T=64 214.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  58.130 / 58.600. Spread ~0.8%.
  card0 timed act=2767 cur=2800
  throttle=1. card1 ramped 550
  to 2800.

VERDICT -> SLM-K T=16 is 58 us
  pipe_host both cards, near
  T-linear 53. throttle=1 on
  hold. Do not freeze 58 as
  2800. Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmk_t16_s0_card1.txt`,
  `results/k7/esimd_delta_slmk_t16_s4000_card0.txt`.

## ESIMD tree hsum T=64 is 109-125 us (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_slmh`. T=64
  blk=16. Both cards. card0
  spin=0, card1 spin=4000.
  Prior: SLM-K T=64 214 at 2800.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  108.823 / 124.803. Spread
  ~15%. card0 act=cur=2800
  throttle=0. card1 act=2450
  throttle=1.

VERDICT -> Tree hsum T=64 is
  109-125 us pipe_host both
  cards. Clock spread, not a
  kernel split. ~1.97x SLM-K 214
  at card0 2800. New leftover
  class. Do not freeze 109 as
  2800. Rank pipe_host.

Evidence: `results/k7/esimd_delta_slmh_t64_s0_card0.txt`,
  `results/k7/esimd_delta_slmh_t64_s4000_card1.txt`.

## ESIMD tree hsum T=1 does not replace fused 7.1 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_h` AOT
  `intel_gpu_bmg_g31`. T=1
  esimd::reduce hsum. Card1.
  spin=4000. Prior: fused 7.1 at
  2800.

RESULT -> cosine=1.0 max_abs
  0.0625 / 2 ok=1. pipe_host
  6.085 event 7.237. 525 GB/s.
  timed act=cur=2800 throttle=0.
  fused max_abs 0.015625 /
  max_abs_o=0.

VERDICT -> Tree hsum T=1 is 6.09
  us pipe_host card1 at 2800,
  ~1.16x fused 7.1. Numeric
  looser (max_abs_o=2). Do not
  replace fused 7.1. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_h_s4000_card1.txt`.

## ESIMD tree hsum T=16 is 34 us both cards (K7)

CONFIG -> backend `sycl+l0`,
  same `gdn_delta_slmh`. T=16
  blk=16. Both cards. spin=4000.
  Prior: SLM-K T=16 58.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  33.801 / 33.683. Spread ~0.3%.
  act 2583 / 2633 cur=2800
  throttle=1 both.

VERDICT -> Tree hsum T=16 is 34
  us pipe_host both cards, ~1.72x
  SLM-K 58. throttle=1. Do not
  freeze 34 as 2800. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmh_t16_s4000_card0.txt`,
  `results/k7/esimd_delta_slmh_t16_s4000_card1.txt`.

## ESIMD tile-fused reduce is 260 us T=256 (K7)

CONFIG -> backend `sycl+l0`,
  standalone `gdn_delta_slmht` AOT
  `intel_gpu_bmg_g31`. T=256 blk=16
  one 16-wide acc then reduce.
  Card0. spin=0. Prior: tree
  hsum 426-477.

RESULT -> cosine=1.0 max_abs
  1.5e-5 / 2.4e-4 ok=1. pipe_host
  260.132. timed act=2600
  cur=2800 throttle=0.

VERDICT -> Tile-fused reduce
  T=256 is 260 us pipe_host
  card0, ~1.64x tree hsum 426.
  New leftover class. Do not
  freeze 260 as 2800. One-card.
  Sibling before promote. Rank
  pipe_host.

Evidence: `results/k7/esimd_delta_slmht_t256_s0_card0.txt`.

## K5 producer+GEMM N=17408 is 155 us both cards (K5)

CONFIG -> backend `sycl+l0`, `dpas_s8_prod`
  WG-256 RMSNorm-quant then RC=4 s8 GEMM.
  NT=2 spin=4000. M=1 N=17408 K=5120.
  Both cards. Named clock 2800.

RESULT -> timed act=cur=2800 throttle=0.
  card0 prod 10.846 gemm 143.529 pair
  154.505 pipe_host 156.354. card1 prod
  10.818 gemm 142.625 pair 153.581
  pipe_host 154.033 vs square 44 vs s8
  GEMM 141.6 vs W8A8 158.1 vs napkin
  151. cosine=1.0 max_abs=0. M=4 tracks.
  Spread ~1.5%. Extra ~11 us over GEMM.

VERDICT -> New producer+GEMM N=17408
  floor 155 us pipe_host at 2800 both
  cards. ~3.52x square, N-linear. The
  producer tax stays ~11 us (K=5120),
  not N. Beats W8A8 158.1. Numeric
  closed. Rank pipe_host.

Evidence: `results/k5/prod_n17408_n2_s4000_card0.txt`,
  `results/k5/prod_n17408_n2_s4000_card1.txt`.

## K5 producer+GEMM K=17408 is 294 us both cards (K5)

CONFIG -> backend `sycl+l0`, same
  `dpas_s8_prod`. NT=2 spin=4000.
  M=1 N=5120 K=17408. Both cards.
  Named clock 2800. Napkin ~297.

RESULT -> timed act=cur=2800 throttle=0.
  card0 prod 33.102 gemm 260.284 pair
  293.518 pipe_host 294.411. card1 prod
  33.099 gemm 261.068 pair 294.305
  pipe_host 294.453 vs square 44 vs
  N-wide 155 vs s8 GEMM 261.6 vs W8A8
  155.3 vs napkin 297. cosine=0.999995
  max_abs=0.064 ok=1. M=4 tracks.
  Spread ~0.01%. Extra ~33 us over GEMM.

VERDICT -> New producer+GEMM K=17408
  floor 294 us pipe_host at 2800 both
  cards. ~6.68x square. Producer tax is
  K-linear (~33 us); GEMM matches s8
  261.6. Loses to W8A8 155.3 (~1.90x).
  Qwen FFN producer decode map is
  closed. Rank pipe_host.

Evidence: `results/k5/prod_k17408_n2_s4000_card0.txt`,
  `results/k5/prod_k17408_n2_s4000_card1.txt`.

## 27B NVFP4 persist-s8 is 29.0 GiB weights-only (K6)

CONFIG -> CPU envelope of
  `qwen3.8-27b/nvfp4-radixark` 3 shards.
  MXFP4 counted from `hf_quant_config.json`.

RESULT -> U8 packed 8.561 GiB (193 tensors), s8
  unpack 17.122 GiB, F8 7.789 GiB, bf16 4.066 GiB.
  resident 4-bit+rest 20.416 GiB fits 30.3.
  persist-s8+rest 28.977 GiB leaves ~1.3 GiB.
  Layers: NVFP4 193 all g16, MXFP4 0, FP8 208.

VERDICT -> Resident 4-bit is the 27B VRAM path.
  Load-time s8 is the 8B-class path. MXFP4 is a
  labeled third format and is not this checkpoint.

Evidence: `results/k6/persist_vram.txt`,
  `results/k6/g16_scale_landmine.txt`.

## Untuned 8x16 DPAS does not beat 45 us W8A8 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8` / `dpas_s4`,
  same RC=8 N=16 tile. Qwen3.8-ish N,K=5120. M=8 is padded
  decode. Both cards. oneDNN W8A8 GEMM-only floor: 42-46 us
  at M=1, 46-49 at M=64, 74-76 at M=256.

RESULT -> Numeric max_abs=0. M=64 5120: s8 274-373 us, s4
  387-393 us. M=256: s8 892-1064 us, s4 691 us. Padded M=8
  s4 swings 34-120 us (repeat 64-87); s8 56-230 us.

VERDICT -> This tile is not the incumbent. Do not call the
  34 us M=8 s4 a beat of 45 us M=1: clocks disagree and the
  work is 8 rows. Hand ESIMD must change schedule (block, GRF,
  prefetch) before it can beat oneDNN in us.

Evidence: `results/k2/SUMMARY.md`.

## A-reuse NT=2/4 does not close the W8A8 gap (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_block`. Load A
  once per K-chunk, dpas across NT=2 (8x32) or NT=4 (8x64) B
  panels. Transformed LSC B. Both cards. Prior: wider N wins
  by cutting A traffic.

RESULT -> max_abs=0. M=64 5120: NT=2 269-352 us, NT=4 318-474 us
  vs 8x16 274-373 us vs oneDNN 46-49 us. M=256: NT=4 694-876 us
  vs 8x16 892-1064 vs oneDNN 75 us.

VERDICT -> The napkin is weak here. A-reuse is not the 6x.
  oneDNN still owns the floor. Next levers: SLM pack, GRF 256,
  prefetch, or steal the W8A8 ngen schedule (K1 dump).

Evidence: `results/k2/SUMMARY.md`.

## ESIMD RC=4 is dpas.8x4 and does not beat 45 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_rc4`, RepeatCount=4,
  Transformed B. Both cards. Prior: oneDNN M=1 uses dpas.8x4.

RESULT -> IGA `dpas.8x4 (16|M0) ... r14:b r11.0:b`. max_abs=0.
  Padded decode M=4 5120: 77-168 us vs W8A8 M=1 42-46 us. M=64:
  614-894 us vs RC=8 274-373 vs W8A8 46-49.

VERDICT -> RC=4 compiles and matches the stolen encoding. It is
  not the 45 us kernel. Prefill prefers RC=8 on this tile.
  `grf_size<256>` and `pvc:large` left zebin `grf_count: 128`.

Evidence: `results/k2/SUMMARY.md`, `results/k2/rc4_dpas_line.txt`.

## SLM A share plus per-K barrier loses at decode (K2)

CONFIG -> backend `sycl+l0`, `dpas_s8_slm`, RC=4, WG=16, A 4x32
  in SLM, Transformed B. Both cards. Prior: W8A8 M=1 wg 8x2 + SLM
  is why it is 45 us.

RESULT -> max_abs=0. M=4 5120: 372-463 us vs RC=4 no-SLM 77-168
  vs W8A8 M=1 42-46. M=64/256 also slower than the no-SLM RC=4
  tile.

VERDICT -> Copying "use SLM" without the rest of the ngen
  schedule (k64, unrolled 64 dpas, wg 8x2 layout) is a miss.
  Barrier-per-K-chunk A broadcast is not the 45 us kernel.

Evidence: `results/k2/SUMMARY.md`.

## k64 blocking with 4-16 dpas.8x4 is not the 45 us kernel (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_k64`, RC=4,
  k64=two K=32 DPAS per step, NT=2/4 A-reuse, no SLM.
  Both cards. Prior: W8A8 M=1 ngen is 64x `dpas.8x4` k64.

RESULT -> IGA `dpas.8x4 (16|M0) ... :b :b`. zebin `grf_count
  128`, `has_dpas true`. NT=2 binary has 4 static dpas (K
  loop remains). NT=4 has 16. max_abs=0. M=4 5120: 92-396 us
  vs RC=4 no-SLM 77-168 vs W8A8 M=1 42-46. M=64: 293-728 vs
  W8A8 46-49. M=256: 599-962 vs W8A8 74-76. First pair started
  D3hot/2800 and was the slow M=4; warm swap was 92-111.

VERDICT -> The ngen k64 *blocking* is not enough. IGC kept
  a K loop; this is not 64 unrolled dpas. Floor stays 45 us.
  Clocks explain the us spread; do not freeze 92 us.

Evidence: `results/k2/SUMMARY.md`, `results/k2/k64_dpas_lines.txt`.

## 64 static dpas.8x4 lights and does not beat 45 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_u64`, RC=4,
  k64 steps unrolled so 2*NT*UNROLL=64 dpas. NT=4 U=8 and
  NT=2 U=16. Both cards. Prior: ngen M=1 is 64x `dpas.8x4`.

RESULT -> IGA 64x `dpas.8x4 (16|M0) ... :b :b` on both NT
  kernels. zebin `grf_count 128`. max_abs=0. M=4 5120: NT=2
  53-69 us warm vs NT=4 107-316 (D3hot first 316) vs W8A8
  M=1 42-46. M=64: 314-570 vs 46-49. M=256: 594-1198 vs
  74-76.

VERDICT -> Matching ngen's dpas *count* is not the 45 us
  kernel. Warm NT=2 is the closest decode so far; do not
  freeze 53 us. Floor stays 45 us. Remaining steal is wg
  8x2 / ska / prefetch, not more unroll.

Evidence: `results/k2/SUMMARY.md`, `results/k2/u64_dpas_lines.txt`.

## Prefetch-before-load on 64 dpas is not a 45 us beat (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_pf`, same
  64x `dpas.8x4` tile as u64 plus `lsc_prefetch_2d` of next
  k64 A/B (cached/cached). Both cards. Prior: ngen M=1 `ff`.

RESULT -> IGA null-dest `load_block2d.ugm.d8 ... rd:0` plus
  64x `dpas.8x4`. GRF 128. max_abs=0. M=4 NT=2: 83-208 us vs
  u64 no-pf 53-69 vs W8A8 M=1 42-46. M=64: 229-677 vs u64
  314-570. First pair D3hot/2800 was the slow M=4.

VERDICT -> The ff encoding landed. Issuing prefetch *before*
  the current loads taxes decode vs no-pf. Floor stays 45 us.
  Do not freeze 229 us M=64 (clocks 229 vs 530). Next steal is
  ngen overlap (prefetch during dpas), not more pre-load sends.

Evidence: `results/k2/SUMMARY.md`, `results/k2/pf_dpas_lines.txt`.

## Overlapped prefetch on 64 dpas is not a 45 us beat (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_ov`. Same
  64x `dpas.8x4` as u64. Prologue `lsc_prefetch_2d` of k=0,
  then load, first dpas, then ff of next k64 (ngen order).
  Both cards. Prior: pf-before-load was slower than no-pf.

RESULT -> IGA 64x `dpas.8x4` plus null-dest
  `load_block2d.ugm.d8 ... rd:0`. GRF 128. ff sends 34/18
  (unroll*2+prologue) vs pf 131/99. max_abs=0. M=4 NT=2:
  100-264 us vs u64 no-pf 53-69 vs pf 83-208 vs W8A8 M=1
  42-46. M=64: 350-970 vs u64 314-570. Warm NT=2 was 100 us
  at 2083 MHz; first NT=4 was D3hot/2800 at 371 us.

VERDICT -> Moving ff after the first dpas (and prologue
  prefetch) is not the 45 us kernel. Decode still loses to
  no-pf u64. Do not freeze 100 us. Floor stays 45 us.
  Remaining steal is ngen wg 8x2 / ska double-buffer loads,
  not more ff micro-moves.

Evidence: `results/k2/SUMMARY.md`, `results/k2/ov_dpas_lines.txt`.

## A double-buffer on 64 dpas is not a 45 us beat (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_ska`. Same
  64x `dpas.8x4` as u64. Prologue load A[k=0], then load
  A[k+64] before dpas of current A (software pipeline, no
  ff). Both cards. Prior: ff overlap lost to no-pf u64.

RESULT -> IGA 64x `dpas.8x4`, GRF 128, ff=0. Extra A d8
  loads (34/18) mixed with dpas. max_abs=0. Warm 2600-2800
  MHz. M=4 NT=2: 79-81 us vs u64 no-pf 53-69 vs ov-ff
  100-264 vs W8A8 M=1 42-46. M=64: 271-650 vs u64 314-570.
  M=256: 965-1100 vs W8A8 74-76.

VERDICT -> Real next-A loads in GRF are not the 45 us
  kernel. Warm decode still loses to no-pf u64. Do not
  freeze 79 us. Floor stays 45 us. Remaining steal is ngen
  wg 8x2 2D launch, not another K-pipe.

Evidence: `results/k2/SUMMARY.md`, `results/k2/ska_dpas_lines.txt`.

## ngen wg 8x2 2D launch is not a 45 us beat (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_wg`. Same
  64x `dpas.8x4` as u64. `nd_range<2>` local {8,2}=(N,M),
  no SLM, no extra ff. Both cards. Prior: ngen M=1 is wg 8x2.

RESULT -> IGA 64x `dpas.8x4`, GRF 128, no barrier. max_abs=0.
  M=4 NT=2: 69-225 us vs 1D u64 53-69 vs W8A8 M=1 42-46.
  Warm 69 us was card1 after NT=4 at 1550 MHz. M=4 NT=4:
  122-316. M=64: 468-963 vs u64 314-570. Half the WG idles
  at M=4 (m_blocks=1 padded to 2).

VERDICT -> Relabeling the 16-thread WG as 8x2 (N,M) is not
  the 45 us kernel. 1D u64 still owns the hand-tile decode
  floor. Do not freeze 69 us. Floor stays 45 us. Next is
  8x2 along N (no idle) or SLM+64 dpas as one ngen bundle.

Evidence: `results/k2/SUMMARY.md`, `results/k2/wg_dpas_lines.txt`.

## 8x2 along N beats 1D u64 decode, not 45 us W8A8 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_wgn`. Same
  64x `dpas.8x4` as u64. `nd_range<2>` local {8,2} both on
  N (16 live N-groups). M is group(1). No SLM. Both cards.
  Prior: (N,M) 8x2 idled half the WG at M=4.

RESULT -> IGA 64x `dpas.8x4`, GRF 128, no barrier. max_abs=0.
  Warm ~2800 MHz. M=4 NT=2: 47-50 us vs 1D u64 53-69 vs
  (N,M) 8x2 69 vs W8A8 M=1 42-46. NT=4 M=4: 73 us both
  cards. M=64 NT=4: 304-317 vs u64 314-570.

VERDICT -> Mapping 8x2 onto N (no idle) is a real decode
  us win vs 1D local=16 at the same M=4 pad. It is not a
  beat of 45 us M=1 W8A8. Do not freeze 47 us. That 47-50
  was later shown not held-2800; the 2800 floor is ~36 us
  (see batched-spin finding). Remaining then: ngen SLM.

Evidence: `results/k2/SUMMARY.md`, `results/k2/wgn_dpas_lines.txt`.

## SLM A-broadcast plus 64 dpas is not a 45 us beat (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_slm64`. Same
  8x2-along-N 64x `dpas.8x4` as wgn. lid0 stores A[k64] to
  SLM, two barriers per k64. Both cards. Prior: ngen M=1
  bundles SLM + 64 dpas + wg 8x2; old SLM-per-k32 lost.

RESULT -> IGA 64x `dpas.8x4`, GRF 128, `slm_size` 1024,
  unrolled barriers. max_abs=0. M=4 NT=2: 101-140 us vs
  wgn no-SLM 47-50 vs old SLM 372-463 vs W8A8 M=1 42-46.
  NT=4 M=4: 111-114. M=64: 422-634 vs wgn 304-611.

VERDICT -> The ngen bundle is not "broadcast A in SLM."
  This is faster than per-k32 SLM and slower than no-SLM
  wgn at decode. Hand floor stays 47-50 us. W8A8 floor
  stays 45 us. Next: ngen SLM packing, not more A-share.

Evidence: `results/k2/SUMMARY.md`, `results/k2/slm64_dpas_lines.txt`.

## ngen M=1 SLM is d32, not A-pack; M=1 pad is not 45 us (K2)

CONFIG -> CPU IGA of ngen M=1 W8A8 bin (both cards,
  md5-identical). GPU: backend `sycl+l0`, `dpas_s8_dec`,
  same 8x2-along-N 64x `dpas.8x4` as wgn, M=1 zero-padded
  to RC=4. Both cards.

RESULT -> ngen SLM is `store/load/fence.slm.d32` (14 ops),
  not `load_block2d` A pack. GPU max_abs=0. Within-run M=1
  us tracks M=4. Warm card1 NT=2 D0/2800: M=1 49 us vs
  W8A8 42-46 vs wgn M=4 47-50. D3hot card0: M=1 97 us.

VERDICT -> The packing napkin was a misread of ngen SLM.
  Pad M=1 does not unlock a 4x cheaper kernel. Do not
  freeze 49 us (clocks). Hand floor stays 47-50 us. W8A8
  45 us remains the M=1 incumbent. Next steal is d32 ska
  remainder, not more A-pack.

Evidence: `results/k1/igc_card0_int8a8_jit/dnnl_dump_gpu_gemm_kernel.0.bin.xe2.asm`,
  `results/k2/SUMMARY.md`.

## Heat-then-decode does not hold 2800 MHz (K2)

CONFIG -> backend `sycl+l0`. Heat `dpas_s8` 1024^3 80
  iters, then `dpas_s8_dec` NT=2. Both cards. 0.2s gt0
  samples in `hold_n2_cardN.freq`.

RESULT -> max_abs=0. Heat 141 us card0 / 160 us card1.
  After heat cur=1167. M=1 5120: 61.5 / 65.4 us. M=4:
  132 / 141 us. Freq log sits ~717-750 MHz for most of
  the hold, not 2800. Prior D3hot M=1 97 us vs warm 49
  us still stands as a cold-card effect.

VERDICT -> Do not quote 46-48 us or a 3-5 us gap from
  this fire. The hold log is 61-65 us at drooped clocks.

Evidence: `results/k2/hold_n2_card0.txt`,
  `results/k2/hold_n2_card1.txt`.

## Batched same-kernel spin holds 2800; decode is 36 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_clk`
  (wgn 8x2-along-N 64x `dpas.8x4`, pad M to RC=4). No
  1024^3 heat. Both cards, NT=2. Batched spin=4000 of
  the same kernel (wait every 256), then 40 timed iters.
  Event us plus pipelined host us (matched to W8A8
  `us_bench`). min_freq not writable without root.

RESULT -> max_abs=0. timed_begin/end act=cur=2800.
  M=1 5120: event 35.80-35.96 us (min-max 34.4-37.2),
  pipelined host 36.43-36.44 us. M=4 tracks M=1
  (event 36.04-36.15, pipe 36.65-36.70). Per-iter sysfs
  or 1024^3 heat does not hold 2800. Prior 47-50 us was
  not held-2800. W8A8 K4 host 42.1/46.1 applies scales;
  this kernel stores s32.

VERDICT -> The way to hold 2800 on a short decode is a
  batched spin of that decode, not a long square GEMM.
  New hand floor is ~36 us at 2800 (raw s32). An
  in-kernel R-repeat fire gave 34.3-34.5 us_per at the
  same clock (launch removed). Do not call a serving
  beat of 42-46 until the scale epilogue is on the same
  clock and the same contract.

Evidence: `results/k2/clk_n2_p0_s4000_card0.txt`,
  `results/k2/clk_n2_p0_s4000_card1.txt`,
  `results/k2/rep_n2_card0.txt`,
  `results/k2/rep_n2_card1.txt`.

## Scale-to-f16 decode beats same-session W8A8 M=1 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc`.
  Same 8x2-along-N 64x `dpas.8x4` as wgn. acc * A_scale
  * B_scale, store f16. Scales 0.02, fill [-64,64] like
  the W8A8 bench. Spin=4000, both cards, NT=2. Control:
  `int8_gemm_w8a8` M=1 after M=64 heat, same image,
  pipelined host us.

RESULT -> max_abs=0 cosine=1.0. timed act=cur=2800.
  M=1 event 33.29/33.39 us, pipe_host 34.12/34.34 vs
  W8A8 44.55/43.82. M=4 tracks M=1. IGA 64x `dpas.8x4`
  + `store_block2d.ugm.d16`. K4 first-shape 42-46 was
  not this session's cold 79-85.

VERDICT -> This is a kernel beat in us of the live
  W8A8 GEMM-only at decode M=1, same scale/f16
  contract, held 2800, both cards. Rank us. Do not
  quote tok/s. Pad still does RC=4 work.

Evidence: `results/k2/sc_n2_s4000_card0.txt`,
  `results/k2/sc_n2_s4000_card1.txt`,
  `results/k2/w8a8_m1hold_card0.txt`,
  `results/k2/w8a8_m1hold_card1.txt`,
  `results/k2/sc_dpas_lines.txt`.

## Scale-to-f16 RC=4 loses W8A8 at M=64 (K2)

CONFIG -> backend `sycl+l0`, same `dpas_s8_sc` 8x2-N
  64x `dpas.8x4` as the M=1 beat. Spin=4000, NT=2,
  both cards. Control: same-day `int8_gemm_w8a8`
  M=64 46.17/46.45 us.

RESULT -> cosine=1.0 max_abs=0. timed cur=2800,
  act 2683-2750, throttle=1. M=64 5120: event
  246.88/244.83 us, pipe_host 247.16/242.53 vs
  W8A8 46.17/46.45. ~7.4x this tile's M=1, not 16x
  (occupancy: 10 vs 160 WGs).

VERDICT -> The M=1 us beat does not carry to
  prefill on this RC=4 tile. Rank us. Next steal
  is ngen M=64 RC=8/GRF256/SLM, not more spin.

Evidence: `results/k2/sc_m64_n2_s4000_card0.txt`,
  `results/k2/sc_m64_n2_s4000_card1.txt`,
  `results/k2/w8a8_hold_card0.txt`,
  `results/k2/w8a8_hold_card1.txt`.

## RC=8 dpas.8x8 halves M=64 vs RC=4, not 46 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8`.
  RC=8, 64x `dpas.8x8`, 8x2 along N, f16 scales 0.02.
  `grf_size<256>` requested. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8` `:b/:b`, `store_block2d.d16`,
  zebin `grf_count` 128. cosine=1.0 max_abs=0. timed
  cur=2800 act~2770 throttle=1. M=64 pipe_host
  119.6/121.0 vs RC=4 247/243 vs W8A8 46.17/46.45.

VERDICT -> RC=8 is a real 2x vs this tile's RC=4
  (half the M-blocks). It is not the ngen M=64 kernel.
  GRF256 request still does not change the zebin.
  Next: ngen wg 4x2x4 / SLM pack.

Evidence: `results/k2/sc8_m64_n2_s512_card0.txt`,
  `results/k2/sc8_m64_n2_s512_card1.txt`,
  `results/k2/sc8_dpas_lines.txt`.

## ngen 4x2x4 + SLM A pack loses to no-SLM RC=8 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc84`.
  WG 4x2x4, 4x RC=8 (32 rows/thread), SLM A per k64,
  NT=2, 64x `dpas.8x8`, f16 scales 0.02. Both cards,
  spin=512.

RESULT -> IGA 64x `dpas.8x8`, `slm_size` 4096,
  32 store.slm + 32 load.slm, `grf_count` 128.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 136.1/136.0 vs sc8
  120 vs W8A8 46.

VERDICT -> The ngen WG+SLM bundle as stolen here
  is a tax vs the 8x2-N no-SLM RC=8 tile. Do not
  keep adding barriers to chase 46 us. Rank us.

Evidence: `results/k2/sc84_m64_s512_card0.txt`,
  `results/k2/sc84_m64_s512_card1.txt`,
  `results/k2/sc84_dpas_lines.txt`.

## A double-buffer on sc8 M=64 beats no-db, not 46 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8db`.
  RC=8, 64x `dpas.8x8`, ska-style A ping-pong, no SLM,
  f16 scales 0.02. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8` `:b/:b`, `store_block2d.d16`,
  zebin `grf_count` 128, no `slm_size`. cosine=1.0
  max_abs=0. timed act=2783 cur=2800 throttle=1.
  M=64 pipe_host 96.6/100.4 vs sc8 120 vs sc84 136
  vs W8A8 46.17/46.45.

VERDICT -> Software-pipelined A is a real ~17-19%
  win vs the no-db RC=8 tile. SLM A-share is still
  the wrong steal. Not a 46 us beat.

Evidence: `results/k2/sc8db_m64_n2_s512_card0.txt`,
  `results/k2/sc8db_m64_n2_s512_card1.txt`,
  `results/k2/sc8db_dpas_lines.txt`.

## 4-acc M-tile without SLM matches sc8, loses to A-db (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8m4`.
  4x RC=8 (32 rows/thread), 64x `dpas.8x8`, A ping-pong,
  no SLM, f16 scales 0.02. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8` `{Atomic}`, `store_block2d.d16`,
  zebin `grf_count` 128, no `slm_size`. cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0. M=64
  pipe_host 119.8/119.7 vs sc8db 97-100 vs sc8 120
  vs W8A8 46.

VERDICT -> ngen's 4-acc M-tile without SLM is not
  the 46 us kernel and is slower than 8-row A-db.
  Occupancy of the 8-row tile beat extra B reuse.

Evidence: `results/k2/sc8m4_m64_n2_s512_card0.txt`,
  `results/k2/sc8m4_m64_n2_s512_card1.txt`,
  `results/k2/sc8m4_dpas_lines.txt`.

## B pipeline + ca.ca loses to A-db only at M=64 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8bp`.
  sc8db tile plus B ping-pong and `lsc_load_2d` L1/L2
  cached. Both cards, NT=2, spin=512.

RESULT -> IGA `load_block2d.ugm.d8.a64.ca.ca`, 64-68x
  `dpas.8x8`, `grf_count` 128, no SLM. cosine=1.0
  max_abs=0. timed act=2783 cur=2800 throttle=1.
  M=64 pipe_host 105.5/106.5 vs sc8db 96.6/100.4
  vs W8A8 46.

VERDICT -> ngen's B-in-GRF + ca.ca as stolen here
  is a tax vs A-db only. Keep sc8db as the hand
  floor. Not a 46 us beat.

Evidence: `results/k2/sc8bp_m64_n2_s512_card0.txt`,
  `results/k2/sc8bp_m64_n2_s512_card1.txt`,
  `results/k2/sc8bp_dpas_lines.txt`.

## Null-dest prefetch on sc8db is a tax at M=64 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8ff`.
  sc8db A-db plus `lsc_prefetch_2d` cached/cached of
  next k64 A and B. Both cards, NT=2, spin=512.

RESULT -> IGA 32x null `load_block2d.d8.ca.ca` (rd:0),
  64x `dpas.8x8`, `grf_count` 128, no SLM. cosine=1.0
  max_abs=0. timed act~2770 cur=2800 throttle=1.
  M=64 pipe_host 125.8/128.1 vs sc8db 96.6/100.4
  vs W8A8 46.

VERDICT -> ngen ff as stolen here is ~30% slower
  than A-db only. Stop M=64 load-path chasing.
  sc8db remains the hand floor.

Evidence: `results/k2/sc8ff_m64_n2_s512_card0.txt`,
  `results/k2/sc8ff_m64_n2_s512_card1.txt`,
  `results/k2/sc8ff_dpas_lines.txt`.

## k128 A-db at M=256 is ~4.5x M=64, not 75 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8k128`.
  RC=8, k128 (4x k32), A ping-pong, 64x `dpas.8x8`,
  f16 scales 0.02. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8`, `store_block2d.d16`,
  `grf_count` 128, no SLM. cosine=1.0 max_abs=0.
  timed act=2550-2600 cur=2800 throttle=1. M=256
  pipe_host 442.6/439.4 vs napkin 4x98~392 vs
  K4 W8A8 74.9/76.1 (not same-session).

VERDICT -> k128 blocking does not beat oneDNN at
  M=256. Occupancy did not cancel the 4x M. Rank
  us. Clocks lower than M=64 (throttle=1).

Evidence: `results/k2/sc8k128_m256_n2_s512_card0.txt`,
  `results/k2/sc8k128_m256_n2_s512_card1.txt`,
  `results/k2/sc8k128_dpas_lines.txt`.

## wg 4x8 halves M=256 vs 8x2-along-N, not 75 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8w48`.
  Same k128 A-db 8-row tile, launch wg 4 along N x
  8 along M. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8`, `grf_count` 128, no SLM.
  cosine=1.0 max_abs=0. timed act=2733-2750 cur=2800
  throttle=1. M=256 pipe_host 229.5/227.7 vs k128
  8x2-N 440 vs W8A8 75.

VERDICT -> Mapping M onto the WG Y dim is a real
  ~1.9x vs all-N 8x2. Still ~3x oneDNN. Geometry
  beat k128 blocking.

Evidence: `results/k2/sc8w48_m256_n2_s512_card0.txt`,
  `results/k2/sc8w48_m256_n2_s512_card1.txt`,
  `results/k2/sc8w48_dpas_lines.txt`.

## wg 4x8 A-db is the M=64 hand floor at 75 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8db48`.
  sc8db A-db tile, wg 4 along N x 8 along M. Both
  cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8`, `grf_count` 128, no SLM.
  cosine=1.0 max_abs=0. timed act=cur=2800 throttle=0.
  M=64 pipe_host 75.5/75.6 vs sc8db 96.6/100.4 vs
  W8A8 46.17/46.45.

VERDICT -> Mapping M onto WG Y is a real ~1.3x vs
  8x2-along-N at M=64. New hand floor 75 us, ~1.63x
  oneDNN. Geometry beat load-path extras.

Evidence: `results/k2/sc8db48_m64_n2_s512_card0.txt`,
  `results/k2/sc8db48_m64_n2_s512_card1.txt`,
  `results/k2/sc8db48_dpas_lines.txt`.

## 4-acc + wg 4x8 is the M=256 hand floor at 128 us (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8w48m4`.
  4x RC=8 (32 rows/thread), wg 4x8, k128, 256x
  `dpas.8x8`, no A-db. Both cards, NT=2, spin=512.

RESULT -> IGA 256x `dpas.8x8` `{Atomic}`, `grf_count`
  128, no SLM. cosine=1.0 max_abs=0. timed act=2767
  cur=2800 throttle=1. M=256 pipe_host 128.4/128.6
  vs 4x8 8-row 228 vs W8A8 75.

VERDICT -> 4-acc on the 4x8 geometry is a real ~1.8x
  vs 8-row at M=256. New floor 128 us, ~1.7x oneDNN.
  256 dpas landed; 384-count 6-acc lost (be).

Evidence: `results/k2/sc8w48m4_m256_n2_s512_card0.txt`,
  `results/k2/sc8w48m4_m256_n2_s512_card1.txt`,
  `results/k2/sc8w48m4_dpas_lines.txt`.

## 384-count 6-acc loses to 4-acc at M=256 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8w48m6`.
  6x RC=8 (48 rows/thread), wg 4x8, k128, 384x
  `dpas.8x8`, pad M 256->288, no A-db. Both cards,
  NT=2, spin=512. Prior: ngen 384 dpas is 16 acc
  (4M x 4N); 5120 is not a multiple of 768, so
  this is the 384 count that divides K.

RESULT -> IGA 384x `dpas.8x8` (192 `{Atomic}`),
  `grf_count` 128, no SLM, IGC spill 768 B.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host 210.1/210.0 vs
  4-acc 128 vs W8A8 75.

VERDICT -> Hitting ngen's 384 *count* with 6 M-tiles
  is a ~1.64x loss vs 4-acc 256 dpas. Floor stays
  128 us. Do not chase dpas count without the 16-acc
  4M x 4N tile and GRF256.

Evidence: `results/k2/sc8w48m6_m256_n2_s512_card0.txt`,
  `results/k2/sc8w48m6_m256_n2_s512_card1.txt`,
  `results/k2/sc8w48m6_dpas_lines.txt`.

## k32 A-db on 4-acc is a tax at M=256 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8w48m4db`.
  Same 4-acc wg 4x8 k128 256x `dpas.8x8` plus ska-style
  k32 A ping-pong. Both cards, NT=2, spin=512.
  Prior: A-db won at M=64.

RESULT -> IGA 256x `dpas.8x8` (192 `{Atomic}`),
  `grf_count` 128, no SLM, NT=2 no spill.
  cosine=1.0 max_abs=0. timed act=2783 cur=2800
  throttle=1. M=256 pipe_host 135.1/134.9 vs
  4-acc no A-db 128 vs W8A8 75.

VERDICT -> A-db is shape-dependent. On 4-acc M=256
  it is a ~1.05x tax, not a beat. Floor stays 128 us.

Evidence: `results/k2/sc8w48m4db_m256_n2_s512_card0.txt`,
  `results/k2/sc8w48m4db_m256_n2_s512_card1.txt`,
  `results/k2/sc8w48m4db_dpas_lines.txt`.

## 4-acc wg 4x2 M-on-Y is a small win vs 8x2-N at M=64 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8m42`.
  Same 4-acc k64 A-db 64x `dpas.8x8` as sc8m4, wg 4
  along N x 2 along M. Both cards, NT=2, spin=512.

RESULT -> IGA 64x `dpas.8x8` (33 `{Atomic}`),
  `grf_count` 128, no SLM, IGC spill 1792 B.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 115.3/116.0 vs
  sc8m4 120 vs 4x8 A-db 75 vs W8A8 46.

VERDICT -> Mapping M onto WG Y is a real ~1.04x vs
  all-N 8x2 at this 4-acc tile. Not the 75 us
  floor. 8-thread WG occupancy is the leftover.

Evidence: `results/k2/sc8m42_m64_n2_s512_card0.txt`,
  `results/k2/sc8m42_m64_n2_s512_card1.txt`,
  `results/k2/sc8m42_dpas_lines.txt`.

## 4-acc wg 4x2x4 no SLM loses to 8-thread 4x2 at M=64 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc8m424`.
  Same 4-acc k64 A-db 64x `dpas.8x8` as sc8m42, ngen
  wg 4x2x4 (32 threads), no SLM. Both cards, NT=2,
  spin=512. Prior: 8-thread WG occupancy leftover.

RESULT -> IGA 64x `dpas.8x8` (33 `{Atomic}`),
  `grf_count` 128, no SLM, IGC spill 1792 B.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 132.6/132.9 vs
  4x2 115 vs sc84 SLM 136 vs 4x8 A-db 75 vs
  W8A8 46.

VERDICT -> 32-thread ngen 4x2x4 without SLM is a
  ~1.15x loss vs 8-thread 4x2. Occupancy was not
  the leftover. Stop M=64 4-acc. INT8 hand floor
  stays 75 us.

Evidence: `results/k2/sc8m424_m64_n2_s512_card0.txt`,
  `results/k2/sc8m424_m64_n2_s512_card1.txt`,
  `results/k2/sc8m424_dpas_lines.txt`.

## s4 on the 4x8 A-db tile is 33.6 us at M=64 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_db48`.
  Packed s4 A/B (2/byte along K), Transformed B,
  RC=8, 32x `dpas.8x8` `:s4/:s4`, wg 4x8, k64 A-db,
  f16 scales 0.02, fill [-8,7]. Both cards, NT=2,
  spin=512. Prior: s8 same tile 75 us; s4 1.49x
  s8 at 1024^3 / ~583 MHz.

RESULT -> IGA 32x `dpas.8x8` rW:s4 rA:s4,
  `store_block2d.d16`, `grf_count` 128, no SLM.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 33.61/33.74 vs
  s8 4x8 A-db 75.49/75.61 vs W8A8 46.17/46.45.

VERDICT -> Native s4 on the winning s8 schedule
  is a real ~2.24x vs that s8 tile and under the
  W8A8 46 us wall time at M=64 5120. New s4 hand
  floor 33.6 us at 2800. This is not an INT8
  kernel; s8 floor stays 75 us. Do not freeze
  the 1.49x 1024^3 ratio as the serving-shape
  story. Rank us. Do not quote tok/s.

Evidence: `results/k2/s4db48_m64_n2_s512_card0.txt`,
  `results/k2/s4db48_m64_n2_s512_card1.txt`,
  `results/k2/s4db48_dpas_lines.txt`.

## s4 4-acc M=256 is 48.6 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_w48m4`.
  Packed s4, 4x RC=8, wg 4x8, k128, 128x `dpas.8x8`
  `:s4/:s4`, no SLM. Both cards, NT=2, spin=512.

RESULT -> IGA 128x `dpas.8x8` rW:s4 rA:s4,
  `grf_count` 128, no SLM, NT=2 no spill.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host 48.65/48.47 vs
  s8 128 vs W8A8 75. Spread ~0.4%.

VERDICT -> New s4 M=256 hand floor 48.6 us at
  2800 both cards. ~2.63x the s8 4-acc tile and
  under W8A8 75 us. Different dtype than W8A8.
  INT8 s8 floor stays 128 us. Rank us.

Evidence: `results/k2/s4w48m4_m256_n2_s512_card0.txt`,
  `results/k2/s4w48m4_m256_n2_s512_card1.txt`,
  `results/k2/s4w48m4_dpas_lines.txt`.

## s4 RC=4 decode is 16.5 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_sc`.
  Packed s4, RC=4, wg 8x2 along N, 32x `dpas.8x4`
  `:s4/:s4`, pad M to 4, f16 scales 0.02. Both
  cards, NT=2, spin=4000.

RESULT -> IGA 32x `dpas.8x4` rW:s4 rA:s4,
  `store_block2d.d16`, `grf_count` 128, no SLM.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 16.41/16.58 vs s8
  34 vs W8A8 44. M=4 tracks. Spread ~1%.

VERDICT -> New s4 decode floor 16.5 us at 2800
  both cards. ~2.05x s8 34 us. Pad still does
  RC=4 work. Different dtype than W8A8. Rank us.

Evidence: `results/k2/s4sc_n2_s4000_card0.txt`,
  `results/k2/s4sc_n2_s4000_card1.txt`,
  `results/k2/s4sc_dpas_lines.txt`.

## s4 A-db on 4-acc is a tax at M=256 (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s4_w48m4db`. Same 4-acc wg 4x8 k128
  plus k64 A ping-pong. Card0 only, NT=2,
  spin=512. Prior: s8 A-db was a tax here.

RESULT -> IGA 128x `dpas.8x8` rW:s4 rA:s4,
  `grf_count` 128, no SLM, NT=2 no spill.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host 51.94 vs no
  A-db 48.65 vs W8A8 75.

VERDICT -> A-db is still a tax on this 4-acc
  M=256 tile in s4 (~1.07x). Floor stays
  48.6 us. Do not keep chasing A-db here.

Evidence: `results/k2/s4w48m4db_m256_n2_s512_card0.txt`,
  `results/k2/s4w48m4db_dpas_lines.txt`.

## s4 decode N=17408 is 29.5 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_sc`.
  Same RC=4 8x2-N tile, M=1 N=17408 K=5120.
  Both cards, NT=2, spin=4000.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 29.24/29.75 vs
  N=5120 16.5 vs napkin 56. M=4 tracks.
  Spread ~1.8%.

VERDICT -> New s4 wide-N floor 29.5 us at 2800
  both cards. ~1.80x N=5120, not 3.4x. Rank us.

Evidence: `results/k2/s4sc_n17408_n2_s4000_card0.txt`,
  `results/k2/s4sc_n17408_n2_s4000_card1.txt`.

## s4 decode K=17408 is 53.4 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_sc`.
  Same RC=4 8x2-N tile, M=1 N=5120 K=17408.
  Both cards, NT=2, spin=4000. Prior: K-linear
  ~56 us; same B bytes as N=17408.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 53.47/53.37 vs
  K=5120 16.5 vs N=17408 29.5 vs napkin 56.
  M=4 tracks. Spread ~0.2%.

VERDICT -> New s4 down-proj floor 53.4 us at
  2800 both cards. ~3.24x K=5120, near linear.
  Slower than wide-N at the same B bytes.
  Rank us.

Evidence: `results/k2/s4sc_k17408_n2_s4000_card0.txt`,
  `results/k2/s4sc_k17408_n2_s4000_card1.txt`.

## s4 M=64 N=17408 is 94.7 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_db48`.
  Same 4x8 A-db tile, M=64 N=17408 K=5120. Both
  cards, NT=2, spin=512. Prior: N-linear ~114 us.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 94.81/94.56 vs
  N=5120 33.6 vs napkin 114. Spread ~0.3%.

VERDICT -> New s4 M=64 wide-N floor 94.7 us at
  2800 both cards. ~2.81x N=5120, closer to
  linear than M=1's 1.80x. Rank us.

Evidence: `results/k2/s4db48_m64_n17408_n2_s512_card0.txt`,
  `results/k2/s4db48_m64_n17408_n2_s512_card1.txt`.

## s4 M=64 K=17408 is 106.0 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_db48`.
  Same 4x8 A-db tile, M=64 N=5120 K=17408. Both
  cards, NT=2, spin=512. Prior: K-linear ~114 us.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 106.10/105.84 vs
  K=5120 33.6 vs N=17408 94.7 vs napkin 114.
  Spread ~0.2%.

VERDICT -> New s4 M=64 wide-K floor 106.0 us at
  2800 both cards. ~3.15x K=5120, near linear.
  Slower than wide-N at the same B bytes. Rank us.

Evidence: `results/k2/s4db48_m64_k17408_n2_s512_card0.txt`,
  `results/k2/s4db48_m64_k17408_n2_s512_card1.txt`.

## s4 M=256 N=17408 is 140.0 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_w48m4`.
  Same 4-acc wg 4x8 tile, M=256 N=17408 K=5120.
  Both cards, NT=2, spin=512. Prior: N-linear
  ~165 us.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host 139.44/140.53 vs
  N=5120 48.6 vs napkin 165. Spread ~0.8%.

VERDICT -> New s4 M=256 wide-N floor 140.0 us at
  2800 both cards. ~2.88x N=5120, like M=64's
  2.81x, not 3.40x. Rank us.

Evidence: `results/k2/s4w48m4_m256_n17408_n2_s512_card0.txt`,
  `results/k2/s4w48m4_m256_n17408_n2_s512_card1.txt`.

## s4 M=256 K=17408 is 149.0 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s4_w48m4`.
  Same 4-acc wg 4x8 tile, M=256 N=5120 K=17408.
  Both cards, NT=2, spin=512. Prior: K-linear
  ~165 us.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=256 pipe_host 148.77/149.16 vs
  K=5120 48.6 vs N=17408 140.0 vs napkin 165.
  Spread ~0.3%.

VERDICT -> New s4 M=256 wide-K floor 149.0 us at
  2800 both cards. ~3.07x K=5120, near linear.
  Qwen FFN s4 map is closed. Rank us.

Evidence: `results/k2/s4w48m4_m256_k17408_n2_s512_card0.txt`,
  `results/k2/s4w48m4_m256_k17408_n2_s512_card1.txt`.

## s8 M=64 N=17408 is 338.9 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8_sc8db48`. Same 4x8 A-db INT8 tile,
  M=64 N=17408 K=5120. Both cards, NT=2,
  spin=512. Prior: N-linear ~255 us; s4 94.7.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 339.63/338.15 vs
  N=5120 75 vs s4 94.7 vs napkin 255.
  Spread ~0.4%.

VERDICT -> New s8 M=64 wide-N floor 338.9 us at
  2800 both cards. ~4.52x N=5120, worse than
  linear. s4 94.7 is ~3.58x this tile. Rank us.
  No oneDNN N=17408 control this fire.

Evidence: `results/k2/sc8db48_m64_n17408_n2_s512_card0.txt`,
  `results/k2/sc8db48_m64_n17408_n2_s512_card1.txt`.

## s8 M=64 K=17408 is 374.7 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8_sc8db48`. Same 4x8 A-db INT8 tile,
  M=64 N=5120 K=17408. Both cards, NT=2,
  spin=512. Prior: K-linear ~255 us; s4 106.0.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=64 pipe_host 375.90/373.57 vs
  K=5120 75 vs N=17408 338.9 vs s4 106.0 vs
  napkin 255. Spread ~0.6%.

VERDICT -> New s8 M=64 wide-K floor 374.7 us at
  2800 both cards. ~5.00x K=5120, worse than
  linear and slower than wide-N 338.9 at the
  same B bytes. s4 106.0 is ~3.53x this tile.
  Rank us.

Evidence: `results/k2/sc8db48_m64_k17408_n2_s512_card0.txt`,
  `results/k2/sc8db48_m64_k17408_n2_s512_card1.txt`.

## s8 M=256 N=17408 is 469.8 us at cur=2800 (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8_sc8w48m4`. Same 4-acc wg 4x8 INT8
  tile, M=256 N=17408 K=5120. Both cards, NT=2,
  spin=512. Prior: N-linear ~435 us; s4 140.0.

RESULT -> cosine=1.0 max_abs=0. timed act=2700/2733
  cur=2800 throttle=1 (same flag as the 128 us
  N=5120 floor on this tile). M=256 pipe_host
  471.66/467.88 vs N=5120 128 vs s4 140.0 vs
  napkin 435. Spread ~0.8%.

VERDICT -> New s8 M=256 wide-N floor 469.8 us at
  cur=2800 both cards, throttle=1. ~3.67x N=5120,
  near linear, better than M=64's 4.52x. s4
  140.0 is ~3.36x this tile. Rank us.

Evidence: `results/k2/sc8w48m4_m256_n17408_n2_s512_card0.txt`,
  `results/k2/sc8w48m4_m256_n17408_n2_s512_card1.txt`.

## s8 M=256 K=17408 is 477.4 us at cur=2800 (K2)

CONFIG -> backend `sycl+l0`, standalone
  `dpas_s8_sc8w48m4`. Same 4-acc wg 4x8 INT8
  tile, M=256 N=5120 K=17408. Both cards, NT=2,
  spin=512. Prior: K-linear ~435 us; s4 149.0.

RESULT -> cosine=1.0 max_abs=0. timed act=2733/2750
  cur=2800 throttle=1 (same flag as the 128 us
  N=5120 floor on this tile). M=256 pipe_host
  477.80/476.93 vs K=5120 128 vs N=17408 469.8
  vs s4 149.0 vs napkin 435. Spread ~0.2%.

VERDICT -> New s8 M=256 wide-K floor 477.4 us at
  cur=2800 both cards, throttle=1. ~3.73x K=5120,
  near linear. s4 149.0 is ~3.20x this tile.
  Qwen FFN s8 prefill map is closed. Rank us.

Evidence: `results/k2/sc8w48m4_m256_k17408_n2_s512_card0.txt`,
  `results/k2/sc8w48m4_m256_k17408_n2_s512_card1.txt`.

## s8 decode N=17408 is 141.6 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc`.
  Same RC=4 8x2-N INT8 tile, M=1 N=17408 K=5120.
  Both cards, NT=2, spin=4000. Prior: N-linear
  ~116 us; s4 29.5.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 141.22/142.05 vs
  N=5120 34 vs s4 29.5 vs napkin 116. M=4
  tracks. Spread ~0.6%.

VERDICT -> New s8 decode wide-N floor 141.6 us
  at 2800 both cards. ~4.16x N=5120, worse than
  linear. s4 29.5 is ~4.80x this tile (s4 was
  1.80x N=5120). Rank us.

Evidence: `results/k2/sc_n17408_n2_s4000_card0.txt`,
  `results/k2/sc_n17408_n2_s4000_card1.txt`.

## s8 decode K=17408 is 261.6 us at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_sc`.
  Same RC=4 8x2-N INT8 tile, M=1 N=5120 K=17408.
  Both cards, NT=2, spin=4000. Prior: K-linear
  ~116 us; s4 53.4.

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0. M=1 pipe_host 261.68/261.51 vs
  K=5120 34 vs N=17408 141.6 vs s4 53.4 vs
  napkin 116. M=4 tracks. Spread ~0.06%.

VERDICT -> New s8 decode wide-K floor 261.6 us
  at 2800 both cards. ~7.69x K=5120, much worse
  than linear. s4 53.4 is ~4.90x this tile (s4
  was 3.24x). Qwen FFN s8 map is closed. Rank us.

Evidence: `results/k2/sc_k17408_n2_s4000_card0.txt`,
  `results/k2/sc_k17408_n2_s4000_card1.txt`.

## oneDNN W8A8 M=1 N=17408 is 158.1 us at 2800 (K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`, sglang
  int8 mtp6 `int8_gemm_w8a8` GEMM-only. Both cards.
  spin=2000 of M=1 then timed. Host oracle after
  timed. n=17408 k=5120. Prior: K4 sweep 161 us.

RESULT -> cosine=1.000 max_abs=0.055. timed
  act=cur=2800 throttle=0. M=1 158.13/158.01 us
  vs K4 sweep 161 vs hand s8 141.6 vs s4 29.5.
  Spread ~0.08%.

VERDICT -> New oneDNN W8A8 wide-N decode floor
  158.1 us at 2800 both cards. Hand s8 141.6 is
  ~1.12x this incumbent. Rank us.

Evidence: `results/k2/w8a8_m1hold_n17408_k5120_card0.txt`,
  `results/k2/w8a8_m1hold_n17408_card1.txt`.

## oneDNN W8A8 M=1 K=17408 is 155.3 us at 2800 (K4)

CONFIG -> backend `pytorch-xpu` on `sycl+l0`, same
  `int8_gemm_w8a8` GEMM-only. Both cards. spin=2000
  of M=1. n=5120 k=17408. Prior: N-linear ~150 us;
  hand s8 261.6.

RESULT -> cosine=1.000 max_abs=0.070-0.104. timed
  act=cur=2800 throttle=0. M=1 155.31/155.37 us
  vs N=17408 158.1 vs hand s8 261.6 vs s4 53.4
  vs W8A8 5120 44. Spread ~0.04%.

VERDICT -> New oneDNN W8A8 wide-K decode floor
  155.3 us at 2800 both cards. ~3.53x the 5120
  floor, N/K-symmetric with 158.1. Hand RC=4
  8x2-N 261.6 loses to this incumbent (~1.68x).
  Qwen FFN oneDNN W8A8 decode map is closed.
  Rank us.

Evidence: `results/k2/w8a8_m1hold_n5120_k17408_card0.txt`,
  `results/k2/w8a8_m1hold_n5120_k17408_card1.txt`.

## Scalar RMSNorm-quant inside the GEMM is not a 34 us fuse (K5)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_fuse`.
  f16 A RMSNorm+s8 then 64x `dpas.8x4` f16 out. Scalar
  sqrt/rint in the k-loop. Both cards, NT=2, spin=4000.

RESULT -> timed act=cur=2800. M=1 314/313 us vs GEMM
  34 vs K5 extra launch 7-36. cosine 0.73 max_abs 50.

VERDICT -> This is the K6 scalar-LUT tax again. Do not
  fuse by putting a scalar quant in the DPAS loop.
  Two-launch WG-256 producer + 34 us GEMM stays the
  robust decode path.

Evidence: `results/k5/fuse_n2_s4000_card0.txt`,
  `results/k5/fuse_n2_s4000_card1.txt`.

## Vectorized RMSNorm-quant fuse is 72 us, not 34 (K5)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_fusev`.
  simd convert/reduce/hmax/rnde, f16 2D load pitch in
  bytes, 64x `dpas.8x4` f16 out. Both cards, NT=2,
  spin=4000. Prior: scalar fuse 313 us cosine 0.73.

RESULT -> IGA 64x `dpas.8x4`, `rnde (32|M0)`,
  `math.rsqt` x4, A `load_block2d.d16`, B `d8v`,
  `store_block2d.d16`, GRF 128, no SLM. cosine=1.0
  max_abs=0.015625. timed act=cur=2800 throttle=0.
  M=1 pipe_host 72.47/72.28 vs scalar 313 vs GEMM
  34 vs two-launch 41-70. M=4 tracks M=1.

VERDICT -> Vectorizing closes the fuse (~4.3x the
  scalar arm). It is not a launch win vs WG-256
  producer + 34 us GEMM. Scalar 0.73 was f16 pitch
  in elements. Do not re-read A on every GEMM thread
  to chase 34 us.

Evidence: `results/k5/fusev_n2_s4000_card0.txt`,
  `results/k5/fusev_n2_s4000_card1.txt`,
  `results/k5/fusev_dpas_lines.txt`.

## Producer+GEMM two-kernel is 44 us, beats fusev 72 (K5)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_prod`.
  WG-256 RMSNorm-quant writes s8 A + scale, then
  64x `dpas.8x4` f16 GEMM. In-order queue, no host
  wait between. Both cards, NT=2, spin=4000.

RESULT -> cosine=1.0 max_abs=0.015625. timed
  act=cur=2800 throttle=0. M=1 prod 10.5/10.4,
  gemm 33.1/33.2, pipe_host 44.3/44.4 vs fusev 72
  vs GEMM-only 34. M=4 tracks. This pair is the
  two-launch (prior 41-70 range was producer us
  plus GEMM, not one queue).

VERDICT -> Do not re-read A inside the GEMM. A
  standalone producer plus the 34 us GEMM is the
  decode quant path. Extra ~10 us over GEMM-only.

Evidence: `results/k5/prod_n2_s4000_card0.txt`,
  `results/k5/prod_n2_s4000_card1.txt`.

## ngen d32 flag broadcast is not the 36 us kernel (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_d32`.
  Prologue SLM d32 token + group barrier on the wgn
  64-dpas tile. Both cards. CPU ocloc of the AOT zebin.

RESULT -> IGA `store.slm.d32` + `fence.slm.none.group`
  + `send.gtwy` barrier + `load.slm.d32` + 64x
  `dpas.8x4`. GRF 128. max_abs=0. M=1: 90-194 us vs
  held-clock no-SLM 36 us vs no-hold wgn 47-50.

VERDICT -> The ngen d32 encoding landed. A dummy flag
  plus barrier is a decode tax, not the 3-5 us steal.
  Hand floor stays no-SLM 36 us at 2800.

Evidence: `results/k2/d32_dpas_lines.txt`,
  `results/k2/SUMMARY.md`.

## In-kernel GEMM repeats: 34 us body at 2800 (K2)

CONFIG -> backend `sycl+l0`, standalone `dpas_s8_rep`.
  Same 8x2-along-N 64x `dpas.8x4`. One launch repeats
  GEMM 4096x (M=1) or 2048x (M=4). Both cards. Freq
  every 0.1s during the kernel.

RESULT -> max_abs=0. 8 samples act=2800/cur=2800 both
  cards. us_per 34.36-34.46 (M=1) and 34.47-34.51 (M=4).
  Matches clk event min 34.4 us. Held one-shot is 36 us.

VERDICT -> Body and batched one-shot agree at 34-36 us
  at 2800. Do not quote 34 as a one-token serving beat.
  Scale epilogue is the remaining W8A8-contract gap.

Evidence: `results/k2/rep_n2_card0.txt`,
  `results/k2/rep_n2_card1.txt`.

## Vectorized in-register nibble LUT is ~6-8x the scalar arm (K6)

CONFIG -> same VNNI4 in-register spoof, simd nibble decode +
  simd VNNI4 `select`. Backend `sycl+l0`. Both cards, 1024^3.

RESULT -> max_abs=0. 304 us card0 (start 633 MHz), 406 us
  card1 (start 2800). Scalar unroll was 2316 us both cards.

VERDICT -> Vectorizing the LUT is a real us win vs the scalar
  in-register arm. It is not yet a matched-clock beat of
  two-launch unpack. Keep both arms.

Evidence: `results/k6/SUMMARY.md`.

Open campaign (questions, not findings): docs/KERNEL_CAMPAIGN.md.
Sibling-lab claims to reproduce before they can enter this file.
Now local (K2): s4 DPAS exists. 1.49x s8 at 1024^3 / ~583 MHz;
~2.05x at M=1 (16.5 vs 34), ~2.24x at M=64 (33.6 vs 75),
~2.63x at M=256 (48.6 vs 128) held 2800. INT2 DPAS exists
(s2xs2 and s2xs8 closed). Remaining:

- ESIMD `dpas<s4,s4>` ~2x s8 MAC rate -- 1.49x at 1024^3;
  serving shapes at 2800 are ~2.05-2.63x. Tile-dependent.
- Xe2 has no native FP8 XMX -- reproduced: fp8_gemm_w8a16 is
  E4M3 unpack + bf16 dpas.8x8 (K1 JIT dump).
- NVFP4 E2M1 x2 is exact int8; +-12 overflows s4; bitcast is wrong.
  Local: nibble LUT -> s8 DPAS closed (K6); two-term s4 compose
  closed (K3). In-register VNNI4 pack closed; scalar LUT lost
  us; simd LUT is ~6-8x that arm (304-406 us) and still clock-
  bound vs two-launch unpack. Serving-shaped simd LUT on the
  RC=4 8x2-N tile is 158 us merge / 134.8 us closed-form
  at 2800 both cards (cq / 03ad), numeric closed, ~4.0x
  s8 34. Packed E2M1 stays in HBM. 16-entry
  iselect table is 1022 us (cr), a loss; stop gather tables.
  Scalar two-launch unpack is 265 us (2026-09-03a), a loss
  vs 158; s8ctrl 34.5. k64 combined load is 169 us
  (2026-09-03b), a small loss. Vectorized unpack is
  314.6 us both cards (2026-09-03f), a loss. Keep
  two k32 closed-form as the s8-A spoof. E2M1
  two-term s4 decode is 28.5 us both cards
  (2026-09-03e) vs s4 16.5 vs s8 34, A=s4.
  Wide-N is 103.5 us both cards (~3.63x, not
  s4's 1.80x). Wide-K is 193.6 us both cards
  (~6.79x vs s4 53.4). 8x2-N compose at M=64
  is 217.9 us card0 (loss vs s4 33.6) and at
  M=256 is ~607 us both cards (loss vs s4
  48.6, throttle=1). 8x2-N LUT at M=64 is
  ~656 us both cards (loss vs s8 75). Stop
  this tile at prefill. compose on s4 4x8
  A-db M=64 is 68.7 us both cards (~2.04x
  s4 33.6, ~3.17x faster than 8x2-N).
  compose 4x8 A-db M=256 is 194.9 us both
  cards (~3.12x 8x2-N 607). nibble LUT on
  s8 4x8 A-db M=64 is 392.4 us both cards
  (~1.67x 8x2-N 656, still ~5.23x s8 75).
  compose 4x8 A-db M=64 N=17408 is 326.9 us
  both cards (~4.76x square vs s4 94.7).
  compose M=64 K=17408 is 403.4 us both
  cards (~5.87x square vs s4 106, loses to
  s8 374.7). compose 4x8 A-db M=256 N=17408
  is 984.3 us both cards (~5.05x square vs
  s4 140, ~2.10x s8 469.8). compose M=256
  K=17408 is 968.7 us both cards (~4.97x
  square vs s4 149, ~2.03x s8 477.4).
  throttle=0. Qwen FFN compose M=256 map
  is closed. compose on s4 4-acc M=256 is
  411 us card0 (loss vs 4x8 194.9, ~8.46x
  s4 48.6). nibble LUT 4x8 A-db M=64
  N=17408 is 1032 us both cards (~2.63x
  square vs s8 338.9). LUT M=64 K=17408
  is 1333 us both cards (K-linear ~3.40x
  vs s8 374.7). Qwen FFN LUT M=64 map is
  closed. LUT 4x8 A-db M=256 is 1203 us
  both cards (~3.07x M=64 vs s8 128).
  closed-form LUT on 4x8 A-db M=64 is
  331.6 us both cards (~1.18x merge 392.4).
  closed-form LUT 4x8 A-db M=256 is 1083
  us both cards (~3.27x M=64, ~1.11x
  merge 1203). closed-form LUT 4x8 A-db
  M=64 N=17408 is 880 us both cards
  (~2.65x square vs s8 338.9,
  throttle=1). closed-form LUT 4x8 A-db
  M=64 K=17408 is 1125 us both cards
  (K-linear ~3.39x vs s8 374.7). Qwen
  FFN closed-form LUT M=64 map is closed.
  closed-form LUT 4x8 A-db M=256 N=17408
  is 3138 us both cards (~2.90x square vs
  s8 469.8, throttle=1). closed-form LUT
  4x8 A-db M=256 K=17408 is 3428 us both
  cards (~3.17x square vs s8 477.4,
  throttle=1). Qwen FFN closed-form LUT
  M=256 map is closed. 4x8 LUT loses to
  s8/s4/compose at FFN prefill.
  Held-clock nvfp4_gemm_w4a16 M=1 is
  34.7 us both cards at 2800 (bf16-A,
  same us class as s8 34, under W8A8
  44). nvfp4_gemm_w4a16 M=64 is 37.1 us
  both cards (act 2150-2400/2800,
  ~1.07x M=1, under W8A8 46).
  nvfp4_gemm_w4a16 M=256 is 118 us
  both cards (act 2350-2550/2800,
  ~3.18x M=64, loses to W8A8 75).
  nvfp4_gemm_w4a16 M=1 N=17408 is 97 us
  both cards (throttle=1, ~2.80x square,
  beats s8 141.6). nvfp4_gemm_w4a16 M=1
  K=17408 is 101 us both cards (throttle=1,
  ~2.92x square, beats s8 261.6). Qwen
  FFN w4a16 decode map is closed.
  nvfp4_gemm_w4a16 M=64 N=17408 is 142 us
  both cards (act 2050-2300/2800, ~3.82x
  square, beats s8 338.9). nvfp4_gemm_w4a16
  M=64 K=17408 is 130 us both cards (act
  2100-2400/2800, ~3.51x square, ~K-linear,
  beats s8 374.7). Qwen FFN w4a16 M=64
  map is closed. nvfp4_gemm_w4a16 M=256
  N=17408 is 394 us both cards (act
  2250-2300/2800, throttle=1, ~3.34x
  square, ~N-linear, beats s8 469.8).
  nvfp4_gemm_w4a16 M=256 K=17408 is 377 us
  both cards (act 2250-2317/2800, throttle=1,
  ~3.19x square, under K-linear, beats s8
  477.4). Qwen FFN w4a16 M=256 map is
  closed. oneDNN W8A8 M=256 N=17408 is
  248 us both cards (act 2467-2517/2800,
  throttle=1, ~3.31x square, beats w4a16
  394). oneDNN W8A8 M=256 K=17408 is 226
  us both cards (act 2483-2567/2800,
  throttle=1, ~3.01x square, beats w4a16
  377). Qwen FFN W8A8 M=256 map is closed.
  oneDNN W8A8 M=64 N=17408 is 202 us
  both cards (act=2783/2800, throttle=1,
  ~4.39x square, loses to w4a16 142).
  oneDNN W8A8 M=64 K=17408 is 181 us
  both cards (act~2725/2800, throttle=1,
  ~3.93x square, loses to w4a16 130).
  Qwen FFN W8A8 M=64 map is closed.
  ESIMD s2 RC=4 decode is 11.5 us both
  cards at 2800 (cosine=1 max_abs=0,
  ~1.43x s4 16.5). s2 4x8 A-db M=64
  is 20 us both cards at 2800, beats
  s4 33.6 (~1.68x) and W8A8 46
  (~2.21x). New M=64 hand floor. s2
  4x8 M=64 N=17408 is 53.1 us both
  cards at 2800, beats W8A8 202
  (~3.81x). s2 4x8 M=64 K=17408 is
  64 us both cards at 2800, beats
  W8A8 181 (~2.83x). Qwen FFN s2
  M=64 map is closed (20 / 53.1 /
  64). s2 4x8 A-db M=256 is 55.5
  us both cards at 2800, beats
  W8A8 75 (~1.35x), loses to s4
  4-acc 48.6 (~1.14x). s2 4-acc
  M=256 is 37.4 us both cards at
  2800, new M=256 hand floor,
  beats s4 48.6 (~1.30x) and W8A8
  75 (~2.01x). s2 4-acc M=256
  N=17408 is 110 us both cards at
  2800, beats s4 140 and W8A8 248
  (~2.25x). s2 4-acc M=256 K=17408
  is 108 us both cards at 2800,
  beats s4 149 and W8A8 226
  (~2.09x). Qwen FFN s2 4-acc
  M=256 map is closed (37.4 / 110
  / 108). s2 4-acc M=64 is 37 us
  card0, occupancy pad, loses to
  4x8 20. Stop 4-acc at M=64. s2
  4-acc NT=4 is 307 us card1,
  ~8.2x NT=2. Stop NT=4. s2 4-acc
  A-db M=256 is 37.2 us both cards,
  wash vs no-db 37.4. Stop A-db.
  K7 GDN: eager conv1d K=4 is ~115
  us launch-bound both cards;
  eager delta recurrent is 308 us
  both cards (~7x W8A8 44). Decode
  leftover after INT8 projections.
  GDN q/v W8A8 M=1 is ~46 us (v
  both-card; q 45-58 clocks), same
  class as square 44, under mixer.
  ESIMD fused conv1d is ~4.4 us
  pipe_host both cards at 1400-
  1700 (~26x eager; clocks not
  2800). ESIMD fused delta is
  7.1 us pipe_host both cards at
  2800 (~43x eager, 450 GB/s).
  Mixer 4.4+7.1 ~11.5 us under
  W8A8 46; leftover moves to
  qkvz. o-proj W8A8 is 46-47 us
  both cards, same class as
  v-proj. Fused qkv conv is
  4.4-4.9 us both cards vs trio
  ~13.8 (~3x). Clocks 1183/2800
  on fused; do not freeze 4.44
  as 2800. Packed qkv W8A8 is 96
  us both cards vs 3x 46 ~138.
  Mixer conv+delta is 8.2-8.7 us
  both cards at 2800 vs 11.5,
  spread 6.3%. Do not freeze 8.23.
  Packed qkv M=64 is 138-142 us
  both cards, wash vs 3x 46.
  ESIMD conv T=64 is 10.1 us both
  cards at 2800 vs eager 115.
  ESIMD conv T=256 is 37.7 us
  both cards at 2800, ~3.72x T=64.
  Packed qkv M=256 is 164 us both
  cards, ~1.17x M=64. ESIMD conv
  T=256 C=6144 is 38.0 us both
  cards at 2800, wash vs C=2048
  37.7 not 3x. Occupancy. ESIMD
  delta T=64 is 265-271 us both
  cards, throttle=1, ~37x decode
  7.1 not 64x. Do not freeze 265
  as 2800. ESIMD delta T=256 is
  1100-1109 us both cards,
  throttle=1, ~4.1x T=64. Hold
  retry 1086 us card1 still
  throttle=1 (2026-09-03gg). Do
  not freeze 1086 as 2800. ESIMD mixer T=64 is
  395-399 us both cards,
  throttle=1, ~1.45x sequential
  ~275. Stop two-kernel packed
  mixer at prefill. Do not freeze
  395 as 2800. ESIMD conv T=64
  C=6144 is 10.2-10.4 us both
  cards at 2800 (2026-09-03fx/ga),
  wash vs C=2048 10.1 not 3x.
  Occupancy. ESIMD conv T=64
  C=10240 is 10.5-10.7 us both
  cards at 2800 (2026-09-03gb/gc),
  wash vs C=2048 10.1 not 5x.
  Occupancy. ESIMD conv T=256
  C=10240 is 40.7-40.8 us both
  cards at 2800 (2026-09-03gd/ge),
  ~1.07x C=6144 38.0 not 5x.
  Occupancy. ESIMD chunk/WY C=16
  T=256 is 3210 us card1 at 2800
  (2026-09-03gf), cosine=1,
  ~2.92x fused 1100. Stop C=16
  vs fused. ESIMD chunk/WY C=64
  T=256 is 95420 us card0 at
  2800 (2026-09-03gh), ~88x
  fused 1086, ~30x C=16. Stop
  C=64. Stop this WY path. ESIMD
  SLM-K T=256 is 847-858 us both
  cards (2026-09-03gi/gk), ~1.27x
  fused 1086, throttle=1. New
  leftover class. Do not freeze
  847 as 2800. Row-block rb=4 is
  1034 us card1 (2026-09-03gj) at
  2800, ~1.05x fused. rb=8 is
  2060 us card0 (2026-09-03gl) at
  2800, ~2x rb=4. Stop rb=8.
  SLM-K+rb=4 is 999 us card0
  (2026-09-03gm) at 2800, ~1.18x
  SLM-K 847. Stop combine.
  SLM-K blk=32 is 832-862 us both
  cards (2026-09-03gn/go), wash vs
  blk=16 847-858, throttle=1. Do
  not freeze 832 as 2800. SLM-K
  blk=64 is 835 us card1
  (2026-09-03gp), wash vs blk=32.
  Stop larger blk. SLM-K T=64 is
  214-218 us both cards
  (2026-09-03gq/gs), ~1.23x fused
  265. card0 2800, card1
  throttle=1. Do not freeze 214
  as 2800. SLM a/b T=256 is 854
  us card1 (2026-09-03gr), wash
  vs 847. Stop a/b SLM.
  v-prefetch T=256 is 873 us
  card0 (2026-09-03gt), ~1.03x
  SLM-K 847. Stop v-prefetch.
  SLM-K T=1 blk=1 is 8.15 us
  card0 (2026-09-03gu) at 2800,
  ~1.16x fused 7.1. Stop SLM-K
  at decode. inner unroll T=256
  is 856 us card1 (2026-09-03gv),
  wash vs SLM-K 847-858,
  throttle=1. Stop inner unroll.
  SLM f32 k/q T=256 is 868 us
  card0 (2026-09-03gw), ~1.02x
  SLM-K 847. Stop f32 SLM.
  SLM db T=256 is 843 us card1
  (2026-09-03gx), throttle=1,
  wash vs 847-858. Stop
  double-buffer. tree hsum T=256
  is 426-477 us both cards
  (2026-09-03gy/hb), ~1.99x
  SLM-K 847 at card0, throttle=1.
  Clock spread 12%. Do not
  freeze 426 as 2800. SLM-K T=16
  is 58 us both cards
  (2026-09-03gz/ha), near
  T-linear 53. throttle=1. Do
  not freeze 58 as 2800. tree
  hsum T=64 is 109-125 us both
  cards (2026-09-03hc/hf). Clock
  spread 15%. Do not freeze 109
  as 2800. tree hsum T=1 is 6.09
  us card1 (2026-09-03hd) at
  2800, ~1.16x fused 7.1.
  max_abs_o=2. Do not replace
  fused 7.1. tree hsum T=16 is
  34 us both cards
  (2026-09-03he/hh), ~1.72x
  SLM-K 58, spread ~0.3%.
  throttle=1. Do not freeze 34
  as 2800. tile-fused reduce
  T=256 is 260 us card0
  (2026-09-03hg), ~1.64x tree
  hsum 426. New leftover class.
  Do not freeze 260 as 2800.
  Sibling before promote.
  s2 4x8
  M=256 N=17408 is 171 us both
  cards at 2800, throttle=1, beats
  W8A8 248 (~1.45x), loses to s4
  140. s2 4x8 M=256 K=17408 is
  201 us both cards at 2800,
  throttle=0, beats W8A8 226
  (~1.12x), loses to s4 149.
  Qwen FFN s2 M=256 map is closed
  (55.5 / 171 / 201). ESIMD
  s2xs8 4x8 A-db M=64 is 33.2 us
  both cards at 2800 (beats W8A8
  46 and s8xs4 43.3, loses to s2
  20). s2xs8 4x8 M=64 N=17408 is
  100.5 us both cards at 2800
  (beats W8A8 202 and mix 129,
  loses to s2 53.1). s2xs8 4x8
  M=64 K=17408 is 107 us both
  cards at 2800 (beats W8A8 181
  and mix 144.7, loses to s2 64).
  Qwen FFN s2xs8 M=64 map is
  closed (33.2 / 100.5 / 107).
  s2xs8 4x8 A-db M=256 is 96 us
  both cards at 2800, throttle=1,
  a loss vs W8A8 75 (~1.27x) and
  s2 55.5. Stop 4x8 mix at M=256
  prefill vs W8A8. ESIMD s2xs8 decode
  mix is 14.1 us both cards at 2800
  (beats s8 34, loses to s2xs2 11.5).
  K5 producer+GEMM N=17408 is 155 us
  both cards (prod ~11 + gemm 143,
  beats W8A8 158.1). K5 producer+GEMM
  K=17408 is 294 us both cards (prod
  ~33 + gemm 261, loses to W8A8 155.3).
  Qwen FFN producer decode map is
  closed. Mixed s8xs4 host-s32 closed
  both cards (max_abs=0 both mixes).
  GPTQ INT4 codes are s4 and feed
  ESIMD s4 both cards (max_abs=0;
  qzeros=7). ESIMD s8xs4 decode is
  22.1 us both cards at 2800 (beats
  s8 34, loses to s4 16.5). GPTQ s4
  group-scale f16 closed both cards.
  s8xs4 N=17408 is 38.6 us both cards
  (~1.74x square, under linear).
  s8xs4 K=17408 is 73.2 us both cards
  (~3.31x square). Qwen FFN s8xs4
  decode map is closed. GPTQ s4 RC=4
  decode is 29.9 us both cards (~1.81x
  s4 16.5). s8xs4 8x2-N M=64 is 114 us
  card1, a loss vs s4 33.6. GPTQ N=17408
  is 100 us both cards (~3.35x square).
  s8xs4 4x8 A-db M=64 is 43.3 us both
  cards (beats W8A8 46, loses to s4
  33.6). GPTQ K=17408 is 174.6 us
  both cards (~5.84x square, loses
  to W8A8 155.3). Qwen FFN GPTQ
  decode map is closed (29.9 / 100 /
  174.6). s8xs4 4x8 A-db M=256 is
  123 us both cards throttle=1, a
  loss vs W8A8 75 and s4 48.6. Stop
  4x8 mix at M=256 prefill. mix 4x8
  M=64 N=17408 is 129 us both cards
  (beats W8A8 202). mix 4x8 M=64
  K=17408 is 144.7 us both cards
  (beats W8A8 181). Qwen FFN mix
  M=64 map is closed (43.3 / 129 /
  144.7). GPTQ 8x2-N M=64 is 123.5
  us card0 throttle=1, a loss vs
  W8A8 46. GPTQ 8x2-N M=256 is 355
  us card1 throttle=1, a loss vs
  W8A8 75. Stop 8x2-N GPTQ at
  prefill. GPTQ 4x8 A-db M=64 is
  102.9 us card0 at 2800, beats
  8x2-N 123.5, loses to W8A8 46
  (~2.24x). GPTQ 4x8 A-db M=256 is
  303 us card1 at 2800, a loss vs
  W8A8 75. Stop GPTQ 4x8 at prefill
  vs W8A8.
- Load-time s8 NVFP4 spoof fit 8B and not 27B on one 30.3 GiB card.
  Local envelope: persist-s8 weights 29.0 GiB, resident 20.4 GiB.
- `nvfp4_gemm_w4a16` is 4-bit resident decompress, not INT4 XMX.
  Local dump (v028 so, M=64 heat, clocks not held 2800): ~37 us
  folded / ~39 us f8scale at M=1 5120. Held 2800 both cards:
  34.7 us folded / 37.8 us f8scale. M=64 both cards: 37.1 us
  folded at act 2150-2400/2800. M=256 both cards: 118 us
  folded at act 2350-2550/2800. M=1 N=17408 both cards:
  97 us folded, throttle=1. M=1 K=17408 both cards:
  101 us folded, throttle=1. M=64 N=17408 both
  cards: 142 us folded, act 2050-2300/2800.
  M=64 K=17408 both cards: 130 us folded,
  act 2100-2400/2800. M=256 N=17408 both
  cards: 394 us folded, throttle=1. M=256
  K=17408 both cards: 377 us folded,
  throttle=1.
  Stock mtp6 image lacks the
  op. Bitcast s4 is an explicit numeric negative. Sparse-hi dies
  on this ckpt (~25% overflow). Mixed s8xs4 DPAS lights and is
  host-s32 closed both cards (max_abs=0 both mixes). s2xs4
  and s8 K=16 dpas do not compile. Product LUT GEMV is a
  numeric-closed us loss. MXFP4 is absent from this checkpoint.
  Qwen3.8 GPTQ-INT4 g128 codes are s4 (q-8, ov=0) and match
  ESIMD s4 DPAS both cards. Stored qzeros are 7.
- M=1 decode is tens to hundreds of times under the compute roof.
- W8A8 decode paid ~160 activation-quant launches that W8A16 skips.
- Transformed LSC VNNI loads were bit-exact; flat prepack was not.
- Push all-reduce is a fabric prototype worth a TP=2 arm.

Until an xe2x2 run with named backend, card pin, compiler identity,
and health repeats a bullet, it stays a hypothesis.

Napkin math: compose-of-s8 loses is now measured false on the K3
tile. "we cannot beat oneDNN" is false at decode M=1 5120
scale-to-f16 s8 (34 vs 44 us, and 141.6 vs 158.1
us at N=17408) and s4 (16.5 vs 44 us), at
M=64 s4 (33.6 vs 46 us), s2 (20 vs
46 us), s2xs8 (33.2 vs 46 us), and
at M=256 s4 (48.6 vs 75 us), s2 4x8
(55.5 vs 75 us), and s2 4-acc
(37.4 vs 75 us), and at M=256
N=17408 s2 4-acc (110 vs W8A8 248)
and s2 4x8 (171 vs W8A8 248 us;
throttle=1) and K=17408 s2 (201
vs W8A8 226 us). s2xs8 M=64
N=17408 is 100.5 vs W8A8 202.
s2xs8 M=64 K=17408 is 107 vs
W8A8 181. s2xs8 M=256 is 96 vs
W8A8 75 (loss).
Different dtype than W8A8, not a W8A8 replacement. s4 M=1
N=17408 is 29.5 us both cards (1.80x N=5120, not 3.4x).
s4 M=1 K=17408 is 53.4 us both cards (~3.24x, near
K-linear). s4 M=64 N=17408 is 94.7 us both cards
(~2.81x N=5120). s4 M=64 K=17408 is 106.0 us both
cards (~3.15x). s4 M=256 N=17408 is 140.0 us both
cards (~2.88x). s4 M=256 K=17408 is 149.0 us both
cards (~3.07x). Qwen FFN s4 map is closed. s8
M=64 N=17408 is 338.9 us both cards (~4.52x
N=5120, worse than linear; s4 94.7 is ~3.58x
this). s8 M=64 K=17408 is 374.7 us both
cards (~5.00x K=5120 vs s4 106.0; slower
than wide-N 338.9). s8 M=256 N=17408 is
469.8 us both cards (~3.67x N=5120 vs s4
140.0; throttle=1 like the 128 us floor).
s8 M=256 K=17408 is 477.4 us both cards
(~3.73x K=5120 vs s4 149.0). Qwen FFN s8
prefill map is closed. s8 decode N=17408 is
141.6 us both cards (~4.16x N=5120 vs s4
29.5; worse than linear). s8 decode K=17408
is 261.6 us both cards (~7.69x K=5120 vs s4
53.4). Qwen FFN s8 map is closed. Hand s8
decode N=17408 141.6 vs oneDNN W8A8 158.1 us
both cards at 2800 (~1.12x). oneDNN W8A8 M=1
K=17408 is 155.3 us both cards (~3.53x 5120;
hand s8 261.6 loses ~1.68x). Qwen FFN oneDNN
W8A8 decode map is closed.
It is
still true for INT8 s8 at M=64
(best hand wg 4x8 A-db
75 vs 46 us; 8x2-N A-db 97-100; 4-acc wg 4x2 115 vs 4x2x4
133 vs 8x2-N 120) and at M=256 s8
(4-acc wg 4x8 128 vs 8-row 4x8 228 vs 6-acc 384-count
  210 vs A-db 4-acc 135 vs K4 W8A8 75).
Decode quant: producer+GEMM 44 us beats fusev 72; extra
~10 us over GEMM-only 34. N-wide pair
155 us both cards (beats W8A8 158.1).
K-wide pair 294 us both cards (loses to
W8A8 155.3; producer tax K-linear).
Qwen FFN producer decode map is closed.
Remaining
hypotheses: decode cannot use INT2, PP=2 cannot win decode,
we cannot beat XeTLA.
Serving-shaped work ranks by us, not TOPS%. Four B70s are
evidence-gated. Model shelf after the math floor: docs/MODELS.md.

## PP=2

No xe2x2-owned pipeline-parallel finding yet.

## Mixed 2x2

Blocked until TP=2 and PP=2 each have a passing correctness + health
run in this repo.
