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
~2.24x s8 at M=64 4x8 A-db / 2800 (33.6 vs 75 us). INT2 DPAS
exists (s2xs2 and s2xs8 closed). Remaining:

- ESIMD `dpas<s4,s4>` ~2x s8 MAC rate -- 1.49x at 1024^3; ~2.24x
  wall time at M=64 4x8 A-db held 2800. Tile-dependent.
- Xe2 has no native FP8 XMX -- reproduced: fp8_gemm_w8a16 is
  E4M3 unpack + bf16 dpas.8x8 (K1 JIT dump).
- NVFP4 E2M1 x2 is exact int8; +-12 overflows s4; bitcast is wrong.
  Local: nibble LUT -> s8 DPAS closed (K6); two-term s4 compose
  closed (K3). In-register VNNI4 pack closed; scalar LUT lost
  us; simd LUT is ~6-8x that arm (304-406 us) and still clock-
  bound vs two-launch unpack.
- Load-time s8 NVFP4 spoof fit 8B and not 27B on one 30.3 GiB card.
- `nvfp4_gemm_w4a16` is 4-bit resident decompress, not INT4 XMX.
- M=1 decode is tens to hundreds of times under the compute roof.
- W8A8 decode paid ~160 activation-quant launches that W8A16 skips.
- Transformed LSC VNNI loads were bit-exact; flat prepack was not.
- Push all-reduce is a fabric prototype worth a TP=2 arm.

Until an xe2x2 run with named backend, card pin, compiler identity,
and health repeats a bullet, it stays a hypothesis.

Napkin math: compose-of-s8 loses is now measured false on the K3
tile. "we cannot beat oneDNN" is false at decode M=1 5120
scale-to-f16 s8 (34 vs 44 us). s4xs4 on the M=64 4x8 A-db
tile is 33.6 us, under W8A8 46 in wall time (different dtype,
not a W8A8 replacement). It is still true for INT8 s8 at M=64
(best hand wg 4x8 A-db
75 vs 46 us; 8x2-N A-db 97-100; 4-acc wg 4x2 115 vs 4x2x4
133 vs 8x2-N 120) and at M=256 s8
(4-acc wg 4x8 128 vs 8-row 4x8 228 vs 6-acc 384-count
  210 vs A-db 4-acc 135 vs K4 W8A8 75).
Decode quant: producer+GEMM 44 us beats fusev 72; extra
~10 us over GEMM-only 34. Remaining
hypotheses: decode cannot use INT2, PP=2 cannot win decode,
we cannot beat XeTLA.
Serving-shaped work ranks by us, not TOPS%. Four B70s are
evidence-gated. Model shelf after the math floor: docs/MODELS.md.

## PP=2

No xe2x2-owned pipeline-parallel finding yet.

## Mixed 2x2

Blocked until TP=2 and PP=2 each have a passing correctness + health
run in this repo.
