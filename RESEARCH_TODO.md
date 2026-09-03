# RESEARCH_TODO.md -- xe2x2 work order

Updated 2026-09-02. P0 is the only hard gate. After P0, kernel
workstreams (K0-K6) may run in any order. Do not mix environment
refresh, kernel changes, and parallelism-map changes in one
comparison.

Campaign map (open questions, not a locked path):
`docs/KERNEL_CAMPAIGN.md`.

## Dual-card scheduling

Two B70s. Independent one-card jobs: different arms on
`gpu-run --card 0` || `--card 1`. Same arm on both cards only
for new dtype/ISA/numeric, a new floor, or clock mismatch
(`docs/AGENT_LAUNCH.md`). Two-card collectives: one agent,
pause the one-card matrix.

## P0: freeze the host baseline

- Record kernel, KMD, firmware, UMD / Compute Runtime, Level Zero
  loader, IGC, oneCCL, and PyTorch XPU identities into docs/HOST.md
  (extend the creation snapshot; do not replace it blindly).
- Record `sycl-ls` (level_zero vs opencl) and whether L0 V2 is the
  live adapter. See docs/BACKENDS.md.
- Run b70_ai_things per-card health on both B70s through gpu-run.
- Run two-rank collective health with P2P disabled.
- Confirm neither card is display-held and no live serve holds the
  lease.

Exit gate: identities recorded, both health layers green, no live
server, JOURNAL entry written.

P0 passed 2026-09-02g (`results/p0/SUMMARY.md`, docs/HOST.md freeze).
K0-K6 have both-card RESULTS. Held-clock scale-to-f16
M=1 (ap) beats W8A8 34 vs 44 us. M=64 INT8 hand floor
is wg 4x8 A-db 75 us (bc), ~1.63x W8A8 46. s4 on that
same tile is 33.6 us (bi), ~2.24x s8 and under W8A8 46
at 2800 both cards, numeric closed. M=256 hand floor
is 4-acc wg 4x8 128 us (bd), ~1.7x W8A8 75;
8-row 4x8 was 228 (bb). 384-count 6-acc is 210 us
(be), a loss. GRF256 still zebin 128.
Decode quant: producer+GEMM (ba) is 44 us.
k32 A-db on 4-acc M=256 is 135 us (bf), a tax.
4-acc wg 4x2 M-on-Y is 115 us (bg) vs 8x2-N 120;
4x2x4 no SLM is 133 us (bh), a loss. Stop M=64
4-acc s8. s4 M=1 RC=4 is 16.5 us both cards (bl),
~2.05x s8 34. s4 M=256 4-acc is 48.6 us both
cards (bm), ~2.63x s8 128, under W8A8 75.
s4 M=64 4x8 A-db stays 33.6 us (bi). s4 A-db on
4-acc M=256 is a tax 51.9 vs 48.6 (bn). s4
decode N=17408 is 29.5 us both cards (bp),
1.80x N=5120 not 3.4x. s4 decode K=17408 is
53.4 us both cards (br), ~3.24x K-linear.
s4 M=64 N=17408 4x8 A-db is 94.7 us both
cards (bt), ~2.81x N=5120. s4 M=64 K=17408 is
106.0 us both cards (bv), ~3.15x K-linear.
s4 M=256 N=17408 4-acc is 140.0 us both cards
(bx), ~2.88x N=5120. s4 M=256 K=17408 is 149.0
us both cards (bz), ~3.07x K-linear. Qwen FFN
s4 map is closed. s8 M=64 N=17408 is 338.9 us
both cards (cb), ~4.52x N=5120 vs s4 94.7.
s8 M=64 K=17408 is 374.7 us both cards (cd),
~5.00x K-linear vs s4 106.0. s8 M=256 N=17408
is 469.8 us both cards (cf), ~3.67x N=5120 vs
s4 140.0, throttle=1. s8 M=256 K=17408 is
477.4 us both cards (ch), ~3.73x K-linear vs
s4 149.0, throttle=1. Qwen FFN s8 prefill map
is closed. s8 decode N=17408 is 141.6 us
both cards (cj), ~4.16x N=5120 vs s4 29.5.
s8 decode K=17408 is 261.6 us both cards (cl),
~7.69x K-linear vs s4 53.4. Qwen FFN s8 map
is closed. oneDNN W8A8 M=1 N=17408 is 158.1 us
both cards (cn) at 2800 vs hand s8 141.6.
oneDNN W8A8 M=1 K=17408 is 155.3 us both
cards (cp) at 2800 vs hand s8 261.6 (hand
loses ~1.68x). Qwen FFN oneDNN W8A8 decode
map is closed. K6 nibble_lut_sc on the s8
RC=4 decode tile is 158 us both cards (cq)
at 2800, cosine=1 max_abs=0, ~4.65x s8 34.
Packed E2M1 stays in HBM. Never bitcast s4.
iselect 16-entry table LUT is 1022 us both
cards (cr), a loss vs merge 158. Stop gather
tables. Scalar two-launch unpack is 265 us
both cards (2026-09-03a), ~1.67x the 158 us
LUT; s8ctrl 34.5; throttle=1. k64 combined
load is 169 us both cards (2026-09-03b), a
small loss. Vectorized unpack is 314.6 us
both cards (2026-09-03f), a loss vs scalar
265 and LUT 158. Stop that unpack path.
E2M1 two-term s4 decode is 28.5 us both
cards (2026-09-03e) vs s4 16.5 vs s8 34,
A=s4. Keep fused 158 us LUT for s8-A.
compose N=17408 is 103.5 us both cards
(2026-09-03i), ~3.63x square vs s4 29.5.
compose K=17408 is 193.6 us both cards
(2026-09-03j), ~6.79x vs s4 53.4. Qwen FFN
compose decode map is closed. A=s4.
compose M=64 8x2-N is 217.9 us card0
(2026-09-03k), a loss vs s4 33.6. compose
M=256 is ~607 us both cards (2026-09-03m),
throttle=1, a loss vs s4 48.6. Stop 8x2-N
compose at prefill. nibble_lut_sc M=64 is
~656 us both cards (2026-09-03o), a loss
vs s8 75. Stop 8x2-N LUT at prefill too.
compose on s4 4x8 A-db M=64 is 68.7 us
both cards (2026-09-03q), ~2.04x s4 33.6,
~3.17x faster than 8x2-N 217.9. Beats s8
75, loses to W8A8 46. A=s4. compose 4x8
A-db M=256 is 194.9 us both cards
(2026-09-03t), ~3.12x 8x2-N 607, ~4.0x
s4 4-acc 48.6, loses to W8A8 75. nibble
LUT on s8 4x8 A-db M=64 is 392.4 us both
cards (2026-09-03u), ~1.67x 8x2-N 656,
still ~5.23x s8 75. compose 4x8 A-db
M=64 N=17408 is 326.9 us both cards
(2026-09-03x), ~4.76x square vs s4 94.7.
compose M=64 K=17408 is 403.4 us both
cards (2026-09-03y), ~5.87x square vs s4
106.0, loses to s8 374.7. Qwen FFN compose
M=64 map is closed. A=s4. compose 4x8
A-db M=256 N=17408 is 984.3 us both cards
(2026-09-03ac), ~5.05x square vs s4 140,
~2.10x s8 469.8. compose M=256 K=17408
is 968.7 us both cards (2026-09-03ab),
~4.97x square vs s4 149, ~2.03x s8 477.4.
throttle=0. Qwen FFN compose M=256 map
is closed. A=s4. 4x8 A-db loses to s4
and s8 at FFN prefill M=256.
compose on s4 4-acc M=256 is 411 us
card0 (2026-09-03af), a loss vs 4x8
compose 194.9 and ~8.46x s4 48.6.
nibble LUT 4x8 A-db M=64 N=17408 is
1032 us both cards (2026-09-03ah),
~2.63x square vs s8 338.9. LUT M=64
K=17408 is 1333 us both cards (2026-09-03aj),
K-linear ~3.40x vs s8 374.7. Qwen FFN LUT
M=64 map is closed. LUT 4x8 A-db M=256 is
1203 us both cards (2026-09-03al), ~3.07x
M=64 vs s8 128. closed-form LUT on 4x8
A-db M=64 is 331.6 us both cards
(2026-09-03an), ~1.18x merge 392.4.
closed-form LUT 4x8 A-db M=256 is 1083
us both cards (2026-09-03ap), ~3.27x
M=64, ~1.11x merge 1203. closed-form
LUT 4x8 A-db M=64 N=17408 is 880 us
both cards (2026-09-03ar), ~2.65x
square vs s8 338.9, throttle=1.
closed-form LUT 4x8 A-db M=64 K=17408
is 1125 us both cards (2026-09-03at),
K-linear ~3.39x vs s8 374.7. Qwen FFN
closed-form LUT M=64 map is closed.
closed-form LUT 4x8 A-db M=256 N=17408
is 3138 us both cards (2026-09-03av),
~2.90x square vs s8 469.8, throttle=1.
closed-form LUT 4x8 A-db M=256 K=17408
is 3428 us both cards (2026-09-03ax),
~3.17x square vs s8 477.4, throttle=1.
Qwen FFN closed-form LUT M=256 map is
closed. 4x8 LUT loses to s8/s4/compose
at FFN prefill. Held-clock
nvfp4_gemm_w4a16 M=1 is 34.7 us both
cards (2026-09-03az) at 2800, bf16-A,
same us class as s8 34, under W8A8 44.
nvfp4_gemm_w4a16 M=64 is 37.1 us both
cards (2026-09-03bb), act 2150-2400/2800,
~1.07x M=1, under W8A8 46. nvfp4_gemm_w4a16
M=256 is 118 us both cards (2026-09-03bd),
~3.18x M=64, loses to W8A8 75.
nvfp4_gemm_w4a16 M=1 N=17408 is 97 us
both cards (2026-09-03bf), ~2.80x
square, beats s8 141.6, throttle=1.
nvfp4_gemm_w4a16 M=1 K=17408 is 101 us
card1 (2026-09-03bg), ~2.92x square,
beats s8 261.6, throttle=1. One-card.
K6 12-idea sprint (2026-09-03ae):
closed-form LUT 134.8 us is the new
Family-A floor. Bitcast s4 is an
explicit negative. Sparse-hi dies
(~25% overflow). Mixed s8xs4 lights;
s2xs4 and s8 K=16 dpas refuse.
Product LUT GEMV is a numeric-closed
us loss. oneDNN nvfp4_gemm_w4a16
lights at ~37 us unheld / 34.7 us
held 2800 both. MXFP4 absent.
Persist-s8 29.0 GiB vs resident 20.4.
Next: split. card0: sibling nvfp4_gemm_w4a16
M=1 K=17408 (throttle=1 last). card1:
held-clock nvfp4_gemm_w4a16 M=64 N=17408.
Loop every 10m.

## After P0: kernel workstreams (parallelizable)

Pick a directory, one question per run. Details in each README.

- K0 `kernels/roofline/` -- copy + GEMM + GEMV roofs
- K1 `kernels/onednn_isa/` -- dump incumbent Intel ops
- K2 `kernels/esimd_dpas/` -- hand s8 / s4 / s2 DPAS, light INT2
- K3 `kernels/precision_compose/` -- INT2/INT4 as INT8 or E2M1
- K4 `kernels/w8_compare/` -- FP8 W8A16 vs INT8 W8A16 vs W8A8
- K5 `kernels/epilogue_quant/` -- INT8 without ~160 quant launches
- K6 `kernels/nvfp4/` -- every NVFP4 spoof / LUT / split
- K7 `kernels/gdn/` -- GDN hybrid leftover (Qwen3.8 is not plain attn)

Launch pairing: `docs/AGENT_LAUNCH.md`. A literature agent may fetch
`docs/REFERENCES.md` campaign papers with no GPU lease.

Exit per workstream: JOURNAL entry, artifact under results/. Promote
a new floor to FINDINGS.md after the sibling card has run it, or
immediately when the both-card rule already required two cards.
Schedule steals on an already-matched family may JOURNAL from one
held-clock card.
Napkin priors (compose loses, INT2 is useless, ...) are CONFIG, not
RESULT. Measure.

## After a math floor: models

See `docs/MODELS.md`. Do not start a serve to skip K0-K3.

- Dense: Qwen3.8-27B (already on disk: BF16, FP8, W8A8, INT4, NVFP4).
- MoE body: Qwen3.6-35B-A3B when fetched; Ornith-1.5-35B-A3B is the
  on-disk stand-in of the same size class.
- Compact MoE: Gemma 4 26B A4B (fetch when needed).
- Stretch: Qwen3.8-Flash-Next (NVMe expert/PLE, not a 2xB70 resident).

Quants: INT8 W8A16 / W8A8, integer INT4, NVFP4 spoof. FP8 W8A16 is
the dense incumbent control.

## P2: TP=2  (`parallel/tp2/`)

Needs P0. Happier if K0 exists so collective GB/s has a roof.

- Synthetic all-reduce / all-gather / send-recv.
- P2P off first. P2P on only as a labeled control.
- Push all-reduce, fused RMSNorm+AR, and minimum call count are
  first-class arms, not footnotes.
- Tiny sharded matmul only after the synthetic collective is correct.

Exit gate: correctness, matched rank evidence, teardown, post-health.

## P3: PP=2  (`parallel/pp2/`)

- Two-stage synthetic activation handoff, one stage per card.
- Bubble, copy path, stage-memory split.
- Tiny-model PP=2 only after the synthetic handoff is correct.

Exit gate: same as P2.

## P4: mixed 2x2  (`parallel/2x2/`)

Only after P2 and P3 pass on this host with the current UMD.
TP=2 inside a two-stage pipeline, then the reverse, as separate
experiments.

## Standing bans

- Do not skip health to get a speed number.
- Do not reuse quarantined wheels, .so files, or compiler caches from
  b70_ai_things as proof that xe2x2 work is clean.
- Do not promote a serving wrapper from this repo. Hand findings back
  to b70_ai_things.
- Do not mix a slot-move topology A/B into a kernel matrix.
- Do not cite sibling-lab tok/s or ISA notes as FINDINGS until this
  host repeats them.
- Do not treat XeTLA / oneDNN / an Intel paper as a ceiling.
- Rank serving-shaped work by us, not TOPS% or BW%.
- Do not assume TP=2 for every op. Four B70s wait on evidence.
