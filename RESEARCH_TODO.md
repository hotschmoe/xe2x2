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
both cards (2026-09-03bh), ~2.92x
square, beats s8 261.6, throttle=1.
Qwen FFN w4a16 decode map is closed.
nvfp4_gemm_w4a16 M=64 N=17408 is 142 us
both cards (2026-09-03bj), ~3.82x square,
beats s8 338.9, act 2050-2300/2800.
nvfp4_gemm_w4a16 M=64 K=17408 is 130 us
both cards (2026-09-03bl), ~3.51x square,
~K-linear, beats s8 374.7, act 2100-2400.
Qwen FFN w4a16 M=64 map is closed.
nvfp4_gemm_w4a16 M=256 N=17408 is 394 us
both cards (2026-09-03bn), ~3.34x square,
~N-linear, beats s8 469.8, throttle=1.
nvfp4_gemm_w4a16 M=256 K=17408 is 377 us
both cards (2026-09-03bp), ~3.19x square,
under K-linear, beats s8 477.4,
throttle=1. Qwen FFN w4a16 M=256 map
is closed. oneDNN W8A8 M=256 N=17408
is 248 us both cards (2026-09-03br),
~3.31x square, beats w4a16 394,
throttle=1. oneDNN W8A8 M=256 K=17408
is 226 us both cards (2026-09-03bt),
~3.01x square, under K-linear, beats
w4a16 377, throttle=1. Qwen FFN W8A8
M=256 map is closed. oneDNN W8A8 M=64
N=17408 is 202 us both cards (2026-09-03bv),
~4.39x square, loses to w4a16 142,
throttle=1. oneDNN W8A8 M=64 K=17408
is 181 us both cards (2026-09-03bx),
~3.93x square, loses to w4a16 130,
throttle=1. Qwen FFN W8A8 M=64 map is
closed. ESIMD s2 RC=4 decode is 11.5 us
both cards (2026-09-03bz) at 2800,
cosine=1 max_abs=0, ~1.43x s4 16.5.
ESIMD s2xs8 decode mix is 14.1 us
both cards (2026-09-03cb) at 2800,
beats s8 34, loses to s2xs2 11.5.
K5 producer+GEMM N=17408 is 155 us
both cards (2026-09-03cd), prod ~11
+ gemm 143, beats W8A8 158.1.
K5 producer+GEMM K=17408 is 294 us
both cards (2026-09-03cf), prod ~33
+ gemm 261, loses to W8A8 155.3.
Qwen FFN producer decode map is
closed. Mixed s8xs4 host-s32 closed
both cards (2026-09-03ch), max_abs=0
both mixes. GPTQ INT4 codes feed
ESIMD s4 both cards (2026-09-03cj),
s4_ov=0 max_abs=0 qzeros=7. ESIMD
s8xs4 decode is 22.1 us both cards
(2026-09-03cl) at 2800, beats s8 34,
loses to s4 16.5. GPTQ group-scale
f16 closed both cards (2026-09-03cn).
s8xs4 N=17408 is 38.6 us both cards
(2026-09-03cp) at 2800, ~1.74x square
not 3.4x, loses to s4 29.5. s8xs4 K=17408 is 73.2 us both cards
(2026-09-03cr) at 2800, ~3.31x square.
Qwen FFN s8xs4 decode map is closed
(22.1 / 38.6 / 73.2). GPTQ s4 RC=4
decode is 29.9 us both cards
(2026-09-03ct) at 2800, ~1.81x s4
16.5, beats s8 34. s8xs4 8x2-N M=64
is 114 us card1 (2026-09-03cu), a
loss vs s4 4x8 33.6. GPTQ N=17408 is
100 us both cards (2026-09-03cy) at
2800, ~3.35x square. s8xs4 4x8 A-db
M=64 is 43.3 us both cards
(2026-09-03cx), beats W8A8 46,
loses to s4 33.6. GPTQ K=17408 is
174.6 us both cards (2026-09-03dc)
at 2800, ~5.84x square, loses to
W8A8 155.3. Qwen FFN GPTQ decode
map is closed (29.9 / 100 / 174.6).
s8xs4 4x8 A-db M=256 is 123 us both
cards (2026-09-03db), throttle=1, a
loss vs W8A8 75 and s4 48.6. Stop
4x8 mix at M=256 prefill. mix 4x8
M=64 N=17408 is 129 us both cards
(2026-09-03dg), ~2.98x square,
beats W8A8 202. mix 4x8 M=64
K=17408 is 144.7 us both cards
(2026-09-03df) at 2800, ~3.34x
square, beats W8A8 181. Qwen FFN
mix M=64 map is closed
(43.3 / 129 / 144.7). GPTQ 8x2-N
M=64 is 123.5 us card0
(2026-09-03dh), throttle=1, a loss
vs W8A8 46. GPTQ 8x2-N M=256 is
355 us card1 (2026-09-03di),
throttle=1, a loss vs W8A8 75.
Stop 8x2-N GPTQ at prefill. GPTQ
4x8 A-db M=64 is 102.9 us card0
(2026-09-03dj) at 2800, beats
8x2-N 123.5, loses to W8A8 46
(~2.24x). GPTQ 4x8 A-db M=256 is
303 us card1 (2026-09-03dk) at
2800, a loss vs W8A8 75. Stop
GPTQ 4x8 at prefill vs W8A8. s2 4x8
A-db M=64 is 20 us both cards
(2026-09-03dm) at 2800, beats s4
33.6 (~1.68x) and W8A8 46 (~2.21x).
New M=64 hand floor. s2 4x8 M=64
N=17408 is 53.1 us both cards
(2026-09-03dq) at 2800, ~2.65x
square, beats W8A8 202 (~3.81x).
s2 4x8 M=64 K=17408 is 64 us both
cards (2026-09-03dp) at 2800,
~3.20x square, beats W8A8 181
(~2.83x). Qwen FFN s2 M=64 map is
closed (20 / 53.1 / 64). s2 4x8
A-db M=256 is 55.5 us both cards
(2026-09-03dt) at 2800, ~2.77x
M=64, beats W8A8 75 (~1.35x),
loses to s4 4-acc 48.6 (~1.14x).
s2xs8 4x8 A-db M=64 is 33.2 us
both cards (2026-09-03du) at 2800,
beats W8A8 46 (~1.39x) and s8xs4
43.3, loses to s2 20 (~1.66x).
s2 4x8 M=256 N=17408 is 171 us
both cards (2026-09-03dx) at 2800,
throttle=1, ~3.08x square, beats
W8A8 248 (~1.45x), loses to s4
140 (~1.22x). s2xs8 4x8 M=64
N=17408 is 100.5 us both cards
(2026-09-03dy) at 2800, ~3.03x
square, beats W8A8 202 (~2.01x)
and mix 129, loses to s2 53.1
(~1.89x). s2 4x8 M=256 K=17408
is 201 us both cards (2026-09-03eb)
at 2800, throttle=0, ~3.62x
square, beats W8A8 226 (~1.12x),
loses to s4 149 (~1.35x). Qwen
FFN s2 M=256 map is closed
(55.5 / 171 / 201). s2xs8 4x8
M=64 K=17408 is 107 us both cards
(2026-09-03ec) at 2800, ~3.23x
square, beats W8A8 181 (~1.69x)
and mix 144.7, loses to s2 64
(~1.67x). Qwen FFN s2xs8 M=64
map is closed (33.2 / 100.5 / 107).
s2xs8 4x8 A-db M=256 is 96 us
both cards (2026-09-03ee) at 2800,
throttle=1, ~2.88x M=64, beats
mix 123, loses to s2 55.5 (~1.72x)
and W8A8 75 (~1.27x). Stop 4x8
mix at M=256 prefill vs W8A8. s2
4-acc M=256 is 37.4 us both cards
(2026-09-03eg) at 2800, new M=256
hand floor, beats s4 48.6 (~1.30x)
and W8A8 75 (~2.01x). Napkin 29
miss. s2 4-acc M=256 N=17408 is
110 us both cards (2026-09-03ej)
at 2800, ~2.94x square, beats s4
140 (~1.27x) and W8A8 248
(~2.25x). s2 4-acc M=256 K=17408
is 108 us both cards (2026-09-03ek)
at 2800, ~2.89x square, beats s4
149 (~1.38x) and W8A8 226
(~2.09x). Qwen FFN s2 4-acc M=256
map is closed (37.4 / 110 / 108).
s2 4-acc M=64 is 37 us card0
(2026-09-03el) at 2800, same us
as M=256, loses to 4x8 20 (~1.86x).
Stop 4-acc at M=64. s2 4-acc NT=4
M=256 is 307 us card1 (2026-09-03em)
at 2800, ~8.2x NT=2. Stop NT=4.
s2 4-acc A-db M=256 is 37.2 us
both cards (2026-09-03eo) at 2800,
wash vs no-db 37.4. Stop A-db on
s2 4-acc. Floor stays 37.4.
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
K7 GDN inventory (2026-09-03es):
Qwen3.8 48 GDN + 16 full attn.
Eager conv1d K=4 is ~115 us
launch-bound both cards. Eager
delta recurrent is 308 us both
cards (~7x W8A8 44). State 72 MiB
all layers. GDN q-proj W8A8 M=1
5120x2048 is 45-58 us (clocks).
v-proj 5120x6144 is 46 us both
cards (2026-09-03ev). ESIMD fused
conv1d is ~4.4 us pipe_host both
cards at 1400-1700 (2026-09-03fa),
~26x eager, cosine=1. Do not
freeze 4.4 as 2800. ESIMD fused
delta is 7.1 us pipe_host both
cards at 2800 (2026-09-03ez),
~43x eager, 450 GB/s. Mixer
4.4+7.1 ~11.5 us under W8A8 46.
Leftover is qkvz. o-proj W8A8
M=1 6144x5120 is 46-47 us both
cards (2026-09-03fe). Fused qkv
conv C=10240 is 4.4-4.9 us both
cards vs trio ~13.8 (2026-09-03fd).
Clocks 1183/2800 on fused. Do
not freeze 4.44 as 2800. Packed
qkv W8A8 n=10240 is 96 us both
cards (2026-09-03fi) vs 3x 46
~138. Mixer conv+delta is 8.2-8.7
us both cards at 2800
(2026-09-03fh) vs 11.5, spread
6.3%. Do not freeze 8.23. Packed
qkv M=64 is 138-142 us both
cards (2026-09-03fm), wash vs
3x 46. ESIMD conv T=64 is 10.1
us pipe_host both cards at 2800
(2026-09-03fl) vs eager 115.
ESIMD conv T=256 is 37.6 us
pipe_host card0 at 2800
(2026-09-03fn), ~3.72x T=64.
Packed qkv M=256 is 164 us both
cards (2026-09-03fp), ~1.17x
M=64. ESIMD conv T=256 is 37.7
us both cards at 2800
(2026-09-03fq). ESIMD conv
T=256 C=6144 is 38.0 us both
cards at 2800 (2026-09-03fr/fu),
wash vs C=2048 37.7 not 3x.
ESIMD delta T=64 is 265-271 us
both cards (2026-09-03fs/ft),
throttle=1, ~37x decode 7.1
not 64x. Do not freeze 265 as
2800. ESIMD delta T=256 is
1100-1109 us both cards
(2026-09-03fv/fw), throttle=1,
~4.1x T=64. Prefill leftover.
Do not freeze 1100 as 2800.
ESIMD mixer T=64 is 395-399 us
both cards (2026-09-03fy/fz),
throttle=1, ~1.45x sequential
~275. Stop two-kernel packed
mixer at prefill. Do not freeze
395 as 2800. ESIMD conv T=64
C=6144 is 10.2-10.4 us both
cards (2026-09-03fx/ga) at 2800,
wash vs C=2048 10.1 not 3x.
ESIMD conv T=64 C=10240 is
10.5-10.7 us both cards
(2026-09-03gb/gc) at 2800, wash
vs C=2048 10.1 not 5x.
Occupancy. ESIMD conv T=256
C=10240 is 40.7-40.8 us both
cards (2026-09-03gd/ge) at 2800,
~1.07x C=6144 38.0 not 5x.
ESIMD chunk/WY C=16 T=256 is
3210 us card1 (2026-09-03gf) at
2800, cosine=1, ~2.92x fused
1100. Stop C=16 vs fused. Fused
delta T=256 hold retry is 1086
us card1 (2026-09-03gg), still
throttle=1. Chunk/WY C=64 T=256
is 95420 us card0 (2026-09-03gh)
at 2800, ~88x fused 1086. Stop
C=64. Stop this WY path. ESIMD
SLM-K T=256 is 847-858 us both
cards (2026-09-03gi/gk), ~1.27x
fused 1086, throttle=1. Do not
freeze 847 as 2800. Row-block
rb=4 is 1034 us card1
(2026-09-03gj) at 2800. rb=8 is
2060 us card0 (2026-09-03gl) at
2800, ~2x rb=4. Stop rb=8.
SLM-K+rb=4 is 999 us card0
(2026-09-03gm) at 2800, loses to
SLM-K 847. Stop combine. SLM-K
blk=32 is 832-862 us both cards
(2026-09-03gn/go), wash vs
blk=16 847-858, throttle=1. Do
not freeze 832 as 2800. SLM-K
blk=64 is 835 us card1
(2026-09-03gp), wash vs 832.
Stop larger blk. SLM-K T=64 is
214 us card0 at 2800
(2026-09-03gq), ~1.24x fused
265, T-linear vs 847. SLM a/b
T=256 is 854 us card1
(2026-09-03gr), wash vs 847.
Stop a/b SLM. SLM-K T=64 is
214-218 us both cards
(2026-09-03gq/gs), ~1.23x fused
265. Do not freeze 214 as 2800.
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
SLM-K 847. Stop f32 SLM. SLM
db T=256 is 843 us card1
(2026-09-03gx), throttle=1,
wash vs 847-858. Stop
double-buffer. tree hsum T=256
is 426-477 us both cards
(2026-09-03gy/hb), ~1.99x
SLM-K 847 at card0, throttle=1.
Clock spread 12% (2417 vs
2700). Do not freeze 426 as
2800. SLM-K T=16 is 58 us
both cards (2026-09-03gz/ha),
near T-linear 53. throttle=1.
Do not freeze 58 as 2800.
tree hsum T=64 is 109 us card0
(2026-09-03hc) at 2800, ~1.97x
SLM-K 214. Napkin 107. Sibling
before promote. tree hsum T=1
is 6.09 us card1 (2026-09-03hd)
at 2800, ~1.16x fused 7.1.
max_abs_o=2 vs fused 0. Do not
replace fused 7.1. tree hsum
T=16 is 34 us card0
(2026-09-03he), ~1.72x SLM-K
58, throttle=1. Do not freeze
34 as 2800. Sibling before
promote. tree hsum T=64 is
109-125 us both cards
(2026-09-03hc/hf). Clock
spread 15% (2450 vs 2800). Do
not freeze 109 as 2800. tree
hsum T=16 is 34 us both cards
(2026-09-03he/hh), ~1.72x
SLM-K 58, spread ~0.3%.
throttle=1. Do not freeze 34
as 2800. tile-fused reduce
T=256 is 260 us card0
(2026-09-03hg), ~1.64x tree
hsum 426. New leftover class.
Do not freeze 260 as 2800.
Sibling before promote. tile-
fused T=64 is 67 us card0
(2026-09-03hi), napkin 66.
Clocks 550 to 2700. Do not
freeze 67 as 2800. Hold retry.
tile-fused T=256 is 260-294 us
both cards (2026-09-03hg/hj).
Clock spread 13% (2283 vs
2600). Do not freeze 260 as
2800. tile-fused T=16 is 22 us
card0 (2026-09-03hk), ~1.56x
tree hsum 34, napkin 21.
throttle=1. Do not freeze 22
as 2800. Sibling before
promote. tile-fused T=64 is
67-77 us both cards
(2026-09-03hi/hl). Clock
spread 15% (2367 vs 2700). Do
not freeze 67 as 2800. tile-
fused T=16 is 22 us both cards
(2026-09-03hk/hn), ~1.56x tree
hsum 34, spread ~2%. throttle=1.
Do not freeze 22 as 2800.
slmht tt unroll T=256 is 276
us card0 (2026-09-03hm),
~1.06x slmht 260. Stop inner
unroll vs slmht. pack a/b/v
T=256 is 266 us card0
(2026-09-03ho), ~1.02x slmht
260. Stop pack a/b/v vs
slmht. tile-fused T=1 is 5.54
us card1 at 2800 (2026-09-03hp)
vs fused 7.1, max_abs_o=2.
Do not replace fused 7.1.
slmht blk=8 T=256 is 269 us
card0 (2026-09-03hq), ~1.04x
slmht 260. Stop blk=8 vs
slmht. T=1 tile-fused scalar
hsum is 6.09 us card1 at 2800
(2026-09-03hr), wash vs tree
hsum, max_abs_o=2. Numeric is
the fused acc. Stop scalar
hsum vs reduce. slmht blk=32
T=256 is 252 us card0 at 2600
(2026-09-03hs), ~1.03x slmht
260. packed-o T=256 is 247 us
card1 at 2700 (2026-09-03ht),
~1.05x slmht 260. Possible
leftover cuts. Do not freeze
252 or 247 as 2800. packed-o
sibling is 297 us card0
(2026-09-03hu) at 2233
throttle=1, spread ~20% vs 247.
Clock-linear. Stop packed-o vs
slmht. blk=32 sibling is 292 us
card1 (2026-09-03hv) at 2233
throttle=1, spread ~16% vs 252.
Clock-linear. Do not freeze 252
as 2800. spin=4000 throttles
T=256. slmht 2-row T=256 is
327 us card0 at 2800
(2026-09-03hw), ~1.26x slmht
260. Stop 2-row vs slmht.
blk=32 T=64 is 67 us card1
(2026-09-03hx), wash vs slmht
67. Stop blk=32 at T=64. slmht
SLM-db T=256 is 268 us card0
(2026-09-03hy), ~1.03x slmht
260. Stop SLM-db vs slmht.
tile-fused T=8 is 13 us card1
at 2750 (2026-09-03hz), ~1.76x
fused 7.1, near half T=16 22.
throttle=1. Do not freeze 13 as
2800. tile-fused T=8 is 13 us
both cards (2026-09-03hz/ia),
spread ~1%. throttle=1. Do not
freeze 13 as 2800. tile-fused
T=32 is 39 us card1
(2026-09-03ib), napkin 40.
throttle=1. Do not freeze 39 as
2800. tile-fused T=32 is 39 us
both cards (2026-09-03ib/ic),
spread ~1%. throttle=1. Do not
freeze 39 as 2800. slmht32
T=128 is 125 us card1 at 2700
(2026-09-03id), napkin 130.
Do not freeze 125 as 2800.
slmht32 T=128 is 125-130 us
both cards (2026-09-03id/ie)
at 2600-2700, spread ~4%. Do
not freeze 125 as 2800.
tile-fused T=128 is 127 us
card1 at 2700 (2026-09-03if),
wash vs slmht32 125. Stop
blk=32 at T=128. tile-fused
T=128 is 127-131 us both cards
(2026-09-03if/ig) at 2600-2700,
spread ~4%. Do not freeze 127
as 2800. T-map closed. mixer
T=256 is 1557 us card1
(2026-09-03ih), ~5.2x seq 298.
Stop packed mixer at T=256.
mixer-slmht T=256 is 471 us
card0 (2026-09-03ii) at 2800,
~3.31x packed 1557, ~1.58x
seq 298. First fuse. Sibling
before promote. skip-hi T=256
is 330 us card1 (2026-09-03ij)
at 2800, ~1.27x slmht 260.
Stop skip-hi vs slmht leftover.
mixer-slmht T=256 is 471 us
both cards (2026-09-03ii/il)
at 2800, ~3.31x packed 1557,
~1.58x seq 298. Promote.
mixer-slmht T=64 is 117 us
card0 (2026-09-03ik), ~3.36x
packed 395, T-linear vs 471.
throttle=1. Do not freeze 117
as 2800. Sibling before citing
the map. mixer-slmht T=64 is
116-117 us both cards
(2026-09-03ik/in), throttle=1.
Do not freeze 117 as 2800.
mixer-slmht T=128 is 232 us
card0 (2026-09-03im), napkin
235, T-linear. Clocks ramped
1600 to 2800. Do not freeze
232 as 2800. mixer-slmht T=128
is 232 us both cards
(2026-09-03im/ip), card1 at
2800. Napkin 235. T-linear.
mixer-slmht T=32 is 60 us
card0 (2026-09-03io), napkin
58. throttle=1. Do not freeze
60 as 2800. mixer-slmht T=32 is
59-60 us both cards
(2026-09-03io/ir), throttle=1.
Do not freeze 60 as 2800.
mixer-slmht T=16 is 31 us
card0 (2026-09-03iq), napkin
29. throttle=1. Do not freeze
31 as 2800. mixer-slmht T=16 is
31 us both cards
(2026-09-03iq/it), throttle=1.
T-map blk=16 closed. Do not
freeze 31 as 2800. conv T=16
C=10240 is 5.7 us card0
(2026-09-03is) at 1650. seq
~28 vs mixer 31. Do not freeze
5.7 as 2800. conv T=16 C=10240
is 4.8 us card1 at 2800
(2026-09-03iv). Card0 5.7 at
1650. Clock-spread. seq ~27 vs
mixer 31. conv T=32 C=10240 is
5.9 us card0 (2026-09-03iu) at
2800. seq ~45 vs mixer 60.
Sibling before citing the map.
conv T=32 C=10240 is 5.8-5.9 us
both cards (2026-09-03iu/ix) at
2800. seq ~45 vs mixer 60.
conv T=128 C=10240 is 20 us
card0 (2026-09-03iw) at 2800,
napkin 20. seq ~147 vs mixer
232. conv T=128 C=10240 is 20 us
both cards (2026-09-03iw/iz) at
2800. T-map C=10240 closed.
mixer L2-out T=256 is 271 us
card0 (2026-09-03iy), wash vs
slmht 260. Packed tax ~4%.
Device L2 is the mixer leftover.
Do not freeze 271 as 2800.
mixer L2-out T=256 is 267-271
us both cards (2026-09-03iy/jb).
Packed tax ~4%. mixer L2-once
T=256 is 312-327 us both cards
(2026-09-03ja/jc), beats mixer
471, loses to seq 298. Extra
launch. Do not freeze 327 as
2800.
mixer conv-L2 fuse T=256 is
358 us card0 (2026-09-03jd), a
loss vs L2-once 327. Stop
per-t SLM L2 in conv. Seq 298
stays the T=256 leftover.
packed qkv ESIMD s8 M=1 is
74 us both cards at 2800
(2026-09-03je/jg), beats W8A8
96. packed qkv s8 M=64 is 214
us card1 (2026-09-03jf) at
2800, a loss vs W8A8 140.
Stop 4x8 A-db. packed qkv s8
M=256 is 274 us card0
(2026-09-03jh), a loss vs
W8A8 164. Stop 4-acc 4x8.
Do not freeze 274 as 2800.
mixer conv-L2r T=256 is 531 us
card0 (2026-09-03ji), a loss
vs L2-once 327 and seq 298.
Stop register-head FIR.
o-proj s8 M=1 is 62 us card1
(2026-09-03jj) at 2800, a loss
vs W8A8 47. Stop this tile.
q/k s8 n=2048 is 28-class both
(2026-09-03jk/jn). v-proj s8
is 53 us (jl), a loss vs 46.
Packed 74 beats split q+k+v
~109. Stop split vs packed.
o-proj NT=4 is 103 (jm), a
loss vs 62 and 47. Stop NT=4.
packed qkv NT=4 is 106 (jo), a
loss vs 74. Stop NT=4.
A=s4 packed qkv map closed
(16.6 / 63 / 95) at 2800 both.
A=s4 o-proj is 19 both.
A=s2 packed qkv M=1 is 12 both.
A=s2 o-proj is 14 both.
Not W8A8-contract beats.
Seq 298 stays the T=256 mixer
leftover. o-proj leftover
CLOSED: NT=1 split-K=2 is
44-class both at 2800, beats
W8A8 47. Packed qkv M=256
W8A8 164 still open (SK=5
393 lost, SK=2 295 lost).
Packed M=64 split-K 115
stands vs 140. Decode packed
s8 74 stands. P2 decode XCCL
AR 99-137; all_gather 2.5 MiB
HANG timeout=45s bounded,
teardown HEALTHY. Host-staged
AR 439/9494 no hang. P3 T=1
77 us. P4 blocked.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.


## 10-hour remaining (ruthless)

One question per fire. Split cards.
P2 bulk hang is a new leftover.

1. new s8 packed-qkv M=256
   kernel vs W8A8 164 (4x8,
   4-acc 274, wg 8x4 279,
   persist 344, split-K 295
   all lost). M=64 split-K
   115 stands vs 140.
2. new s8 o-proj kernel vs
   W8A8 47 (sc 62, NT=4 103,
   NT=1 55, wg 4x2 74, B-pipe
   67, NT=1 B-pipe 60 all
   lost; NT=1 55 is the hand
   floor).
3. P2 XCCL all_gather >=2.5 MiB
   P2P-off with a timeout.
   Host-staged AR works (439 /
   9494) but loses to XCCL
   decode 99-137.
Park: split q/v vs packed 74
(q+k+v ~109), NT=4 s8, conv-L2
per-t SLM (358 vs 327),
conv-L2r (531 vs 327), packed
qkv s8 M=64 4x8 (214 vs 140)
and wg 8x4 (154 vs 140), packed
qkv s8 M=256 4-acc (274 vs 164)
and wg 8x4 (279 throttle=1 vs
164) and persist B-pipe (344),
o-proj s8 sc (62 vs 47), o-proj
B-pipe (67 vs 62), o-proj NT=1
wg 4x2 (74 vs 55), P2 256h
all_gather hang, P4, GRF256
retry (still zebin 128), mixer
T=256 packed (1557 vs seq 298),
skip-hi T=256 (330 vs slmht 260),
C=16/C=64
WY, rb=8, SLM-K+rb=4, blk>32,
a/b SLM, v-prefetch, SLM-K T=1,
tree hsum T=1, tile-fused T=1 reduce,
T=1 scalar hsum, slmht blk=8,
slmht 2-row, slmht blk=32 T=64,
slmht blk=32 T=128,
inner unroll, slmht unroll, pack a/b/v,
packed-o, SLM f32, SLM db, slmht SLM-db,
SLM LUT / u4+sign / skip-hi
kernel, persist-s8 GEMM us
(decode B-pipe and 4-acc B-pipe
already lost).

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
