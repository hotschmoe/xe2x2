# K6 NVFP4 nibble LUT spoof 2026-09-02t

Backend sycl+l0, standalone icpx 2026.1.1 AOT intel_gpu_bmg_g31.
Never bitcast E2M1 onto s4. Packed E2M1 nibbles in HBM (2 per
byte along K). LUT nibble -> q=2*E2M1 s8. A is s8. Control:
host LUT then K2 s8 DPAS tile. Device: unpack kernel + same
DPAS (2 launches). Shape 1024^3. Numeric host s32 of A*q.

| arm | card0 us | card0 start MHz | card1 us | card1 start MHz | max_abs |
|---|---:|---:|---:|---:|---:|
| host_lut_s8_dpas | 270.5 | 550 | 74.7 | 583 | 0 |
| nibble_unpack | 33.1 | 550 | 9.2 | 583 | 0 |
| device_lut_then_dpas | 304.6 | 550 | 83.8 | 583 | 0 |

Check tile 8x16x32 also max_abs=0 both cards.
Unpack tax is ~12% of the s8 DPAS on both cards (clocks
move the us, not the ratio).

## In-register LUT (2026-09-02v)

One launch: packed load, GRF LUT, pack, s8 DPAS. Check-tile
tried pack_raw / pack_vnni4 / pack_kmajor. GT0 cur=2800.

| pack | check 8x16x32 max_abs | 1024^3 max_abs | 1024^3 us card0 | us card1 |
|---|---:|---:|---:|---:|
| raw | 6959 | 124224 | -- | -- |
| vnni4 | 0 | 0 | 2316 | 2317 |
| kmajor | 8240 | 107824 | -- | -- |

VNNI4 is the s8 B layout that matches Transformed LSC. This
scalar-unroll LUT is ~2316 us vs two-launch 84-305 us: layout
closed, us lost.

## Vectorized in-register LUT (2026-09-02x)

simd nibble decode + simd VNNI4 select. max_abs=0 both cards.

| card | start MHz | 1024^3 us | vs scalar 2316 |
|---|---:|---:|---|
| 0 | 633 | 304 | ~7.6x |
| 1 | 2800 | 406 | ~5.7x |

Now in the same us class as two-launch unpack+DPAS (84-305).
Do not freeze a winner without matched clocks. Keep two-launch
as the robust fast spoof.

## Serving-shaped decode tile (2026-09-02cq)

`nibble_lut_sc`: packed E2M1 B, simd LUT, VNNI4, K2
RC=4 8x2-N scale-to-f16. Never bitcast s4. NT=2
spin=4000. cosine=1.0 max_abs=0 both cards.
timed act=cur=2800 throttle=0.

| shape | card | event_us | pipe_host_us | s8 | W8A8 |
|---|---|---:|---:|---:|---:|
| 1 x 5120 | 0 | 157.760 | 158.172 | 34 | 44 |
| 1 x 5120 | 1 | 157.773 | 158.178 | 34 | 44 |
| 4 x 5120 | 0 | 157.768 | 158.304 | 34 | 44 |
| 4 x 5120 | 1 | 157.805 | 158.182 | 34 | 44 |

New serving-shaped NVFP4 LUT floor 158.2 us at
2800 both cards. ~4.65x s8 34. Packed-B 83 GB/s
(LUT tax, not HBM). M=4 tracks. Numeric closed.

## 16-entry iselect table (2026-09-02cr)

`nibble_lut_sct`: same tile, GRF table + iselect.
cosine=1.0 max_abs=0 both cards. timed
act=cur=2800 throttle=0.

| shape | card | pipe_host_us | merge LUT |
|---|---|---:|---:|
| 1 x 5120 | 0 | 1021.728 | 158 |
| 1 x 5120 | 1 | 1021.884 | 158 |

~6.46x merge LUT. Stop iselect tables. Keep merge.

## Two-launch scalar unpack (2026-09-03a)

`nibble_unpack_sc`: unpack then Transformed s8 GEMM.
cosine=1.0 max_abs=0 both cards. timed cur=2800
throttle=1.

| shape | card | pipe_host_us | s8ctrl | fused LUT |
|---|---|---:|---:|---:|
| 1 x 5120 | 0 | 266.098 | 34.546 | 158 |
| 1 x 5120 | 1 | 263.306 | 35.242 | 158 |

Naive unpack loses ~1.67x to fused LUT. Rank pipe.

## k64 packed load (2026-09-03b)

`nibble_lut_sck`: one height-32 packed load per k64.
cosine=1.0 max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | two-k32 LUT |
|---|---|---:|---:|
| 1 x 5120 | 0 | 169.017 | 158 |
| 1 x 5120 | 1 | 169.144 | 158 |

Small loss. Keep two k32 loads. Floor stays 158 us.

## Vectorized two-launch unpack card0 (2026-09-03c)

`nibble_unpack_scv`: ESIMD 16-wide unpack then
Transformed s8 GEMM. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | scalar unpack | fused LUT | s8ctrl |
|---|---|---:|---:|---:|---:|
| 1 x 5120 | 0 | 314.721 | 265 | 158 | 34.291 |
| 4 x 5120 | 0 | 314.387 | 265 | 158 | 33.968 |

Loss vs scalar (~1.19x) and fused LUT (~2.0x).
Sibling card1 pipe 314.444 (2026-09-03f).
Stop this unpack path. Keep fused 158 us.

## E2M1 two-term s4 decode (2026-09-03d/e)

`compose_e2m1_sc`: A s4, two s4 B planes,
acc_lo+8*acc_hi. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0. Never bitcast.

| shape | card | pipe_host_us | s4 | s8 | W8A8 | LUT |
|---|---|---:|---:|---:|---:|---:|
| 1 x 5120 | 0 | 28.520 | 16.5 | 34 | 44 | 158 |
| 1 x 5120 | 1 | 28.544 | 16.5 | 34 | 44 | 158 |
| 4 x 5120 | 0 | 28.685 | 16.5 | 34 | 44 | 158 |
| 4 x 5120 | 1 | 28.549 | 16.5 | 34 | 44 | 158 |

New floor 28.5 us both cards. ~1.73x native
s4, under s8 and W8A8. A is s4.

## E2M1 two-term N=17408 card0 (2026-09-03g)

Same tile, N=17408 K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 N | napkin |
|---|---|---:|---:|---:|---:|
| 1 x 17408 | 0 | 102.729 | 28.5 | 29.5 | 97 |
| 1 x 17408 | 1 | 104.353 | 28.5 | 29.5 | 97 |
| 4 x 17408 | 0 | 101.733 | 28.5 | 29.5 | 97 |
| 4 x 17408 | 1 | 102.142 | 28.5 | 29.5 | 97 |

New wide-N floor 103.5 us both cards.
~3.63x square, near linear, not s4's 1.80x.

## E2M1 two-term K=17408 (2026-09-03h/j)

Same tile, N=5120 K=17408. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 K | napkin |
|---|---|---:|---:|---:|---:|
| 1 x 5120 x 17408 | 0 | 194.021 | 28.5 | 53.4 | 97 |
| 1 x 5120 x 17408 | 1 | 193.096 | 28.5 | 53.4 | 97 |
| 4 x 5120 x 17408 | 0 | 195.476 | 28.5 | 53.4 | 97 |
| 4 x 5120 x 17408 | 1 | 193.146 | 28.5 | 53.4 | 97 |

New wide-K floor 193.6 us both cards.
~6.79x square, K-hostile. Qwen FFN compose
decode map closed.

## E2M1 two-term M=64 8x2-N card0 (2026-09-03k)

Same decode tile, M=64 N=K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | M=1 | s4 4x8 | s8 | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 217.915 | 28.5 | 33.6 | 75 | 46 |

~6.5x s4 4x8. Loss. One-card.

## E2M1 two-term M=256 8x2-N card1 (2026-09-03l)

Same tile, M=256 N=K=5120. cosine=1.0
max_abs=0. timed act=2683 cur=2800
throttle=1.

| shape | card | pipe_host_us | M=1 | s4 4-acc | s8 | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 256 x 5120 | 1 | 601.181 | 28.5 | 48.6 | 128 | 75 |

~12.4x s4 4-acc. Sibling card0 pipe 612.683
(2026-09-03m), throttle=1. Stop 8x2-N
compose at prefill.

## nibble LUT M=64 8x2-N (2026-09-03n/o)

`nibble_lut_sc` M=64 N=K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | M=1 | s8 4x8 | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 665.562 | 158 | 75 | 46 |
| 64 x 5120 | 1 | 645.630 | 158 | 75 | 46 |

~8.6x s8 4x8. Spread ~3.1%. Stop 8x2-N
LUT at prefill.

## E2M1 two-term 4x8 A-db M=64 (2026-09-03p/q)

`compose_e2m1_db48`: A s4, two s4 B planes,
acc_lo+8*acc_hi, RC=8 wg 4x8 A-db.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0. ocloc 64x dpas.8x8 rW:s4 rA:s4,
grf 128, no SLM. Never bitcast.

| shape | card | pipe_host_us | 8x2-N | s4 4x8 | s8 | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 68.732 | 217.9 | 33.6 | 75 | 46 |
| 64 x 5120 | 1 | 68.681 | 217.9 | 33.6 | 75 | 46 |

New floor 68.7 us both cards. ~2.04x native
s4, ~3.17x 8x2-N. Beats s8 75, loses W8A8 46.

## E2M1 two-term 4x8 A-db M=256 card0 (2026-09-03r)

Same tile, M=256 N=K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | M=64 | 8x2-N | s4 4-acc | s8 | W8A8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 x 5120 | 0 | 194.823 | 68.7 | 607 | 48.6 | 128 | 75 |
| 256 x 5120 | 1 | 195.034 | 68.7 | 607 | 48.6 | 128 | 75 |

~3.12x 8x2-N, ~4.0x s4 4-acc. Sibling
card1 pipe 195.034 (2026-09-03t). New
M=256 floor 194.9 us both cards.

## nibble LUT 4x8 A-db M=64 (2026-09-03s/u)

`nibble_lut_db48`: packed E2M1, simd LUT,
VNNI4, s8 DPAS, RC=8 wg 4x8 A-db.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0. ocloc 64x dpas.8x8 rW:b rA:b,
packed B not Transformed, grf 128, no SLM.
Never bitcast.

| shape | card | pipe_host_us | 8x2-N | s8 4x8 | W8A8 | compose |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 392.427 | 656 | 75 | 46 | 68.7 |
| 64 x 5120 | 1 | 392.443 | 656 | 75 | 46 | 68.7 |

New 4x8 LUT floor 392.4 us both cards.
~1.67x 8x2-N, still ~5.23x s8 75.

## E2M1 two-term 4x8 A-db M=64 N=17408 card0 (2026-09-03v)

Same tile, N=17408 K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 N | s8 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 17408 | 0 | 328.026 | 68.7 | 94.7 | 338.9 | 233 |
| 64 x 17408 | 1 | 325.801 | 68.7 | 94.7 | 338.9 | 233 |

New wide-N floor 326.9 us both cards.
~4.76x square vs s4 2.81x.

## E2M1 two-term 4x8 A-db M=64 K=17408 (2026-09-03w/y)

Same tile, N=5120 K=17408. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 K | s8 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 x 17408 | 0 | 403.596 | 68.7 | 106.0 | 374.7 | 233 |
| 64 x 5120 x 17408 | 1 | 403.192 | 68.7 | 106.0 | 374.7 | 233 |

New wide-K floor 403.4 us both cards.
~5.87x square vs s4 3.15x. Loses to s8
374.7. Qwen FFN compose M=64 map closed.

## E2M1 two-term 4x8 A-db M=256 N=17408 (2026-09-03z/ac)

Same tile, N=17408 K=5120. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 N | s8 | M=64 |
|---|---|---:|---:|---:|---:|---:|
| 256 x 17408 | 0 | 985.644 | 194.9 | 140.0 | 469.8 | 326.9 |
| 256 x 17408 | 1 | 982.879 | 194.9 | 140.0 | 469.8 | 326.9 |

New wide-N floor 984.3 us both cards.
~5.05x square vs s4 2.88x. ~2.10x s8.

## E2M1 two-term 4x8 A-db M=256 K=17408 (2026-09-03aa/ab)

Same tile, N=5120 K=17408. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s4 K | s8 | M=64 |
|---|---|---:|---:|---:|---:|---:|
| 256 x 5120 x 17408 | 0 | 964.294 | 194.9 | 149.0 | 477.4 | 403.4 |
| 256 x 5120 x 17408 | 1 | 973.110 | 194.9 | 149.0 | 477.4 | 403.4 |

New wide-K floor 968.7 us both cards.
~4.97x square vs s4 3.07x. ~2.03x s8.
Qwen FFN compose M=256 map closed.

## Closed-form nibble LUT (2026-09-03ad)

`nibble_lut_scf`: exp/mant shift, no 16-entry
table. Same RC=4 8x2-N packed E2M1. Never
bitcast. cosine=1.0 max_abs=0. timed
act=cur=2800 throttle=0.

| shape | card | pipe_host_us | merge | s8 | W8A8 | GBs_packedB |
|---|---|---:|---:|---:|---:|---:|
| 1 x 5120 | 0 | 134.756 | 158 | 34 | 44 | 97.557 |
| 1 x 5120 | 1 | 134.783 | 158 | 34 | 44 | 97.542 |

New Family-A floor 134.8 us both cards.
~1.17x merge LUT. Still ~4.0x s8 34.

## E2M1 two-term 4-acc M=256 card0 (2026-09-03af)

`compose_e2m1_w48m4`: A s4, two s4 B planes,
RC=8 4-acc wg 4x8 k128. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.
ocloc 256x dpas.8x8 rW:s4 rA:s4, grf 128,
no SLM. Never bitcast.

| shape | card | pipe_host_us | 4x8 | s4 4-acc | s8 | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 256 x 5120 | 0 | 411.303 | 194.9 | 48.6 | 128 | 75 |

~8.46x native s4, ~2.11x 4x8 compose.
One-card. Do not freeze 411 us.

## nibble LUT 4x8 A-db M=64 N=17408 (2026-09-03ag/ah)

Same `nibble_lut_db48`, N=17408 K=5120.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | pipe_host_us | 5120 | s8 N | s4 | compose |
|---|---|---:|---:|---:|---:|---:|
| 64 x 17408 | 0 | 1027.424 | 392.4 | 338.9 | 94.7 | 326.9 |
| 64 x 17408 | 1 | 1037.007 | 392.4 | 338.9 | 94.7 | 326.9 |

New wide-N floor 1032 us both cards.
~2.63x square vs s4 2.81x. ~3.05x s8.

## nibble LUT 4x8 A-db M=64 K=17408 (2026-09-03ai/aj)

Same tile, N=5120 K=17408. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.

| shape | card | pipe_host_us | 5120 | s8 K | s4 | compose |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 x 17408 | 0 | 1332.410 | 392.4 | 374.7 | 106.0 | 403.4 |
| 64 x 5120 x 17408 | 1 | 1332.672 | 392.4 | 374.7 | 106.0 | 403.4 |

New wide-K floor 1333 us both cards.
K-linear ~3.40x. ~3.56x s8. Qwen FFN LUT
M=64 map closed.

## nibble LUT 4x8 A-db M=256 (2026-09-03ak/al)

Same `nibble_lut_db48`, M=256 N=K=5120.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | pipe_host_us | M=64 | s8 | compose | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 256 x 5120 | 0 | 1198.437 | 392.4 | 128 | 194.9 | 75 |
| 256 x 5120 | 1 | 1207.283 | 392.4 | 128 | 194.9 | 75 |

New M=256 floor 1203 us both cards.
~3.07x M=64. ~9.4x s8.

## closed-form LUT 4x8 A-db M=64 (2026-09-03am/an)

`nibble_lut_scf_db48`: exp/mant shift on
the 4x8 A-db tile. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0. ocloc 64x
dpas.8x8 rW:b rA:b, grf 128, no SLM.

| shape | card | pipe_host_us | merge | scf decode | s8 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 331.665 | 392.4 | 134.8 | 75 | 335 |
| 64 x 5120 | 1 | 331.554 | 392.4 | 134.8 | 75 | 335 |

New floor 331.6 us both cards. ~1.18x
merge. Napkin held. Still ~4.42x s8.

## closed-form LUT 4x8 A-db M=256 (2026-09-03ao/ap)

Same `nibble_lut_scf_db48`, M=256 N=K=5120.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | pipe_host_us | M=64 | merge | s8 | compose |
|---|---|---:|---:|---:|---:|---:|
| 256 x 5120 | 0 | 1077.148 | 331.6 | 1203 | 128 | 194.9 |
| 256 x 5120 | 1 | 1089.132 | 331.6 | 1203 | 128 | 194.9 |

New M=256 floor 1083 us both cards.
~3.27x M=64. ~1.11x merge 1203.

## closed-form LUT 4x8 A-db M=64 N=17408 (2026-09-03aq/ar)

Same `nibble_lut_scf_db48`, M=64 N=17408
K=5120. cosine=1.0 max_abs=0. timed
act=2767/2783 cur=2800 throttle=1.

| shape | card | pipe_host_us | square | merge | s8 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 17408 | 0 | 882.536 | 331.6 | 1032 | 338.9 | 872 |
| 64 x 17408 | 1 | 877.318 | 331.6 | 1032 | 338.9 | 872 |

New wide-N floor 880 us both cards.
~2.65x square. ~1.17x merge. Throttle=1.

## closed-form LUT 4x8 A-db M=64 K=17408 (2026-09-03as/at)

Same `nibble_lut_scf_db48`, M=64 N=5120
K=17408. cosine=1.0 max_abs=0. timed
act=cur=2800 throttle=0.

| shape | card | pipe_host_us | square | merge | s8 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 x 17408 | 0 | 1125.896 | 331.6 | 1333 | 374.7 | 1127 |
| 64 x 5120 x 17408 | 1 | 1123.392 | 331.6 | 1333 | 374.7 | 1127 |

New wide-K floor 1125 us both cards.
K-linear ~3.39x. ~1.18x merge. Qwen FFN
closed-form LUT M=64 map closed.

## closed-form LUT 4x8 A-db M=256 N=17408 card1 (2026-09-03au)

Same `nibble_lut_scf_db48`, M=256 N=17408
K=5120. cosine=1.0 max_abs=0. timed
act=2683 cur=2800 throttle=1.

| shape | card | pipe_host_us | square | s8 | compose | napkin |
|---|---|---:|---:|---:|---:|---:|
| 256 x 17408 | 1 | 3113.855 | 1083 | 469.8 | 984.3 | 2874 |

~2.88x square. ~6.63x s8. Napkin missed
~8.3%. Throttle=1. One-card. Do not
freeze 3114 us.

## 12-idea sprint (2026-09-03ae)

| idea | result |
|---|---|
| 1 sparse-hi / lo-only | hist ov 24.6-25.1%. loonly 16.34/16.35 us cosine 0.76 ok=0 |
| 2 dyadic s2 | s2xs2 COMPILE_OK max_abs=0. s2xs4 COMPILE_REFUSED. 4-plane not fused |
| 3 mixed dpas | MIX_OK s8xs4 and s4xs8 both cards. no s32 oracle |
| 4 product LUT GEMV | max_abs=0. 697/1106 us clocks unmatched. stop |
| 5 closed-form | 134.8 us both-card held 2800 |
| 6 nvfp4_gemm_w4a16 | v028 so. 36.8/37.2 us after M=64 heat. clocks not 2800 |
| 7 bitcast s4 | max_abs 352/1408 ok=0 both. explicit negative |
| 8 g16 e4m3 | hand K=16 COMPILE_REFUSED. f8scale op ~39 us |
| 9 checkpoint hist | 8 FFN tensors ov_frac 0.246-0.251. sparse-hi dead |
| 10 MXFP4 | 0 layers. NVFP4 193 g16. FP8 208 |
| 11 persist s8 | weights 29.0 GiB vs resident 20.4 GiB |
| 12 ISA toys | skip-hi=1. s8xs4 lights. s2xs4 refuse. SLM/u4 not built |
