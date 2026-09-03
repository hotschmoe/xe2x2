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
