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
