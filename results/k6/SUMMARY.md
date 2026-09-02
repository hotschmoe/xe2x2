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
closed, us lost. Vectorized in-register LUT still open.
