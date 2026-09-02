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
move the us, not the ratio). In-register fused LUT (one
launch, 4-bit B into DPAS) is still open.
