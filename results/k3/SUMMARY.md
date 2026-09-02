# K3 precision compose 2026-09-02o

Backend sycl+l0, standalone icpx 2026.1.1, same K2 tile.
CONFIG prior: schoolbook 4 s4 terms ~2.7x native s8 if s4 is 1.49x.
RESULT: that prior loses on this tile. Numeric max_abs=0.

Absolute us still tracks GT clock. Rank the *within-binary* ratio.

## emulate s8: schoolbook u4_lo + s4_hi vs native s8

| card | start MHz | native_s8 us | schoolbook_s4 us | ratio school/native |
|---|---:|---:|---:|---:|
| 0 | 2800 | 271.8 | 207.4 | 0.76 (compose faster) |
| 1 | 583 | 115.0 | 111.7 | 0.97 (tie) |

Check tiles 8x16x64 also max_abs=0.

## emulate E2M1: two-term w_lo+8*w_hi vs s8 LUT

Hail-mary, never bitcast. A is s4.

| card | start MHz | s8 LUT us | two-term us | ratio two/lut |
|---|---:|---:|---:|---:|
| 1 | 1500 | 373.8 | 221.6 | 0.59 |
| 0 | 700 | 93.7 | 83.2 | 0.89 |

max_abs=0 both cards.

## Karatsuba three s4 DPAS

Skipped. (a0+a1) in [-8,22] is not s4 or u4. Host identity on
65536 pairs max_abs=0; 45056 sums miss s4. No device kernel.
