# K2 ESIMD DPAS 2026-09-02k

Standalone icpx 2026.1.1 AOT intel_gpu_bmg_g31, backend sycl+l0,
B feed lsc_load_2d Transformed=true. Numeric host s32 oracle.
Shape timed 1024^3 unless noted. GT0 cur_freq often 583 MHz on
these short kernels; act_freq sysfs reads 0. TOPS is diagnostic
and clock-contaminated. Rank us.

## Compile

All four standalone binaries compiled (CPU docker + host oneAPI).
No API refusal for s8, s4, s2xs8, s2xs2. Fat-binary metadata
`has_dpas: true` on every arm. LLVM intrinsic names (not ISA
mnemonics; s8 and s4 share a type string, precision is likely an
immediate):

- s8xs8 / s4xs4: `llvm.genx.dpas2.v128i32.v128i32.v128i32.v64i32`
- s2xs8: `llvm.genx.dpas2.v128i32.v128i32.v32i32.v64i32`
- s2xs2: `llvm.genx.dpas2.v128i32.v128i32.v64i32.v32i32`

Need `ocloc` / IGC shader dump for `dpas.s4.s4` vs `dpas.s8.s8`.

## Numeric + us at 1024^3

| arm | card0 us | card0 TOPS | card0 max_abs | card1 us | card1 TOPS | card1 max_abs |
|---|---:|---:|---:|---:|---:|---:|
| s8xs8 | 373.7 | 5.75 | 0 | 374.1 | 5.74 | 0 |
| s4xs4 | 250.3 | 8.58 | 0 | 250.6 | 8.57 | 0 |
| s2xs8 | 277.7 | 7.73 | 0 | 277.8 | 7.73 | 0 |
| s2xs2 | 49.6 / 89.2 | 43.3 / 24.1 | 0 | 223.1 / 222.9 | 9.62 / 9.63 | 0 |

s8/s4/s2xs8 match across cards (both at cur~583 MHz). s2xs2 does
*not*: card0 was at 1950 MHz on the repeat (89 us / 24 TOPS) and
card1 stayed at 583 MHz (223 us / 9.6 TOPS). Same binary, numeric
closed on both. Do not quote a single s2 TOPS.

s4/s8 us ratio at matched clocks: 374/250 = 1.49x, not the sibling
~2x prior.

s2xs8 vs s8 at matched clocks: 278 vs 374 us (1.35x). Paper prior
was same systolic rate as s8; wall time is faster (fewer B bytes).

K0 XVE s8 at 1024^3 was ~2200 us / 0.95 TOPS. DPAS s8 is ~6x in us.

Check tiles (8x16xK) also max_abs=0 for every arm.

## Clocks

Short kernels (50-400 us) do not hold 2800 MHz. cur_freq after
many runs sits at 583 MHz. TOPS% of 367 is not a kernel verdict
until a long occupancy run records clocks during the timed loop.

## Clock-hold repeat (20 warmup + 80 iters, 2026-09-02m)

Same 1024^3, numeric still max_abs=0. us follows the starting
cur_freq, not a new algorithm:

| arm | card | start MHz | timed us | TOPS | end MHz |
|---|---|---:|---:|---:|---:|
| s8 | 0 | 867 | 257.4 | 8.34 | 2800 |
| s8 | 1 | 867 | 110.0 | 19.53 | 1500 |
| s4 | 1 | 1883 | 130.8 | 16.42 | 867 |
| s4 | 0 | 2800 | 69.1 | 31.07 | 1600 |

At a 2800 MHz start, s4 is 69 us / 31 TOPS (still ~8% of 367).
The 1.49x s4/s8 ratio was a 583 MHz pair; do not freeze it.
