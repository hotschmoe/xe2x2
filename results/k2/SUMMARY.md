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

ocloc disasm of the AOT zebin (2026-09-02r): IGA prints s8 as
`:b`, s4 as `:s4`, s2 as `:s2`. Inner loop is always `dpas.8x8
(16|M0)` acc `:d`. s2xs8 is `rW:s2 rA:b`. Runtime
IGC_ShaderDumpEnable did not dump the user kernel (AOT). See
`results/k2/igc_isa.md`.

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

## Serving shapes vs oneDNN W8A8 (2026-09-02w)

Same 8x16 tile. M=8 is padded decode (RC=8). Numeric max_abs=0.
oneDNN W8A8 GEMM-only: M=1 42-46 us, M=64 46-49, M=256 74-76.

| shape | s8 c0 us | s8 c1 us | s4 c0 us | s4 c1 us |
|---|---:|---:|---:|---:|
| 8 x 5120 x 5120 | 230 | 56 | 34 / 64 rpt | 120 / 87 rpt |
| 8 x 17408 x 5120 | 524 | 286 | 127 | 216 |
| 64 x 5120 x 5120 | 274 | 373 | 387 | 393 |
| 256 x 5120 x 5120 | 892 | 1064 | 691 | 691 |

This tile does not beat 45 us W8A8. Clocks swing padded M=8.
M=64/256 lose by ~6-12x. Need a real schedule, not 8x16.

## Blocked NT=2/4 A-reuse (2026-09-02y)

Same s8 DPAS, A loaded once per K-chunk and reused across NT
N-tiles. max_abs=0. oneDNN W8A8 M=64/256: 46-49 / 74-76 us.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | 8x16 s8 |
|---|---:|---:|---:|---:|---|
| 8 x 5120 | 233 | 102 | 120 | 315 | 56-230 |
| 64 x 5120 | 269 | 352 | 318 | 474 | 274-373 |
| 256 x 5120 | 856 | 943 | 694 | 876 | 892-1064 |

A-reuse is closed and is not a 6x win. NT=4 can help M=256
slightly (694 vs 892) and can lose at M=64. Floor stays 45 us.

## RC=4 decode tile (2026-09-02aa)

Stolen from W8A8 ngen M=1 (`dpas.8x4`). ocloc: `dpas.8x4 (16|M0)
r22:d r22:d r14:b r11.0:b`. ze_info grf_count 128. max_abs=0.

| shape | card0 us | card1 us | W8A8 |
|---|---:|---:|---|
| 4 x 5120 (pad M=1) | 168 | 77 | 42-46 (M=1) |
| 64 x 5120 | 894 | 614 | 46-49 |
| 256 x 5120 | 1028 | 1069 | 74-76 |

RC=4 lights and is not a 45 us beat. Prefill is worse than RC=8.

## GRF256 request (2026-09-02aa)

`intelex::grf_size<256>` compiled. `-ftarget-register-alloc-mode=pvc:large`
on dpas_s8 also compiled. Both AOT zebins still `grf_count: 128`.
Not a 256-GRF RESULT. Do not quote those us as a GRF A/B.

## SLM A share WG=16 RC=4 (2026-09-02ab)

oneDNN M=1 is wg 8x2 + SLM. This arm: 16 threads share one A
4x32 in SLM, each does N=16 (WG N=256). max_abs=0. Barrier
per K-chunk.

| shape | card0 us | card1 us | RC=4 no SLM |
|---|---:|---:|---|
| 4 x 5120 | 463 | 372 | 77-168 |
| 64 x 5120 | 525 | 1083 | 614-894 |
| 256 x 5120 | 1373 | 1296 | 1028-1069 |

SLM A + barrier is slower than per-thread A at decode. Floor
stays 45 us.

## k64 NT=2/4 RC=4 (2026-09-02ac)

Stolen k64 blocking from W8A8 ngen M=1. Inner step: two
K=32 DPAS, A reused across NT, no SLM. ocloc: `dpas.8x4
(16|M0) rW:b rA:b`. ze_info grf_count 128. NT=2 has 4 static
dpas; NT=4 has 16. Not ngen's 64. max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | W8A8 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 278 | 92 | 111 | 396 | 42-46 (M=1) |
| 64 x 5120 | 728 | 482 | 293 | 529 | 46-49 |
| 256 x 5120 | 616 | 810 | 962 | 599 | 74-76 |

First pair (c0 nt2, c1 nt4) started D3hot cur=2800. Swap
started ~1620 MHz warm. M=4 92-396 is clock + first-touch,
not a 45 us beat. Floor stays 45 us.

## 64 static dpas.8x4 unroll (2026-09-02ad)

k64 steps unrolled so 2*NT*UNROLL=64. NT=4 U=8 innerK=512;
NT=2 U=16 innerK=1024. ocloc: 64x `dpas.8x4` both kernels,
grf_count 128. max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | W8A8 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 69 | 53 | 316 | 107 | 42-46 (M=1) |
| 64 x 5120 | 570 | 436 | 487 | 314 | 46-49 |
| 256 x 5120 | 1198 | 1186 | 594 | 1044 | 74-76 |

NT=4 c0 started D3hot/2800 (slow M=4). Warm NT=2 is 53-69
us at M=4, closer than k64, not a 45 us beat. Floor stays
45 us.
