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

## lsc_prefetch_2d on 64 dpas (2026-09-02ae)

Same 64x dpas.8x4 as u64 plus next-k64 `lsc_prefetch_2d`.
ocloc: null-dest `load_block2d.ugm.d8 rd:0` (ngen ff).
grf_count 128. max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | u64 NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 208 | 83 | 123 | 335 | 53-69 |
| 64 x 5120 | 677 | 521 | 229 | 530 | 436-570 |
| 256 x 5120 | 688 | 952 | 857 | 602 | 594-1198 |

Prefetch-before-load is slower at decode than no-pf u64.
Floor stays 45 us.

## prefetch overlapped with dpas (2026-09-02af)

Same 64x dpas.8x4 as u64. Prologue ff of k=0, then load,
first dpas, then next-k64 `lsc_prefetch_2d`. ocloc: 64x
`dpas.8x4` plus null-dest `load_block2d.ugm.d8 rd:0`.
ff count 34 (NT=2) / 18 (NT=4) vs pf 131/99. GRF 128.
max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | u64 NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 264 | 100 | 110 | 371 | 53-69 |
| 64 x 5120 | 970 | 608 | 350 | 582 | 436-570 |
| 256 x 5120 | 830 | 1010 | 995 | 513 | 594-1198 |

NT=4 c1 started D3hot/2800 (slow M=4). Warm NT=2 is 100 us,
still above u64 53-69 and W8A8 42-46. Floor stays 45 us.

## A double-buffer software pipeline (2026-09-02ag)

Same 64x dpas.8x4 as u64. Prologue A[k=0], issue A[k+64]
before current dpas. No ff. ocloc: 64x `dpas.8x4`, GRF 128,
ff=0, A d8 34/18 + B d8v 64. max_abs=0. Warm 2600-2800 MHz.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | u64 NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 81 | 79 | 92 | 98 | 53-69 |
| 64 x 5120 | 632 | 650 | 271 | 316 | 436-570 |
| 256 x 5120 | 1084 | 1017 | 965 | 1100 | 594-1198 |

Warm NT=2 is 79-81 us, still above u64 53-69 and W8A8 42-46.
Floor stays 45 us.

## ngen wg 8x2 2D launch (2026-09-02ah)

Same 64x dpas.8x4 as u64. nd_range<2> local {8,2}=(N,M),
no SLM. ocloc: 64x `dpas.8x4`, GRF 128, no barrier. max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | u64 NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 225 | 69 | 122 | 316 | 53-69 |
| 64 x 5120 | 963 | 646 | 468 | 626 | 436-570 |
| 256 x 5120 | 929 | 999 | 1161 | 891 | 594-1198 |

NT=2 c0 and NT=4 c1 started D3hot (slow M=4). Warm NT=2 is
69 us, ties the slow end of 1D u64, not a 45 us beat.
Floor stays 45 us.

## wg 8x2 along N (2026-09-02ai)

Same 64x dpas.8x4 as u64. local {8,2} both on N, M is
group(1). ocloc: 64x `dpas.8x4`, GRF 128, no barrier.
max_abs=0. Warm ~2800 MHz.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | u64 NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 47 | 50 | 73 | 73 | 53-69 |
| 64 x 5120 | 570 | 611 | 304 | 317 | 436-570 |
| 256 x 5120 | 1285 | 1225 | 1184 | 1193 | 594-1198 |

NT=2 M=4 47-50 us both cards beats 1D u64 53-69. Not a
45 us W8A8 M=1 beat (pad M=4). New hand decode floor
~47-50 us.

## SLM A share plus 64 dpas 8x2-N (2026-09-02aj)

Same 8x2-along-N 64 dpas as wgn. lid0 A[k64] to SLM, two
barriers per k64. ocloc: 64x `dpas.8x4`, GRF 128, slm 1024.
max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | wgn NT=2 |
|---|---:|---:|---:|---:|---|
| 4 x 5120 | 101 | 140 | 114 | 111 | 47-50 |
| 64 x 5120 | 625 | 634 | 422 | 443 | 570-611 |
| 256 x 5120 | 899 | 933 | 1187 | 1184 | 1225-1285 |

NT=2 c1 started D3hot (140). Decode loses to no-SLM wgn.
Faster than old per-k32 SLM 372-463. Hand floor stays
47-50 us.

## M=1 pad to RC=4 (2026-09-02ak)

Same 8x2-along-N 64 dpas as wgn. M=1 zero-padded to 4
rows. ngen M=1 SLM is d32 store/load/fence, not A pack.
max_abs=0.

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | W8A8 |
|---|---:|---:|---:|---:|---|
| 1 x 5120 | 97 | 49 | 113 | 75 | 42-46 |
| 4 x 5120 | 131 | 81 | 177 | 119 | 42-46 (M=1) |
| 64 x 5120 | 1039 | 637 | 578 | 347 | 46-49 |

Warm card1 NT=2 M=1 is 49 us (D0/2800). card0 D3hot 97.
M=1 tracks M=4 in-run. Not a 45 us beat. Do not freeze 49.

## heat-then-decode NT=2 (2026-09-02al)

Heat dpas_s8 1024^3 80 iters, then dec NT=2. After heat
cur=1167, not 2800. Freq log ~717-750 MHz during hold.
max_abs=0.

| shape | card0 | card1 | W8A8 |
|---|---:|---:|---|
| heat 1024^3 | 141 us / 15.2 TOPS | 160 / 13.4 | n/a |
| 1 x 5120 | 61.5 | 65.4 | 42-46 |
| 4 x 5120 | 132 | 141 | 42-46 (M=1) |
| 64 x 5120 | 1106 | 1109 | 46-49 |

Do not quote 46-48 us from this fire. Hand floor stays
warm wgn M=4 47-50 us.

## clocks during the timed loop (2026-09-02am)

No 1024^3 heat. `dpas_s8_clk` is the wgn 8x2-N 64-dpas tile
with M pad to RC=4. Per-iter sysfs in the timed loop kept
duty cycle low (M=1 77-180 us at 467-1250 MHz). Batched
spin of the same kernel (4000 submits, wait every 256)
held act/cur=2800 at timed_begin/end. max_abs=0. NT=2.
40 timed iters. min_freq is root-only; not locked.

| shape | card | event_us | wait_host_us | pipe_host_us | timed cur |
|---|---|---:|---:|---:|---|
| 1 x 5120 | 0 | 35.958 | 50.443 | 36.443 | 2800 |
| 1 x 5120 | 1 | 35.797 | 49.987 | 36.431 | 2800 |
| 4 x 5120 | 0 | 36.146 | 51.623 | 36.697 | 2800 |
| 4 x 5120 | 1 | 36.039 | 50.201 | 36.654 | 2800 |

Event min-max at M=1 is 34.4-37.2. pipe_host matches
W8A8's pipelined us_bench. W8A8 K4 host was 42.1/46.1
and applies scales to out_dtype; this kernel stores s32.
Do not call a serving beat. Prior 47-50 us was not
held-2800. New hand decode floor at held 2800: ~36 us
event / ~36.4 us pipelined host (raw s32).

## ngen d32 flag broadcast (2026-09-02an)

Prologue lid0 `slm_block_store` d32 token=0, barrier,
all lanes load, token added into k0. ocloc: 64x
`dpas.8x4`, `store.slm.d32` + `fence.slm.none.group` +
`send.gtwy` barrier + `load.slm.d32`. GRF 128, slm_size
1024, barrier_count 1. max_abs=0. No 2800 spin (same
protocol as old wgn 47-50).

| shape | nt2 c0 | nt2 c1 | nt4 c0 | nt4 c1 | wgn NT=2 |
|---|---:|---:|---:|---:|---|
| 1 x 5120 | 90 | 194 | 91 | 104 | 47-50 (M=4) / 36 held |
| 4 x 5120 | 133 | 197 | 165 | 163 | 47-50 |
| 64 x 5120 | 1023 | 1143 | 552 | 576 | 570-611 |

Dummy d32 flag + barrier lights the ngen encoding and
loses decode. Hand floor stays the no-SLM 36 us at 2800.

## in-kernel GEMM repeats (2026-09-02ao)

One launch, R inner GEMMs, store last. us_per=event/R.
max_abs=0. Mid-run cur=2800.

| shape | R | card0 us_per | card1 us_per |
|---|---:|---:|---:|
| 1 x 5120 | 4096 | 34.460 | 34.318 |
| 4 x 5120 | 2048 | 34.512 | 34.380 |

Agrees with batched-spin 36 us event; ~1.5 us is launch.
Still raw s32.

## W8A8 scale epilogue to f16 (2026-09-02ap)

Same 8x2-N 64 dpas tile. acc * 0.02 * 0.02, store f16.
Fill [-64,64]. Spin=4000, 40 timed, NT=2. max_abs=0
cosine=1.0. timed act/cur=2800. ocloc store_block2d.d16.

| shape | card | event_us | pipe_host_us | W8A8 M=1 hold |
|---|---|---:|---:|---:|
| 1 x 5120 | 0 | 33.292 | 34.118 | 44.545 |
| 1 x 5120 | 1 | 33.386 | 34.341 | 43.817 |
| 4 x 5120 | 0 | 33.474 | 34.113 | n/a |
| 4 x 5120 | 1 | 33.451 | 34.392 | n/a |

W8A8 control is M=64 heat then M=1, same image, pipelined
host. K4 first-shape 42.1/46.1 still stands as the older
floor. Cold first-shape this session was 79/85.

## scale-to-f16 M=64 (2026-09-02aq)

Same binary, spin=4000, warmup 20 iters 20, NT=2.
cosine=1.0 max_abs=0. timed cur=2800, act 2683-2750,
throttle=1. W8A8 M=64 from same-day hold sweep.

| shape | card | event_us | pipe_host_us | W8A8 M=64 hold |
|---|---|---:|---:|---:|
| 64 x 5120 | 0 | 246.880 | 247.161 | 46.167 |
| 64 x 5120 | 1 | 244.833 | 242.533 | 46.450 |

~5.3x W8A8, ~7.4x this tile M=1 (not 16x). RC=4 wgn
is not the M=64 kernel. Next: ngen RC=8/GRF256/SLM.

## RC=8 dpas.8x8 f16 M=64 (2026-09-02ar)

Same f16 scale contract. 64x dpas.8x8, 8x2 along N.
grf_size<256> requested; zebin grf_count 128. spin=512.
cosine=1.0 max_abs=0. timed cur=2800 throttle=1.

| shape | card | event_us | pipe_host_us | RC=4 | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 120.729 | 119.626 | 246.880 | 46.167 |
| 64 x 5120 | 1 | 119.901 | 121.007 | 244.833 | 46.450 |

~2x RC=4 (half M-blocks). Still ~2.6x W8A8. GRF256
refused. Next: ngen wg / SLM pack.

## ngen 4x2x4 + SLM A pack M=64 (2026-09-02as)

32-thread wg 4x2x4, 4x RC=8, SLM A 4096, NT=2 U=4
(64 dpas). spin=512. cosine=1.0 max_abs=0. timed
act=cur=2800 throttle=0. IGA 64x dpas.8x8, 32
store.slm.d64x32t + 32 load, grf_count 128.

| shape | card | event_us | pipe_host_us | sc8 | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 135.172 | 136.102 | 119.626 | 46.167 |
| 64 x 5120 | 1 | 135.938 | 136.003 | 121.007 | 46.450 |

~13% slower than no-SLM sc8. SLM A-share is a tax.
Next: fuse K5 into M=1, not more barriers.

## A double-buffer on sc8 M=64 (2026-09-02av)

Same RC=8 8x2-N f16 contract. ska-style A ping-pong,
no SLM. spin=512. cosine=1.0 max_abs=0. timed
act=2783 cur=2800 throttle=1. IGA 64x dpas.8x8,
store_block2d.d16, grf_count 128, no slm_size.

| shape | card | event_us | pipe_host_us | sc8 | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 97.932 | 96.641 | 119.626 | 46.167 |
| 64 x 5120 | 1 | 103.396 | 100.435 | 121.007 | 46.450 |

~17-19% faster than no-db sc8. Still ~2.1x W8A8.
Next: 4-acc M-tile without SLM, or B pipeline.

## 4-acc M-tile no SLM M=64 (2026-09-02aw)

32 rows/thread, 4x dpas.8x8 acc, own A, A ping-pong,
NT=2 U=4 (64 dpas), 8x2 along N. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800 throttle=0.
IGA 64x dpas.8x8 {Atomic}, store_block2d.d16,
grf_count 128, no slm_size.

| shape | card | event_us | pipe_host_us | sc8db | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 118.990 | 119.833 | 96.641 | 46.167 |
| 64 x 5120 | 1 | 118.750 | 119.688 | 100.435 | 46.450 |

Matches sc8 (~120), loses to A-db (~98). Occupancy
of the 8-row tile beat extra B reuse. Next: B
pipeline + ca.ca on sc8db.

## B pipeline + ca.ca on sc8db M=64 (2026-09-02ax)

Same 8-row A-db tile plus B ping-pong and
lsc_load_2d L1/L2 cached. spin=512. cosine=1.0
max_abs=0. timed act=2783 cur=2800 throttle=1.
IGA load_block2d.ca.ca, 64-68x dpas.8x8, grf 128,
no SLM.

| shape | card | event_us | pipe_host_us | sc8db | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 106.156 | 105.455 | 96.641 | 46.167 |
| 64 x 5120 | 1 | 105.703 | 106.511 | 100.435 | 46.450 |

~6-9% slower than A-db only. Extra B GRF is a tax.
Keep sc8db. Next: null-dest prefetch on sc8db.

## Null-dest prefetch on sc8db M=64 (2026-09-02ay)

A-db plus lsc_prefetch_2d cached/cached of next k64
A and B. spin=512. cosine=1.0 max_abs=0. timed
act~2770 cur=2800 throttle=1. IGA 32x null
load_block2d.d8.ca.ca (rd:0), 64x dpas.8x8, grf 128.

| shape | card | event_us | pipe_host_us | sc8db | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 127.036 | 125.826 | 96.641 | 46.167 |
| 64 x 5120 | 1 | 127.448 | 128.072 | 100.435 | 46.450 |

~30% slower than A-db. Stop M=64 load-path chasing.
sc8db is the hand floor. Next: ngen M=256 k128 or
K5 producer without re-reading A.

## k128 A-db M=256 (2026-09-02az)

RC=8, k128 (4x k32), A ping-pong, NT=2 U=8 (64 dpas).
spin=512. cosine=1.0 max_abs=0. timed act=2550-2600
cur=2800 throttle=1. IGA 64x dpas.8x8, grf 128,
no SLM.

| shape | card | event_us | pipe_host_us | 4x M=64 | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 447.922 | 442.581 | ~392 | 76.1 |
| 256 x 5120 | 1 | 431.531 | 439.442 | ~392 | 74.9 |

~4.5x M=64 A-db, ~5.9x W8A8. k128 blocking is not
the 75 us kernel. Next: K5 producer without
re-reading A, or ngen wg 4x8.

## ngen wg 4x8 k128 M=256 (2026-09-02bb)

Same k128 A-db 8-row tile, wg 4 along N x 8 along M.
spin=512. cosine=1.0 max_abs=0. timed act=2733-2750
cur=2800 throttle=1. IGA 64x dpas.8x8, grf 128,
no SLM.

| shape | card | event_us | pipe_host_us | 8x2-N | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 228.526 | 229.544 | 442.581 | 76.1 |
| 256 x 5120 | 1 | 230.146 | 227.723 | 439.442 | 74.9 |

~1.9x vs 8x2-along-N. Still ~3x W8A8. Geometry beat
k128. Next: 4x8 on M=64 A-db, or 384 dpas unroll.

## wg 4x8 A-db M=64 (2026-09-02bc)

sc8db A-db tile, wg 4 along N x 8 along M. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800 throttle=0.
IGA 64x dpas.8x8, grf 128, no SLM.

| shape | card | event_us | pipe_host_us | 8x2-N A-db | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 74.000 | 75.486 | 96.641 | 46.167 |
| 64 x 5120 | 1 | 74.354 | 75.605 | 100.435 | 46.450 |

~1.3x vs 8x2-N. New M=64 hand floor 75 us, ~1.63x
W8A8. Next: 384 dpas unroll at M=256.

## 4-acc + wg 4x8 k128 M=256 (2026-09-02bd)

32 rows/thread, wg 4x8, k128, 256 dpas.8x8, no A-db.
spin=512. cosine=1.0 max_abs=0. timed act=2767
cur=2800 throttle=1. IGA 256x dpas.8x8 {Atomic},
grf 128, no SLM.

| shape | card | event_us | pipe_host_us | 8-row 4x8 | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 126.890 | 128.390 | 229.544 | 76.1 |
| 256 x 5120 | 1 | 127.891 | 128.568 | 227.723 | 74.9 |

~1.8x vs 8-row 4x8. New M=256 hand floor 128 us,
~1.7x W8A8. Next: 384 unroll or 4-acc on M=64 4x8.

## 384 dpas 6-acc wg 4x8 M=256 (2026-09-02be)

48 rows/thread, pad 288, wg 4x8, k128, 384 dpas.8x8,
no A-db. spin=512. cosine=1.0 max_abs=0. timed
act=cur=2800 throttle=0. IGA 384x dpas.8x8 (192
Atomic), grf 128, no SLM, spill 768 B NT=2.

| shape | card | event_us | pipe_host_us | 4-acc | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 209.484 | 210.060 | 128.390 | 76.1 |
| 256 x 5120 | 1 | 209.599 | 209.951 | 128.568 | 74.9 |

~1.64x slower than 4-acc. This arm is ~2.8x
W8A8. Floor stays 4-acc 128 us (~1.7x). 384
count is not the 75 us kernel. Next: A-db on
4-acc M=256, or 4-acc M=64 wg 4x2.

## k32 A-db on 4-acc M=256 (2026-09-02bf)

Same 4-acc wg 4x8 k128 256 dpas plus k32 A
ping-pong. spin=512. cosine=1.0 max_abs=0.
timed act=2783 cur=2800 throttle=1. IGA 256x
dpas.8x8 (192 Atomic), grf 128, no SLM, NT=2
no spill.

| shape | card | event_us | pipe_host_us | 4-acc | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 135.516 | 135.131 | 128.390 | 76.1 |
| 256 x 5120 | 1 | 135.859 | 134.881 | 128.568 | 74.9 |

~1.05x slower than 4-acc no A-db. Floor stays
128 us. A-db does not transfer from M=64.
Next: 4-acc on M=64 wg 4x2.

## 4-acc wg 4x2 M-on-Y M=64 (2026-09-02bg)

Same 4-acc k64 A-db 64 dpas as sc8m4, wg 4 along
N x 2 along M. spin=512. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0. IGA 64x dpas.8x8
(33 Atomic), grf 128, no SLM, spill 1792 B.

| shape | card | event_us | pipe_host_us | 8x2-N | 4x8 A-db | W8A8 |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 114.708 | 115.326 | 119.8 | 75.486 | 46.167 |
| 64 x 5120 | 1 | 115.115 | 115.965 | 119.7 | 75.605 | 46.450 |

~1.04x vs 8x2-N. Floor stays 75 us. Next: 4-acc
wg 4x2x4 no SLM (32 threads), or stop M=64 4-acc.

## 4-acc wg 4x2x4 no SLM M=64 (2026-09-02bh)

Same 4-acc k64 A-db 64 dpas, ngen wg 4x2x4 32
threads, no SLM. spin=512. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0. IGA 64x dpas.8x8
(33 Atomic), grf 128, no SLM, spill 1792 B.

| shape | card | event_us | pipe_host_us | 4x2 | SLM 4x2x4 | 4x8 A-db |
|---|---|---:|---:|---:|---:|---:|
| 64 x 5120 | 0 | 131.912 | 132.624 | 115.326 | 136 | 75.486 |
| 64 x 5120 | 1 | 131.818 | 132.944 | 115.965 | 136 | 75.605 |

~1.15x slower than 8-thread 4x2. Occupancy was
not the leftover. Stop M=64 4-acc. INT8 floor
75 us. Next: s4 on the M=64 4x8 A-db tile.

## s4 wg 4x8 A-db M=64 (2026-09-02bi)

Packed s4 A/B, k64 A-db, wg 4x8, 32 dpas.8x8
:s4/:s4, f16 scales 0.02, fill [-8,7]. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0. IGA 32x dpas.8x8 rW:s4 rA:s4, grf
128, no SLM.

| shape | card | event_us | pipe_host_us | s8 4x8 A-db | W8A8 |
|---|---|---:|---:|---:|---:|
| 64 x 5120 | 0 | 33.010 | 33.608 | 75.486 | 46.167 |
| 64 x 5120 | 1 | 33.042 | 33.735 | 75.605 | 46.450 |

~2.24x vs s8 75 us. Under W8A8 46 us in wall
time (different dtype). New s4 hand floor 33.6
us at 2800. Next: s4 M=256 4-acc and s4 M=1
decode, split per card.

## s4 4-acc wg 4x8 M=256 (2026-09-02bj/bm)

Packed s4, 4 M-tiles, wg 4x8, k128, 128
dpas.8x8 :s4/:s4, no A-db. spin=512. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.
IGA 128x dpas.8x8 rW:s4 rA:s4, grf 128, no SLM.
NT=2 no spill.

| shape | card | event_us | pipe_host_us | s8 4-acc | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 48.266 | 48.650 | 128.390 | 76.1 |
| 256 x 5120 | 1 | 48.724 | 48.471 | 128.568 | 74.9 |

Sibling matches (~0.4%). New s4 M=256 floor
48.6 us at 2800 both cards. ~2.63x s8 128,
under W8A8 75.

## s4 RC=4 8x2-N M=1 (2026-09-02bk/bl)

Packed s4, RC=4, wg 8x2 along N, 32 dpas.8x4
:s4/:s4, pad M to 4. spin=4000. cosine=1.0
max_abs=0. timed act=cur=2800 throttle=0.
IGA 32x dpas.8x4 rW:s4 rA:s4, grf 128, no SLM.

| shape | card | event_us | pipe_host_us | s8 RC=4 | W8A8 |
|---|---|---:|---:|---:|---:|
| 1 x 5120 | 0 | 16.013 | 16.411 | 34.12 | 44.55 |
| 1 x 5120 | 1 | 15.958 | 16.576 | 34.12 | 44.55 |
| 4 x 5120 | 0 | 15.982 | 16.345 | 34.7 | 44 |
| 4 x 5120 | 1 | 15.995 | 16.346 | 34.7 | 44 |

Sibling matches (~1%). New s4 decode floor
16.5 us at 2800 both cards. ~2.05x s8 34.
M=4 tracks M=1. Next: s4 A-db M=256 vs s4
decode N=17408.

## s4 k64 A-db on 4-acc M=256 card0 (2026-09-02bn)

Same 4-acc wg 4x8 k128 128 dpas plus k64 A
ping-pong. spin=512. cosine=1.0 max_abs=0.
timed act=cur=2800 throttle=0. IGA 128x
dpas.8x8 rW:s4 rA:s4, grf 128, no SLM, NT=2
no spill.

| shape | card | event_us | pipe_host_us | no A-db | W8A8 |
|---|---|---:|---:|---:|---:|
| 256 x 5120 | 0 | 50.740 | 51.937 | 48.650 | 76.1 |

~1.07x tax vs no A-db. Floor stays 48.6 us.

## s4 RC=4 N=17408 card1 (2026-09-02bo)

Same decode tile, N=17408 K=5120. spin=4000.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | N=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 1 x 17408 | 1 | 29.039 | 29.754 | 16.5 | 56 |
| 4 x 17408 | 1 | 29.443 | 29.739 | 16.3 | 56 |

~1.80x N=5120, not 3.40x. Sibling card0 pipe
29.235 (bp). New wide-N floor 29.5 us both
cards.

## s4 RC=4 K=17408 card1 (2026-09-02bq)

Same decode tile, N=5120 K=17408. spin=4000.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | K=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 1 x 5120 x 17408 | 1 | 52.922 | 53.367 | 16.5 | 56 |
| 4 x 5120 x 17408 | 1 | 52.984 | 53.303 | 16.3 | 56 |

~3.24x K=5120, near K-linear. Sibling card0
pipe 53.468 (br). New down-proj floor 53.4 us
both cards.

## s4 4x8 A-db M=64 N=17408 card1 (2026-09-02bs)

Same 4x8 A-db tile, N=17408 K=5120. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | N=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 64 x 17408 | 1 | 94.297 | 94.560 | 33.6 | 114 |

~2.81x N=5120. Sibling card0 pipe 94.805 (bt).
New M=64 wide-N floor 94.7 us both cards.

## s4 4x8 A-db M=64 K=17408 card1 (2026-09-02bu)

Same 4x8 A-db tile, N=5120 K=17408. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | K=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 64 x 5120 x 17408 | 1 | 104.958 | 105.843 | 33.6 | 114 |

~3.15x K=5120. Sibling card0 pipe 106.099 (bv).
New M=64 wide-K floor 106.0 us both cards.

## s4 4-acc M=256 N=17408 card1 (2026-09-02bw)

Same 4-acc wg 4x8 tile, N=17408 K=5120. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | N=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 256 x 17408 | 1 | 140.958 | 140.531 | 48.6 | 165 |

~2.88x N=5120. Sibling card0 pipe 139.436 (bx).
New M=256 wide-N floor 140.0 us both cards.

## s4 4-acc M=256 K=17408 card1 (2026-09-02by)

Same 4-acc wg 4x8 tile, N=5120 K=17408. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | K=5120 | napkin |
|---|---|---:|---:|---:|---:|
| 256 x 5120 x 17408 | 1 | 149.302 | 149.164 | 48.6 | 165 |

~3.07x K=5120. Sibling card0 pipe 148.768 (bz).
New M=256 wide-K floor 149.0 us both cards.
Qwen FFN s4 map closed.

## s8 4x8 A-db M=64 N=17408 card1 (2026-09-02ca)

Same s8 4x8 A-db tile, N=17408 K=5120. spin=512.
cosine=1.0 max_abs=0. timed act=cur=2800
throttle=0.

| shape | card | event_us | pipe_host_us | N=5120 | s4 | napkin |
|---|---|---:|---:|---:|---:|---:|
| 64 x 17408 | 1 | 337.609 | 338.151 | 75 | 94.7 | 255 |

~4.51x N=5120, worse than linear. s4 94.7 is
~3.57x this s8. One-card. Next: sibling vs
s8 M=64 K=17408.
