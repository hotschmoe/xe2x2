# K3 -- precision compose (INT2 / INT4 / INT8 and E2M1)

Question: can several lower-precision DPAS terms reconstruct a
higher-precision MAC, and is that ever faster or more accurate than
the native unit?

Open. See `docs/KERNEL_CAMPAIGN.md` for the schoolbook prior. The
prior says compose-of-s8 loses to native s8, and that E2M1 two-term
/ dyadic splits are the more interesting bet. Neither prior is a
result. Run the code. A compose that "should lose" and then wins is
the kind of FINDING this directory is for.

## Why

s8-from-s4 is four digit-MACs (three Karatsuba) plus shifts. s8-from-s2
is sixteen schoolbook terms. The unmeasured prior is that if s4 is
~2x s8 MAC rate, compose costs ~2x native s8. Measure it. K-depth
and epilogue cost are allowed to kill that prior.

E2M1 magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` are one dyadic or a
sum of two. After x2, codes `{0,1,2,3,4,6,8,12}`; 8 and 12 overflow
s4 [-8,7]. That overflow is a compose problem with a possible two-term
or sparse-correction kernel.

INT2 lighting has a home here even without a model: bitplanes, u2
magnitudes plus a sign, residue splits.

## Suggested arms

- Native s8 DPAS baseline (from K2, do not fork a third tile unless
  needed).
- Schoolbook s8-from-s4, Karatsuba s8-from-s4.
- Schoolbook / other s8-from-s2.
- Bitplane binary or s2 planes with shifts.
- E2M1 two-term `w_lo + 8*w_hi`.
- E2M1 dyadic planes `{0.5,1,2,4}`.
- Sparse correction only on overflowing codes.
- Numeric oracle on random tiles AND on real NVFP4 weight histograms
  if a checkpoint is at hand.

Card0 || card1: split compose families, swap.

## Record

Term count, extra bytes, mismatch vs fp32/fp16 oracle, us, TOPS, GB/s.
Say whether the arm is "emulate s8" or "emulate E2M1". Those are
different papers.

## Exit

A table of compose arms vs native s8 and vs s8 LUT-of-E2M1. FINDINGS
only for a compose that is either numerically closed or a clean
refusal (overflow, ISA, accuracy).

## First binaries (2026-09-02, compile only)

Standalone `icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31`. CPU docker
AOT (`compile_in_docker.sh`), no `--device`, no GPU run this pass.
icpx 2026.1.1. Device kernels name backend `sycl+l0`. CSV:
`arm,m,n,k,terms,us,TOPS,max_abs,ok`.

K2 prior used as CONFIG (NOT a RESULT): at 1024^3 matched ~583 MHz,
native s8 374 us, native s4 250 us (1.49x). Four schoolbook 4-bit
terms ~2.7x native s8; three Karatsuba terms ~2.0x IF the digit sums
fit s4. They do not.

| source | arm | terms | binary | AOT |
|--------|-----|------:|--------|-----|
| `compose_s8_from_s4.cpp` | native s8 + schoolbook u4_lo/s4_hi | 1 vs 4 | `bin/compose_s8_from_s4` | OK |
| `compose_s8_from_s4_karatsuba.cpp` | three s4 DPAS skipped | 3 algebra | `bin/compose_s8_from_s4_karatsuba` | OK (host) |
| `compose_e2m1_two_term.cpp` | s8 LUT vs w_lo+8*w_hi hail-mary | 1 vs 2 | `bin/compose_e2m1_two_term` | OK |

Schoolbook reuses the K2 tile (RC=8, exec N=16, transformed
`lsc_load_2d` B, `xmx::dpas`). Full s8 is `a = 16*a1 + a0` with
unsigned low nibble and signed high nibble, so the four DPAS types
are u4xu4, s4xu4, u4xs4, s4xs4, plus *16 and *256. A pure s4xs4
nibble split does not cover [-128,127].

Karatsuba three s4 DPAS is refused: `a0+a1` with those digits lives
in [-8,22], not a subset of s4 or u4. Recoding the sum adds terms.
The binary is a host algebra witness (identity closed on 65536
pairs; many sums overflow 4-bit). No device kernel.

E2M1: decode nibble to `q=2*w`, then split `q = w_lo + 8*w_hi`.
Never bitcast E2M1 onto s4. A is s4. CONFIG labels hail-mary.

Measured 2026-09-02o both cards (`results/k3/SUMMARY.md`). Schoolbook
did not lose to native s8 in-binary. E2M1 two-term closed and faster
than the s8 LUT. Karatsuba stays a host skip.
