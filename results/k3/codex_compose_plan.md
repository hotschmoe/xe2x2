# K3 precision-compose first measurement plan

Status: plan only. All cost projections below are CONFIG priors, not K3
results. The local K2 inputs are numeric-closed ESIMD measurements at
1024^3: s8 374 us, s4 250 us, and s2xs8 278 us. The rounded s4 speedup
is 374 / 250 = 1.49x. Standalone `icpx` is required because of
intel/llvm#21741.

## Term-count prior

For two s8 operands split into base-16 digits,

```
a = a0 + 16*a1
b = b0 + 16*b1
a*b = a0*b0 + 16*(a0*b1 + a1*b0) + 256*a1*b1
```

Schoolbook therefore needs four digit MACs. Nominal Karatsuba uses

```
p0 = a0*b0
p2 = a1*b1
ps = (a0 + a1)*(b0 + b1)
a*b = p0 + 16*(ps - p0 - p2) + 256*p2
```

and needs three digit MACs. That is an algebraic term count, not yet a
valid three-DPAS implementation: the digit sums can require five bits.

Using the measured 1.49x s4-over-s8 speed as a CONFIG prior:

| arm | nominal lower-precision terms | DPAS-only cost prior vs native s8 | rough time from K2 |
|---|---:|---:|---:|
| native s8 | 1 s8 | 1.00x | 374 us |
| s8-from-s4 schoolbook | 4 s4 | 4 / 1.49 = 2.68x | 4 * 250 = 1000 us |
| s8-from-s4 Karatsuba | 3 s4 | 3 / 1.49 = 2.01x | 3 * 250 = 750 us |
| E2M1 weight two-term | 2 s4 | 2 / 1.49 = 1.34x | 2 * 250 = 500 us |

These are lower bounds for a fused kernel, not predictions of measured
wall time. They omit extra plane loads, two or more accumulators,
shift/add work, register pressure, occupancy effects, and any change in
reuse. Conversely, one fused kernel can schedule independent DPAS chains
better than separate K2 launches, so multiplying K2 time is only a prior.

Four base-4 digits per s8 operand give 16 schoolbook s2xs2 digit
products. A recursive two-level Karatsuba count is nominally 9 base
products, but its sums also outgrow s2 and require recoding or more
terms. Do not assign either count a time from the 278 us s2xs8 result.
That measured arm is mixed s2xs8 with K=32 and fewer weight bytes; it is
not an s2xs2 digit-product rate. The K2 s2xs2 timings also were not
coherent enough across cards to use as a clean compose prior.

## Which target to measure first

Prioritize E2M1 `w_lo + 8*w_hi`, not arbitrary s8 reconstruction.
This is the better experimental bet, although the simple 1.49x prior
still puts two dense s4 terms at 1.34x the DPAS cost of one native s8
term. The reasons to test it first are narrower scope and fewer terms:

- It solves an actual E2M1 representation problem rather than rebuilding
  a type for which native s8 DPAS already exists.
- It needs two dense terms instead of four schoolbook terms. The nominal
  three-term Karatsuba path has a digit-width problem before it has a
  performance case.
- The high plane is nonzero only for the overflowing E2M1 magnitudes, so
  a later real-histogram arm may replace the dense second term with a
  sparse correction.
- It preserves s4 inputs and weights instead of materializing an s8 LUT,
  which may reduce input traffic even if raw DPAS issue count loses.

Let `q = 2*w`, including sign. The exact signed codes are zero and
`+/-{1,2,3,4,6,8,12}`. One canonical split is:

```
abs(q) <= 6: w_lo = q,        w_hi = 0
q = 8:       w_lo = 0,        w_hi = 1
q = 12:      w_lo = 4,        w_hi = 1
q = -8:      w_lo = 0,        w_hi = -1
q = -12:     w_lo = -4,       w_hi = -1
```

Both planes fit signed s4 and `q = w_lo + 8*w_hi` exactly. The integer
GEMM computes the result scaled by two; apply the common factor 1/2 and
any block scale after accumulation. The first binary should compare the
integer result before floating-point scaling so representation and DPAS
errors cannot be confused.

Sparse correction is a second experiment. First measure the dense
two-term kernel with every positive and negative E2M1 code represented.
Only after that closes numerically should a measured real weight
histogram determine whether `w_hi` is sparse enough to exploit.

## Packing and VNNI hazards

- Do not bitcast E2M1 nibbles as s4. E2M1 bit patterns are floating-point
  encodings, while DPAS s4 consumes signed two's-complement values in
  [-8, 7]. Decode to `q`, then construct and pack the two integer planes.
- An arbitrary signed s8 value is not generally two signed s4 digits.
  The ordinary nibble split has a signed high digit in [-8, 7] and an
  unsigned low digit in [0, 15]. A pure s4xs4 schoolbook kernel therefore
  does not cover full s8 without a mixed u4/s4 path or an explicit
  correction. Saturating or truncating the low digit would invalidate
  numeric closure.
- Karatsuba's `(a0+a1)` and `(b0+b1)` can exceed s4 even after a valid
  digit representation is chosen. The advertised three terms are only a
  lower bound until those sums are represented exactly. Count every
  correction DPAS as another term.
- Match K2's known-good s4 layout: pack two signed nibbles per byte along
  logical K, keep `Kc=64`, and feed B with
  `lsc_load_2d(..., Transformed=true)`. Use separate packed B surfaces
  for `w_lo` and `w_hi` and reuse the packed A fragment for both DPAS
  operations.
- Do not use flat host-prepacked VNNI as the only path. It was not
  bit-exact on BMG-G31, while the transformed 2D load was. Also do not
  pretransform data and then request a transformed load; that applies the
  layout operation twice.
- Keep logical K, packed-byte K, pitch, and 2D-load height distinct. For
  s4, two logical K values share a byte. Nibble order must match K2's low
  nibble for `k` and high nibble for `k+1`, including negative values
  masked with `0xf`.
- INT2 has different rules: four values share a byte, signed s2 is
  [-2, 1], and the Xe2 s2 weight operand uses VNNI16, meaning 16 int2
  values from K per dword. Do not import the paper's CPU VNNI4 layout.
- Native s8 uses K=32 per DPAS in the K2 kernel, while s4xs4 uses K=64.
  Compare the same logical M, N, and K, not the same number of K-loop
  iterations or DPAS calls.
- Keep two s32 accumulator tiles for low and high terms, then form
  `acc_lo + 8*acc_hi` in the epilogue. Use an int64 host oracle and check
  that the selected shape's exact result is representable in s32 before
  comparing. This catches shifted-intermediate overflow and endpoint
  errors rather than hiding them in s32 wraparound.

## Minimal first binary

Create one standalone translation unit, tentatively
`k3_e2m1_two_term.cpp`, by reducing the K2 s4 and s8 microkernels rather
than introducing a framework. Compile it directly with standalone
`icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31`; do not place this first
probe in a fat SYCL target because of intel/llvm#21741. Name the backend
`sycl+l0` in CONFIG output.

The binary needs exactly two timed arms over the same logical inputs:

1. `native_s8_lut`: signed s4 activations widened to s8 and signed E2M1
   `q` codes stored as s8, followed by one native s8xs8 DPAS chain.
2. `e2m1_s4_dense`: the activations packed as s4 and each `q` split into
   separately packed s4 `w_lo` and `w_hi` surfaces. Load A once per K
   chunk, run two s4xs4 DPAS chains, and store
   `acc_lo + 8*acc_hi` from the same kernel launch.

Both arms must use transformed 2D B loads and K2's RC=8, N=16 starting
tile. Use a small 8x16x64 check containing every signed E2M1 code and
activation endpoints -8 and 7, then the matched 1024x1024x1024 timed
shape. Packing and host setup stay outside the event-timed region for
both arms, but physical input bytes must be printed so the traffic
difference remains visible.

Emit one row per arm with backend, dtype/semantic target, term count,
M/N/K, tile, B-load path, physical A/B bytes, event time in us,
`max_abs` against the exact host oracle, and pass/fail. Do not derive a
verdict from a single arm or from unmatched clocks. The eventual record
should retain the repository form:

```
CONFIG -> exact backend, binary identity, shapes, layouts, clocks, terms
COMMAND -> standalone build and authorized wrapped execution
RESULT -> matched rows, numeric error, us, bytes, health and teardown
VERDICT -> E2M1 two-term wins, loses, or remains inconclusive
```

No K3 performance or stability conclusion exists until that measurement
is run with matched identity, health, clocks, and teardown evidence.
