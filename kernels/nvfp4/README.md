# K6 -- NVFP4 on Xe2 (spoof, split, LUT, resident 4-bit)

Question: every way a day-one NVIDIA NVFP4 checkpoint can ride B70
XMX or B70 HBM, including the ugly ones.

Open. NVFP4 models will keep showing up because exporters target
NVIDIA. This lab treats that as a format research object, not as
"wrong hardware." MXFP4 is a labeled third format. Integer GPTQ/AWQ
s4 is the true INT4 XMX control, not an NVFP4 alias.

## Why (sibling hypotheses to reproduce)

- E2M1 x2 is exact int8 `{0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}`. Scale
  absorbs the 2. Load-time s8 + K-group {16,1} was bit-exact after
  a square-group bugfix. 8B served; 27B s8 repack did not fit one
  30.3 GiB card.
- E2M1 is a float LUT, not two's-complement s4. +-12 overflows
  s4 [-8,7]. Bitcast to INT4 DPAS is wrong. oneDNN s4 is wrong for
  the same reason.
- `nvfp4_gemm_w4a16` keeps packed nibbles in VRAM and decompresses
  in the JIT. Decode-bandwidth play, not INT4 XMX.
- Built on 1024^3 (nibble_lut_s8 / _reg / _simd). Serving-shaped
  decode tile is `nibble_lut_sc`: LSC load packed nibbles -> simd
  register LUT to s8 -> VNNI4 -> `dpas<s8,s8>` on the K2 RC=4
  8x2-N scale-to-f16 tile. 4-bit HBM, INT8 XMX, E2M1 numerics.
  Never bitcast onto s4.
- True INT4 XMX wants integer s4 weights. ESIMD `dpas<s4,s4>` is
  the door; oneDNN will not open it.

## Arms to try (do all of them over the life of the lab)

1. Load-time s8 LUT + `int8_gemm_w8a16` (oracle and 8B-class path).
2. Resident packed E2M1 + oneDNN `nvfp4_gemm_w4a16` (current 27B
   class path).
3. On-the-fly nibble LUT into s8 DPAS (the unbuilt spoof).
4. Two-term s4 compose `w_lo + 8*w_hi` (overflow split). Cross-link
   K3.
5. Dyadic planes `{0.5,1,2,4}` / at-most-two scaled s2/s4 GEMMs.
6. Sparse correction only on codes 8 and 12.
7. Keep 4-bit in HBM, unpack to s8 in registers without a LUT if a
   closed-form map exists (prove it).
8. MXFP4 (e8m0 group 32) as a separate labeled arm.
9. Integer s4 checkpoint (GPTQ/AWQ/RTN) through ESIMD s4 DPAS, as
   the "this is what INT4 XMX is for" control.
10. Prefill vs decode: 4-bit resident should win M=1 GB/s; s8 XMX
    or s4 XMX should win large-M TOPS. Measure both.
11. Hail mary: 16-code E2M1 product LUT (256 exact products) as a
    decode GEMV and as the numeric oracle. Label hail-mary.

Card0 || card1: split the arm list, swap. Real checkpoints if present
under `/mnt/vm_8tb/github/b70_ai_things` models; otherwise synthetic
E2M1 tensors with a documented histogram.

## Record

VRAM bytes of weights, us, GB/s, TOPS, cosine vs E2M1 reference
dequant, whether the arm fits a 30.3 GiB card at 27B-class size
(can be a back-of-envelope plus one measured 8B or synthetic).

## Exit

A map: arm -> (fits 27B?, decode GB/s, prefill TOPS, numeric).
FINDINGS per arm that is either closed or cleanly refused. Bitcast
to s4 DPAS should be an explicit negative if it miscomputes.
