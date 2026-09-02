# K4 -- W8 kernel-to-kernel

Question: at Qwen3.8-class shapes, is the well-optimized FP8 W8A16
op actually faster than INT8 W8A16 and INT8 W8A8 as *kernels*, or
did serving just skip activation quant?

Open. Call existing oneDNN ops first. Write a new kernel only if an
incumbent loses for a reason K2 can attack.

## Why

Steve's Qwen3.8 FP8 path is `_xpu_C.fp8_gemm_w8a16` (block-scaled FP8
weights, f16 activations, no activation quant). Xe2 has no native FP8
XMX; this is an emulated oneDNN recipe. INT8 W8A16 is native XMX with
the same launch shape. INT8 W8A8 adds per-token activation quant
(~160 launches in a dense 27B decode). Serving mixed those effects.
This directory unmixes them.

## Suggested arms

- A: `fp8_gemm_w8a16`
- B: `int8_gemm_w8a16`
- C: `int8_gemm_w8a8` GEMM-only
- Cq: C plus the quantizer as a second row
- Shapes: real TP=2 projections (hidden 5120, gate/up ~17408, qkv/o
  splits). Sweep M = 1, 2, 4, 64, 256, 1024.
- Scale granularity is in the arm name: block vs per-channel vs
  per-tensor. Do not compare block-FP8 to per-tensor INT8 and call
  it a datatype win.
- Cosine / max-abs vs f16 reference.

Card0 || card1: split the M sweep or the arm list, then swap so every
(arm, shape) has both cards.

## Record

us/iter, GB/s of weights, TOPS, cosine, IGC identity, whether quant
is included. Backend `pytorch-xpu` sitting on `sycl+l0` is fine if
that is how the op is reached; say so.

## Exit

A shape x arm table. FINDINGS if INT8 W8A16 beats or loses to FP8
W8A16 at M=1 and at large M with matched scales. Do not promote a
tok/s.
