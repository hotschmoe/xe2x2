# QServe / QoQ -- arXiv 2405.04532 (W4A8, keep dequant on the systolic)

Status: LANDED (arxiv HTML v3, 2025-05-01).
NVIDIA/CUDA paper. Math only. Kernel names (vadd4, ldmatrix, INT8 tensor cores) do not run on Xe2.

## Source

- Title: QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
- Authors: Lin, Tang, Yang, Zhang, Xiao, Gan, Han (MIT / UMass / NVIDIA)
- arXiv: 2405.04532v3
- Fetched: https://arxiv.org/html/2405.04532
- Code they point at: https://github.com/mit-han-lab/omniserve

Foreign serving throughput stays in the paper.

## Precision choice (why W4A8KV4)

WxAyKVz = x-bit weight, y-bit activation, z-bit KV.

They argue W4A8 dominates both W4A16 (memory-bound, small m) and W8A8 (compute-bound, large m) on an A100 roofline if all MACs stay on INT8 tensor cores. KV4 is a bandwidth play on decode attention (intensity ~1 MAC/element, traffic is KV).

W4A4 is rejected on current Ampere/Hopper: per-group W4A4 needs INT32->FP dequant of partial sums in the sequential main loop on CUDA cores. One CUDA-core op ~ 50 INT4 tensor-core MACs on A100. Atom/QuaRot therefore lose to TRT W8A8 even with a higher theoretical INT4 roof.

Xe2 analogue is an experiment (campaign hail-mary 8): keep dequant in-register / on XMX, do not launch a separate dequant kernel.

## Progressive group quantization

Two-level, not "quant 4-bit then compress the scales" (that is VSQuant / QLoRA DoubleQuant / DGQ).

Level 0: per-channel symmetric INT8

```
W_hat = Qw_s8^(0) * s_fp16^(0)
```

Level 1: per-group asymmetric UINT4 on those INT8 codes

```
Qw_s8^(0) = (Qw_u4 - z_u4) * s_u8^(1)
```

At GEMM time: unpack UINT4 -> INT8 in registers using group scales/zeros, then run as W8A8 per-channel on INT8 tensor cores. All MACs stay INT8. Dequant is weight-side, not partial-sum-side (lower register pressure).

## Protective range [-119, 119]

Naive round-trip INT8 -> UINT4 -> INT8 can land outside [-128, 127]. Their example: group in [-113, 120], s=16, z=7, value 120 quantizes to 15, dequants to (15-7)*16 = 128, which is not a signed int8.

Saturation in the dequant ALU kills throughput (they quote a 67% hit; CUDA-specific).

Bound: with rounding,

```
q_hat_s8 = round(q_s8 / s_u8) * s_u8  <=  q_s8 + (1/2) s_u8
```

and

```
s_u8 = (q_s8_max - q_s8_min) / 15  <=  (127 - (-128)) / 15  = 17
```

so

```
q_hat_s8 <= 127   implies   q_s8 <= 127 - (1/2)*s_u8   implies   q_s8 <= 119.5
```

They shrink the INT8 symmetric range from [-127, 127] to a protective range [-119, 119] so the INT4 round-trip cannot overflow INT8. That is what lets them skip saturating arithmetic.

Versus DoubleQuant/VSQuant: those quantize to s4 first, then quantize the FP scales. Dequant of s4 with the level-1 scale does not produce an INT8 tensor, so compute falls back to float. DGQ forces scale restrictions to map to INT8 tensor cores but splits dequant into a separate kernel, losing the 4-bit bandwidth win.

## Subtract-after-multiply

### Per-channel W4A8 (no level-2 scales)

UINT4 -> UINT8 unpack, then UINT8 -> SINT8 zero-point subtract.

Naive: subtract zeros in the main loop (CUDA-core tax).

They rewrite:

```
O = (Qx odot Sx) * ((Qw - Zw) odot Sw)

  = (Qx * Qw) odot (sw x sx)  -  (Qx odot Sx) * (Zw odot Sw)
```

First term is ordinary W8A8, scales in the epilogue.

Second term: replace Qx odot Sx with unquantized X, and

```
X * (Zw odot Sw) = t_X x (zw odot sw)
t_X = X * 1_k     // per-token sum over K
```

which is another outer product and also lives in the epilogue. Zero-point subtract leaves the main loop. t_X is fused into the preceding memory-bound kernel.

### Per-group W4A8 (level-2 scales present)

Zeros are per-group, so they cannot move to the epilogue. There is an extra INT8 multiply by s_u8^(1) per weight.

Order still matters. NVIDIA has `vadd4` (four INT8 adds in one INT32 ALU) but no 4-way INT8 mul. 4-way mul is simulated by padding 24 zero bits onto the 8-bit scale, which is only valid if each product stays inside INT8.

Subtract-before-multiply overflows that trick. Subtract-after-multiply plus the protective range makes the first multiply stay in INT8, so both mul and sub can run 4-wide in registers.

That is the "register-level parallelism" / "keep dequant on-pipe" claim.

## SmoothAttention (KV4, not GEMM)

Keys have per-head outlier channels (~10x). Values do not. They scale keys down and queries up:

```
Z = (Q Lambda) (K Lambda^{-1})^T
lambda_i = max(|K_i|)^alpha          // alpha = 0.5 enough
```

Queries are not quantized, so there is no weight/activation migration search like SmoothQuant.

RoPE pairs channel i with i+D/2, so they constrain `lambda_i = lambda_{i+D/2}` and fuse Lambda into W_Q / W_K.

## Other QoQ pieces (algorithm, not Xe2 kernels)

- Hadamard rotate on block inputs (QKV, FFN1), absorb into previous weights.
- Smooth block outputs (O proj, FFN2) with alpha near 0 (weight-dominated), unlike SmoothQuant.
- Activation-aware channel reorder so similar-salience channels share a group (instead of Atom mixed-precision).
- Weight clip: minimize block output error, except q_proj/k_proj where they minimize block output MSE after the whole block.

## Xe2 analogue (experiment, not a port)

Keep W4 dequant in-register feeding `dpas<s8,s8>` (or native s4 if that is the real checkpoint). Protective range if a two-level integer path is used. Subtract-after-multiply if zero-points exist. Do not launch a standalone dequant kernel (that is the K5 lesson in another form). CUDA `vadd4` / tensor-core INT8 is not the Xe2 ISA; ESIMD DPAS + XVE integer ops are the mapping question.
