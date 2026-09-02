# NVFP4 vs OCP MX -- they are NOT the same format

Status: LANDED (NVIDIA blog + Transformer Engine docs + OCP MX companion paper 2310.10537 + spec snippets).
Official OCP PDF (ocp-microscaling-formats-mx-v1-0-spec-final-pdf) was not fully extracted as paginated text; tables below are from that spec as quoted by the companion paper, NVIDIA TE, and later papers that cite v1.0. Cross-check the PDF if a bit ever disagrees.

Not local evidence. Element codes are format math. Do not mix NVFP4 checkpoints into an MXFP4 arm.

## One-line difference

Both store E2M1 elements. The scale contract is different:

```
OCP MXFP4:  E2M1 element  +  E8M0 scale / 32 elements. No tensor scale.
NVFP4:      E2M1 element  +  E4M3 scale / 16 elements  +  FP32 tensor scale.
```

Same 4-bit payload encoding. Different grouping, different scale dtype, different reconstruction. Bitcast / oneDNN s4 / MXFP4 kernels are the wrong contract for an NVFP4 checkpoint.

## Signed E2M1 code table (shared by MXFP4 and NVFP4 elements)

Layout: 1 sign, 2 exponent, 1 mantissa. Exponent bias = 1.
No Inf, no NaN encodings. Subnormals supported. roundTiesToEven when converting to FP4.

Decode (IEEE-style with this bias):

```
if E > 0:  value = (1-2S) * 2^(E-1) * (1 + M/2)     // normal
if E = 0:  value = (1-2S) * 2^(1-1) * (M/2)          // subnormal; 0.5 or 0
```

Exact 16 codes:

```
bits  S E M   value
0000  0 00 0  +0.0
0001  0 00 1  +0.5     // min=max subnorm
0010  0 01 0  +1.0     // min normal
0011  0 01 1  +1.5
0100  0 10 0  +2.0
0101  0 10 1  +3.0
0110  0 11 0  +4.0
0111  0 11 1  +6.0     // max normal = 2^2 * 1.5
1000  1 00 0  -0.0
1001  1 00 1  -0.5
1010  1 01 0  -1.0
1011  1 01 1  -1.5
1100  1 10 0  -2.0
1101  1 10 1  -3.0
1110  1 11 0  -4.0
1111  1 11 1  -6.0
```

Finite magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.

This is a float LUT, not two's-complement s4. s4 is [-8, 7]. After x2 the integer codes

```
{0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}
```

and +-8, +-12 overflow s4. oneDNN s4 / ESIMD `dpas<s4,s4>` is the wrong type for these payloads.

Product of two signed E2M1 values is a closed 16x16 table (256 entries). Campaign hail-mary 2.

## OCP MX v1.0 (MXFP4)

Companion paper: arXiv 2310.10537, "Microscaling Data Formats for Deep Learning" (Rouhani et al., MX Alliance: Microsoft/AMD/Intel/Meta/NVIDIA/Qualcomm). Spec: OCP Microscaling (MX) Specification v1.0.

Block:

```
k = 32 elements
scale X : E8M0 (8-bit exponent only)
element P_i : FP4 E2M1 for MXFP4
value v_i = X * P_i     (if X is NaN, whole block is NaN)
```

Concrete MX family (all k=32, all E8M0 scale):

```
MXFP8   FP8 E5M2 or E4M3
MXFP6   FP6 E3M2 or E2M3
MXFP4   FP4 E2M1
MXINT8  INT8
```

E8M0:

- Unsigned biased IEEE binary32 exponent field.
- Representable scales: powers of two in [2^-127, 2^127].
- 0xFF reserved as NaN (invalid / uninitialized scale).
- No Inf in the scale. No mantissa: scale is strictly 2^n.

Example conversion (paper Algorithm 1, follows spec 6.3):

```
emax_elem = exponent of largest normal in the element format   // 2 for E2M1
shared_exp = floor(log2(max_i |V_i|)) - emax_elem
X = 2^shared_exp
P_i = quantize_to_element(V_i / X), clamp normals, keep Inf/NaN
```

Axis of the shared scale is a chosen principal axis (usually the reduction dim). Transpose and MX convert do not commute: W and W^T must be stored as two MX tensors if both are needed.

Storage: 32 * 4 + 8 = 136 bits / 32 values = 4.25 bits/value.

## NVFP4 (NVIDIA, Blackwell)

Sources:

- https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- Reconstruction math also in arXiv 2512.02010 (Four Over Six), citing NVIDIA 2025.

Reconstruction:

```
x = x_e2m1 * s_block * s_global
```

- x_e2m1: E2M1, magnitude in [0, 6]
- s_block: FP8 E4M3, shared by 16 consecutive elements
- s_global: FP32, one per tensor

Scale computation (TE):

```
s_global = global_amax / (fp8_max * fp4_max)
           fp8_max = 448.0   (E4M3)
           fp4_max = 6.0     (E2M1)

s_block  = (block_amax / fp4_max) / s_global
           stored as E4M3
```

E4M3 scale is fractional, not a power of two. That is the point versus E8M0.

Storage: 16 * 4 + 8 = 72 bits / 16 values = 4.5 bits/value, plus one FP32 tensor scale.

NVIDIA blog reconstructs a block as `x = x_q * s` with x_q in [-6, 6] and s = E4M3, then notes the second-level FP32. TE writes the three-factor form. Same format.

Weights may use 2D 16x16 blocks so rowwise and columnwise quant match. Activations/gradients stay 1D groups of 16.

NVFP4 GEMM on Blackwell is TN-only in TE. That is NVIDIA hardware, not Xe2.

## Side-by-side

```
                 MXFP4 (OCP)              NVFP4 (NVIDIA)
element          E2M1                     E2M1  (same 16 codes)
group            32                       16
block scale      E8M0 (power of two)      E4M3 (fractional FP8)
tensor scale     none                     FP32
bits/value       4.25                     ~4.5 + tensor FP32
range handling   2^n block scale          E4M3 * FP32
hardware (Nvidia) Blackwell MX            Blackwell NVFP4
Xe2 native       no                       no
```

Do not feed NVFP4 packs to an MXFP4 dequant (group 32, E8M0). Do not feed either to integer s4 DPAS.

## Compose notes for K3 / K6 (math only)

E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} are one dyadic or a sum of two. After x2, int codes {0,1,2,3,4,6,8,12}. Ideas already in KERNEL_CAMPAIGN.md: two-term `w = w_lo + 8*w_hi`, sparse correction on {8,12}, dyadic planes {0.5,1,2,4}, on-the-fly nibble LUT into s8 DPAS, 256-entry product LUT for M=1.

MXFP4 is a third labeled arm (e8m0 / 32). Never silently alias it to NVFP4.
