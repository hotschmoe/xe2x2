# K1 incumbent dump 2026-09-02l

Backend `pytorch-xpu` on `sycl+l0`. Rank us. Synthetic tensors.
`int8_gemm_w8a16` is absent from stock vllm-xpu-env:int8g-v0251,
sglang int8 runtime mtp6, and the w8a8 vLLM image. It **is**
present in `b70-sglang-xpu-int8-w8a16:20260828-2dd55f3`. That
build's IGC dump is OpenCL `ref_matmul` (scalar mad, no dpas).
Times are ~2 ms at M=1 5120 -- a reference path, not the XMX
floor. Incumbent INT8 XMX here is `int8_gemm_w8a8` GEMM-only.

## fp8_gemm_w8a16

Image: b70-local/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50-s01
torch 2.13.0+xpu. Per-tensor scale, float8_e4m3fn weights, bf16 act.

| M x N x K | card0 us | card1 us |
|---|---:|---:|
| 1 x 5120 x 5120 | 57.5 | 56.2 |
| 1 x 17408 x 5120 | 248.6 | 262.0 |
| 64 x 5120 x 5120 | 92.4 | 94.0 |
| 64 x 17408 x 5120 | 339.2 | 339.4 |
| 256 x 5120 x 5120 | 318.0 | 227.0 |

M=256 disagrees (clock). M=1 and M=64 5120 match.

IGC_ShaderDumpEnable dumped `gemm_zero_fill` only. The GEMM is
oneDNN ngen (`ONEDNN_JIT_DUMP`): `dpas.8x8` **bf16 x bf16**, f32
acc. f8e4m3 is a shift/scale decompress then bf16 DPAS. No
`dpas.s8`. See `fp8_card0_isa.md` and `igc_card0_fp8_jit/`.

## int8_gemm_w8a8 (GEMM only)

Image: b70-sglang-xpu-int8-runtime:20260826-mtp6. Signature
(A s8, A_scale, B s8, B_scale, out_dtype, bias).

| M x N x K | card0 us | card1 us |
|---|---:|---:|
| 1 x 5120 x 5120 | 45.0 | 45.5 |
| 1 x 17408 x 5120 | 164.8 | 163.8 |
| 64 x 5120 x 5120 | 60.8 | 61.7 |
| 64 x 17408 x 5120 | 216.9 | 208.5 |
| 256 x 5120 x 5120 | 114.6 | 94.4 |

M=1 5120: INT8 W8A8 GEMM 45 us vs FP8 W8A16 56-58 us.
K0 naive s8 GEMV at the same shape was 990 us.

ngen ISA 2026-09-02z (both cards, identical bins): M=1 is
`dpas.8x4` wg 8x2 k64 (64 DPAS, some SLM). M=64 is `dpas.8x8`
wg 4x2x4 **grf256** k64 + 53 SLM. M=256 is 384x `dpas.8x8`
grf256 k128, no SLM. Native s8 `:b`, not s4. See
`int8a8_ngen_isa.md`.

## int8_gemm_w8a16 (ref_matmul, not XMX)

Image: b70-sglang-xpu-int8-w8a16:20260828-2dd55f3. Both cards.

| M x N x K | card0 us | card1 us |
|---|---:|---:|
| 1 x 5120 x 5120 | 2027 | 2027 |
| 1 x 17408 x 5120 | 3041 | 3033 |
| 64 x 5120 x 5120 | 12226 | 12180 |
| 256 x 5120 x 5120 | 56104 | 56117 |

ISA: `results/k1/int8_card1_isa.md`. No dpas.

Prior that FP8 W8A16 is the faster *kernel* loses at these shapes
when W8A8 skips activation quant. Serving W8A8 still paid ~160
quant launches; that is K5, not this row. Do not treat the
ref_matmul `int8_gemm_w8a16` as the INT8 floor.
