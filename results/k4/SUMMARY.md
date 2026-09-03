# K4 W8 kernel-to-kernel 2026-09-02q

Backend pytorch-xpu on sycl+l0. Cosine vs host reference.
GEMM-only: no activation-quant launches on the INT8 row.
int8_gemm_w8a16 skipped (ref_matmul, see K1).

Shapes: Qwen3.8-ish N,K. Per-tensor fp8 scale=1. INT8 per-token
act scale and per-N wei scale (synthetic 0.02).

## M x 5120 x 5120 us

| M | fp8 card0 | fp8 card1 | int8a8 card0 | int8a8 card1 |
|---:|---:|---:|---:|---:|
| 1 | 72.1 | 70.3 | 42.1 | 46.1 |
| 2 | 122.0 | 95.3 | 43.3 | 46.5 |
| 4 | 136.8 | 102.0 | 46.9 | 46.4 |
| 64 | 429.6 | 382.2 | 46.1 | 48.9 |
| 256 | 874.3 | 856.1 | 76.1 | 74.9 |
| 1024 | 1125.8 | 1110.8 | 255.5 | 229.8 |

INT8 cosine 1.000, max_abs ~0.06 (scale). FP8 cosine >=0.99997.

M=1 17408: fp8 355-465 us, int8a8 161 us both cards.
Held-2800 repeat both cards (cn): 158.1 us vs
hand s8 141.6. M=1 K=17408 both cards (cp):
155.3 us vs hand s8 261.6 (hand loses ~1.68x).
Qwen FFN oneDNN W8A8 decode map closed.

INT8 W8A8 GEMM-only is faster at every M here. Prefill gap is
large (M=64: ~48 vs ~400 us). Cq (plus quant launches) not in
this table -- K5.
