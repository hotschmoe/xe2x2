# K1 int8_gemm_w8a8 ngen ISA 2026-09-02z

Backend pytorch-xpu on sycl+l0. Image
b70-sglang-xpu-int8-runtime:20260826-mtp6. oneDNN 3.12.0
(80afa710). xe_hp_systolic skipped (jit_xe_hp_systolic.cpp:85).
Winner gpu,matmul,jit:gemm:any. Bins identical on card0 and card1
(md5). IGA Xe2 via libiga64 (CPU). Acc `:d`, operands `:b` (s8).

| shape | bin | catalog | dpas | slm | notes |
|---|---|---|---:|---:|---|
| M=1 5120 | .0/.1 27KB | wg 8x2 sys ska k64 sb256 wx4 | 64x `dpas.8x4` | 14 | RC=4, not 8 |
| M=64 5120 | .2 35KB | wg 4x2x4 kr grf256 k64 sb128 | 64x `dpas.8x8` | 53 | GRF 256 |
| M=256 5120 | .4 42KB | wg 4x8 grf256 k128 sb128 | 384x `dpas.8x8` | 0 | unrolled |

M=1 kernel:
`gemm OO[IH]S N@16N@16N 64 4 ... wg 8x2 wx4 ff sys af k64 ska`

M=64:
`wg 4x2x4 kr sys xaf k64 grf256`

M=256:
`wg 4x8 sys xaf k128 grf256`

No `dpas.s4`. Native s8 XMX. B loads include `load_block2d.ugm.d8`
and `d8v` (VNNI). Our hand tile always used RC=8 / GRF128 / no SLM.
Steal: RC=4 for decode, GRF256 + SLM pack for M=64.

Dump: `results/k1/igc_card0_int8a8_jit/` (card1 md5-identical).
