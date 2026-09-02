# K1 IGC reading -- int8_gemm_w8a16 card1

Backend pytorch-xpu on sycl+l0. Card 1, ZE_AFFINITY_MASK=1.
Image that HAS the op: b70-sglang-xpu-int8-w8a16:20260828-2dd55f3
Id=sha256:91cc53fab0e683a27735667ff0802ee065d328ad66f1d0d2d6c7236d0e1475f3
torch 2.13.0+xpu. IGC dump: results/k1/igc_card1_int8/ (36 files).
Device 0x4fa.0xe223, IGC asm git-hash 53a9734e8b444ef867a1f7f580fbbcee2acc32ee.

Misses (no dump): vllm-xpu-env:int8g-v0251 HAS int8_gemm_w8a16=False;
b70-sglang-xpu-int8-runtime:20260826-mtp6 HAS=False (has int8_gemm_w8a8).

## ISA

Both compiled shaders are OpenCL `ref_matmul` (zeinfo), simd32, 128 GRF.
Compile macros: SRC_DT_F16, WEI_DT_S8 (char), DST_DT_F16, ACC_DT_F32,
WITH_WEI_SCALES=1, WEI_SCALES_MASK=2 (per-N fp16). Layout NN:
A (M,K) f16, B (K,N) s8 unpacked, C (M,N) f16.

Hashes:
- OCL_asm4edb78bd0e63742d -- M=64 N=17408 K=5120
- OCL_asm729735cb284f5d40 -- M=256 N=5120 K=5120

Inner loop is a scalar K walk: convert s8 weight and f16 act to
float, `acc_g += s * w`, then apply wei scale. GEN/VISA is `mad`,
not systolic.

grep of the dump dir: dpas / DPAS = no. s8 = yes (WEI_DT_S8 / char).

This is the incumbent floor for the harness call (unpacked KxN s8).
It is not a win vs K2 hand ESIMD s8 at 1024^3 (~374 us on this card).
