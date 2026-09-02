# K1 card0 fp8_gemm_w8a16 ISA (20-line reading)

Image: b70-local/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50-s01
HAS fp8_gemm_w8a16=yes. Backend pytorch-xpu on sycl+l0. torch 2.13.0+xpu. oneDNN v3.13.0 (0e2a5bfe). IGC 2.38.2 git 3eef0f89d. L0 1.15.39122. binary_kernels=enabled.
Joint: src bf16, wei f8_e4m3, dst bf16. NN. Scale wei mask=2 f32 (per-channel N; harness s=ones(1,N)). Scratchpad user.
xe_hp_systolic skipped: M=1 use_nocopy (jit_xe_hp_systolic.cpp:86); M=64 unsupported f8 combo (:107). Winner: gpu,matmul,jit:gemm:any (gemmstone ngen).

IGC dump results/k1/igc_card0_fp8: 19 files, kernel gemm_zero_fill only (OpenCL, simd32, 36 inst, store.ugm zeros).
grep dpas / DPAS / s8 / decompress on that dir: no matches. IGC dump contains dpas: no.

ONEDNN_JIT_DUMP results/k1/igc_card0_fp8_jit: ngen gemm_kernel.0.bin=26KB (M=1), .4.bin=115KB (M=64); identical twins .2/.5. ocloc fails (raw ISA, not zebin). iga Xe2 disasm ok.
M=1 catalog: gemm qB[SB] N@16N@16N k32 wg 16x1 sys ska. 16x `dpas.8x8 (16|M0) rD:f rAcc:f rW:bf rA.0:bf`. Loads: block2d.ugm.d8v (wei) + d16 (act).
Decompress (not a named loop): `shl :ub << 8`, `cmp abs:hf vs 0x7F0`, `mul :bf * 1.329228e+36`, `or 32767` on nan, then bf16 DPAS. No dpas.s8.
M=64 catalog: sys ks64 grf256 wg 4x8 cab4 k256 nmk. 256x dpas.8x8 bf; 608 send.slm (SLM pack). Same f8->bf shift/scale, then systolic from SLM.

IGC path has no dpas. ngen GEMM has dpas.8x8 bf16xbf16, f32 acc. Floor is emulated FP8, not native FP8 XMX.
Clocks card0: before D3hot gt0 cur=1167 max=2800 temp52C; after D3hot gt0 cur=1583 temp52C; act=0 both (sampled idle).
HardwareCaps: EUCount=256 SubSlice=32 UsDeviceID=0xe223 platform XE2. HAS fp8_gemm=True; int8/nvfp4 ops absent in this image.
Timed us (warmup10/iters30) live in results/k1/fp8_card0.txt. M=256 N=K=5120 also 318.025 us.
