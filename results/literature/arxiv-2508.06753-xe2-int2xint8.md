# arXiv 2508.06753 -- Xe2 int2 x int8 GEMM (XeTLA)

Status: LANDED (use v2, not v1).
Not local evidence. Paper numbers are Intel's, on Arc 140V and Battlemage B580, not these two B70s.

## Source

- Title: Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs
- Authors: Evangelos Georganas, Dhiraj Kalamkar, Alexander Heinecke (Intel)
- arXiv: 2508.06753
- Versions: v1 (2025-08, CPU only). v2 (2026-01-23) adds Xe2 GPU kernels.
- Fetched: https://arxiv.org/html/2508.06753v2
- PDF: https://arxiv.org/pdf/2508.06753

v1 HTML (ar5iv default) is the CPU paper. Campaign K2 cares about v2 section III-E.

## Claim that matters for K2

Xe2 natively supports int2 x int8 -> int32 DPAS. No upconvert of int2 weights to int8 is required (unlike the CPU path in the same paper).

Quote (III-E):

> Contemporary Xe2 Intel GPUs support natively int2 x int8 -> int32 GEMM computations using hardware-accelerated Dot Product Accumulate Systolic (DPAS) instructions. Therefore, for the Xe2 int2 kernels we can natively use these instructions without the need to upconvert the weights to int8 before performing the necessary multiply-add operations with the int8 activations.

They still need BF16 activations for the serving-shaped mixed GEMM, so the kernel they ship is fused:

  int2 weights (prepacked)  x  BF16 activations
  =  on-the-fly BF16->int8 quant  +  native int2xint8 DPAS  +  int32->BF16 dequant

Decoupled quant / GEMM / dequant kernels are called out as a GPU anti-pattern (extra launches and extra passes).

## ISA dump (the encoding K2 should match)

Figure 3 assembly, quoted as printed:

```
mul  (16|M0)  r96.0<1>:f   r87.0<1;1,0>:bf  r243.0<0;1,0>:f   // multiply with inv. scale
mov  (32|M0) (sat) r182.0<4>:b  r96.0<1;1,0>:f                 // convert to int8
mov  (32|M0)  r96.0<1>:b   r182.0<4;1,0>:b                     // move to proper register
...
dpas.8x8 (16|M0)  r63:d  r63:d  r208:s2  r96.0:b   // int2 x int8 DPAS
dpas.8x8 (16|M0)  r55:d  r55:d  r206:s2  r96.0:b   // r96 is reused 8 times
dpas.8x8 (16|M0)  r47:d  r47:d  r210:s2  r96.0:b
dpas.8x8 (16|M0)  r39:d  r39:d  r204:s2  r96.0:b
dpas.8x8 (16|M0)  r31:d  r31:d  r202:s2  ...
...
mov  (16|M0)  r18.0<1>:bf  r160.0<1;1,0>:f                     // convert to bf16
```

Reading of that encoding (cross-checked with IGC DPAS.md):

- Mnemonic: `dpas.8x8` = systolic depth 8, repeat count 8.
- Exec size: `(16|M0)` = 16-wide (PVC/Xe2 class, not DG2's 8).
- dst / src0: `:d` = int32 accumulator.
- src1: `:s2` = signed 2-bit (VNNI-packed weights).
- src2: `:b` = byte / int8 (quantized activations).
- Campaign guess `dpas.8x8 ... r:s2 ... r:b` is exactly this dump.

r96 (the int8 activation vector) is reused across 8 DPAS ops. That is the rectangular sg_tile reuse they describe: quantize B once, then hit it with several int2 A tiles.

## Packing: VNNI16, not VNNI4

Quote:

> The Xe2 int2 x int8 -> int32 DPAS instructions for Xe2 require the int2 operand to be formatted in VNNI16 layout: pack together 16 int2 values from the logical contraction K dimension of the GEMM. Since we are performing inference we can merely pre-format the int2 weights to such VNNI16 layout without any overhead during runtime.

CPU side of the same paper uses VNNI4-interleaved for AVX2 vpdpbssd. Do not mix those layouts. Xe2 wants 16 int2 along K packed into the systolic src1.

Load path in the figure: 2D-load of BF16 B, in-register scale+sat convert to int8, DPAS, then int32->float * dequant scales -> BF16 2D-store. Extra epilogue (bias, activation) is fused while C is still in registers.

## Tile / K-depth / autotune (XeTLA)

Hierarchy (SYCL names):

```
C[M,N]
  workgroup tile: wg_tile_m x wg_tile_n
    subgroup tile: sg_tile_m x sg_tile_n
      DPAS microkernel (2D-load + dpas.8x8)
```

K loop: iterate tiles along K. Accumulation is int32. Dequant of C happens after full K, or after a user-defined K granularity.

K-splitting: supported for small GEMMs (parallelism across K, then reduce). Same XeTLA knob as int8.

B scales: abs-max reduction, either a separate SYCL kernel before GEMM, or fused at workgroup level into SLM. Fuse-vs-not is a tuning knob. GEMV (N=1) wanted fused scales. Compute-bound GEMM (large N) wanted a separate scale kernel.

C dequant scales = (A_scales from the int2 weight matrix) * (B_scales).

Autotune framework inside XeTLA searches:

- fuse / not-fuse B scale calculation
- wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n
- K-splitting on/off

## Claimed MAC mix (paper, not a local roof)

Quote:

> the int2 x int8 DPAS instruction has the same throughput as the int8 x int8 DPAS instruction

They treat oneDNN int8xint8 as the attainable compute roof for the fused int2xBF16 kernel. That is the opposite of "int2 is 4x int8 TOPS". Mixed int2xint8 does not get a bitwidth rate multiplier; the win they advertise is 2-bit weight traffic at int8 systolic rate.

IGC DPAS.md agrees: OPS_PER_CHAN is 4 whenever either source is 8-bit, so K = 8 * 4 = 32 for s2xs8, same K as s8xs8. s2xs2 (both <8-bit) would use OPS_PER_CHAN=8 and K=64. See `igc-dpas-isa.md`.

Paper-reported platforms (do not copy as xe2x2 FINDINGS):

- Arc 140V (LNL iGPU): 8 Xe2 cores, shared DDR5. int8 peak they quote ~50 TOPS.
- Arc B580 (BMG G21): 20 Xe2 cores, 12 GB GDDR6, they quote int8 peak 233 TOPS, ~456 GB/s. Not a B70 (BMG-31, 32 Xe-cores).

## What the paper is not

- Not a B70 measurement.
- Not native int2 x int2 (they light mixed int2xint8).
- Not an ESIMD `xmx::dpas` listing; they wrap XeTLA templates.
- End-to-end serving numbers exist in the paper. They stay in the paper. Do not promote them.

## K2 takeaways

1. First mixed INT2 arm should be `s2 x s8` / `u2 x u8` -> s32, VNNI16 on the 2-bit operand.
2. Expect IGC text like `dpas.8x8 (16|M0) ... :d ... :s2 ... :b`.
3. Do not expect a 4x MAC roof vs s8 for this mix. Measure it.
4. Host-prepack VNNI16 is what they did; campaign already flags host-prepack as a BMG-G31 landmine vs transformed LSC 2D. Re-test both.
5. Standalone vs fat-tree is a separate bug (issue 21741). Their XeTLA kernels are a large SYCL tree.
