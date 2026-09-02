# K1 -- incumbent Intel kernel ISA

Question: what does IGC actually emit for the well-optimized Intel
ops we already call from vLLM / oneDNN / XeTLA / sycl-tla?

Open. The dump is a floor. The lab attitude is that no one has
tried a real SYCL/L0 kernel yet; matching Intel is not a win.

## Why

Hand ESIMD that ignores the incumbent schedule will lose for dumb
reasons (VNNI layout, scratchpad, scale epilogue, K-blocking). Steal
the schedule first.

Sibling hypotheses to reproduce: oneDNN owns `fp8_gemm_w8a16`,
`int8_gemm_w8a16`, `int8_gemm_w8a8`, `nvfp4_gemm_w4a16` (E2M1
decompress, not INT4 DPAS); portable APIs floor at INT8; transformed
LSC VNNI was the bit-exact load path on BMG-G31.

## Suggested arms

- Capture IGC ISA / NEO shader dumps for each reachable op at one
  decode shape (M=1) and one prefill shape (M>=64).
- Note joint dtype, scale mask (per-tensor / per-channel / block),
  NT vs NN, scratchpad.
- If XeTLA or sycl-tla (`intel_gpu_bmg_g31`) is reachable in the
  current image, dump those too as controls.
- Disasm: is there a `dpas.s8`, `dpas.s4`, or decompress loop?

Card0 || card1 can split the op list, then swap.

## Record

Op name, backend (`pytorch-xpu` or `sycl+l0`), IGC version, dump
path, a short reading of the inner loop (DPAS or not, K-depth,
repeat count). No tok/s.

## Exit

Dumps under `results/` (or gitignored artifact dir with a pointer).
A JOURNAL reading. FINDINGS only if the ISA fact is stable across
both cards and a second IGC.
