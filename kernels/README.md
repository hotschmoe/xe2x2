# kernels

Device-side work for Xe2 / Arc Pro B70.

Campaign map: `../docs/KERNEL_CAMPAIGN.md`. One question per
experiment directory. Name the backend in the experiment README
(`sycl+l0` default; OpenCL and Vulkan are controls). AOT for these
cards is `intel_gpu_bmg_g31`. See `../docs/BACKENDS.md`.

Language default: C++ SYCL / ESIMD for device code, Python for the
harness.

## Dual-card

One-card work runs two-wide: `gpu-run --card 0` || `gpu-run --card 1`.
Never mix a two-card collective with a one-card job. Split a matrix
across cards, then swap so every arm has both-card evidence.

## Workstreams

| Id | Directory | Question (open) |
|----|-----------|-----------------|
| K0 | `roofline/` | Percent of 608 GB/s and 367 INT8 TOPS |
| K1 | `onednn_isa/` | What IGC emits for current Intel ops |
| K2 | `esimd_dpas/` | Hand s8 / s4 / s2 DPAS microkernels |
| K3 | `precision_compose/` | INT2/INT4 terms as INT8 or E2M1 |
| K4 | `w8_compare/` | FP8 W8A16 vs INT8 W8A16 vs INT8 W8A8 |
| K5 | `epilogue_quant/` | INT8 without ~160 quant launches |
| K6 | `nvfp4/` | Every NVFP4 spoof, split, and LUT |
| K7 | `gdn/` | GDN / hybrid linear-attn leftover |

Parallel maps (TP=2, PP=2, 2x2) live under `../parallel/`.

## Standing questions this tree exists to answer

- What actually runs well on Battlemage EU / XVE / XMX for this host.
- How IGC compiles the kernels we care about (GEMM, attention,
  epilogue, collect, copy).
- Single-card vs two-card occupancy, bandwidth, and cache behavior.
- Which kernel families are the bottleneck once TP=2 or PP=2 is
  attached.

Do not dump serving wrappers here. If a kernel only matters as part
of a parallel map, keep the kernel here and the protocol under
`../parallel/`.
