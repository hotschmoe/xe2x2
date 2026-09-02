# results/literature -- campaign paper notes

Read-only fetch, 2026-09-02. ASCII. Not FINDINGS. No local GPU work.

Banner: nothing in this directory is a xe2x2 measurement. Paper TOPS, GB/s, and serving rates stay in the source. Do not promote them.

## Priority sources (KERNEL_CAMPAIGN + REFERENCES campaign literature)

```
file                                     source                         fetch
arxiv-2508.06753-xe2-int2xint8.md        arXiv 2508.06753 v2            LANDED
intel-llvm-issue-21741.md                intel/llvm#21741               LANDED (still OPEN)
esimd-dpas-api.md                        intel/llvm sycl_ext_intel_esimd LANDED
igc-dpas-isa.md                          IGC visa DPAS.md               LANDED (companion)
xetla-int8-gemm.md                       intel.github.io/xetla 0.3.7    LANDED
nvfp4-vs-ocp-mx.md                       NVFP4 blog + TE + OCP MX       LANDED (see caveat)
qserve-qoq-2405.04532.md                 arXiv 2405.04532v3             LANDED
gated-deltanet.md                        arXiv 2412.06464v3 + FLA       LANDED
oneapi-gpu-opt-xe-architecture.md        oneAPI GPU opt guide 2025.2    LANDED
```

### Caveats

- 2508.06753: v1 HTML is CPU-only. Use v2 (2026-01-23) for Xe2 int2xint8, VNNI16, XeTLA autotune.
- intel/llvm#21741: open, no official workaround. Standalone oracle is the practical split.
- OCP MX v1.0 official PDF (opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) was not fully dumped as text here. Element/scale tables are from that spec as quoted by arXiv 2310.10537, NVIDIA TE, and later papers. If a bit ever fights the PDF, the PDF wins.
- XeTLA public tree is archived; int2 autotune from 2508.06753 is not in v0.3.7 docs.
- oneAPI Xe2-HPG table is Arc B580 (20 Xe-cores), not B70 (32).

## Campaign extras, this pass

```
sycl-tla / CUTLASS Xe, intel_gpu_bmg_g31     NOT FETCHED (named as XeTLA successor only)
PTI-GPU / unitrace                           NOT FETCHED
QuaRot / FlatQuant / PrefixQuant             SKIPPED (campaign: only when W4A4/Hadamard is the question)
FlashAttention on XPU                        SKIPPED (campaign: control, do not start there)
b70_ai_things docs/P2P_GPU.md J.13-J.19      SKIPPED (sibling fabric, not a paper)
Gated DeltaNet-2 2605.22791                  noted inside gated-deltanet.md, not a full file
```

No 404s. No paywalls. GitHub issue HTML view flaked once; API body + comments landed.

## K2 facts in one place (literature, not silicon)

Native mix: Xe2 DPAS does int2 x int8 -> int32. Encoding in 2508.06753:

```
dpas.8x8 (16|M0)  rD:d  rAcc:d  rW:s2  rA:b
```

Packing: VNNI16 on the int2 operand (16 int2 along K). Throughput claimed equal to int8xint8 DPAS, not 4x. IGC: mixed 8-bit forces OPS_PER_CHAN=4, K=32, same as s8xs8. s2xs2 would be K=64.

s2 integer range is [-2, 1], not E2M1 and not s4.

Issue 21741: ESIMD DPAS can be correct standalone and wrong for WI!=0 inside a fat SYCL process on B70. Still open. K2 needs a standalone oracle.
