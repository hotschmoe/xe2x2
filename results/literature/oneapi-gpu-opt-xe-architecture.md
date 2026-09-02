# oneAPI GPU optimization guide -- Xe architecture (XVE vs XMX, GRF, SLM, DPAS)

Status: LANDED. Guide 2025.2, dated 2025-07-10.
Not local evidence. Table rows are Intel's product sheet. B70 is BMG-31 / 32 Xe-cores; the guide's Xe2-HPG row is B580 (20 Xe-cores). Scale, do not copy.

## Source

- Overview: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html
- Xe architecture: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- GRF modes: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/small-register-mode-vs-large-register-mode.html
- SLM: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html
- XMX / joint_matrix: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html
- PDF: https://cdrdv2.intel.com/v1/dl/getContent/859422?fileName=oneapi_optimization-guide-gpu_2025.2-771772-859422.pdf

## Hierarchy (guide's terms)

```
GPU architecture:  Xe
generation:        Xe, Xe2, Xe3, ...
micro-arch:        Xe LP/LPG, Xe HPG, Xe HPC, Xe2-HPG, Xe2-LPG, ...
model:             Arc B580, Arc A770, ...
building block:    Xe-core (XC)
  Vector Engine    (XVE)  -- formerly EU
  Matrix Engine    (XMX)  -- DPAS / systolic, formerly "DPAS part of EU"
```

Xe-core has vector ALUs and matrix ALUs. Xe2-HPG: 8 XVE and 8 XMX per Xe-core (guide text for Xe HPC says 8 and 8; Xe2-HPG table matches 8 VE/core).

One or two Xe Stacks make a GPU. Client dGPUs are one stack.

## Table row: Xe2-HPG (guide cites Arc B580, not B70)

```
Xe-cores                         20          (B70 is 32; do not use 20 as ours)
XVE per Xe-core                   8
XVE count                       160
HW threads per XVE                8
HW thread count                1280
XMX / DPAS                      Yes
FP64 native                     Yes
GRF per thread                  128 / 256 (regular / large)
Register width                  512 bits     // 64 B per GRF
Global memory                   12 GB        (B70 is 32 GB class; not this row)
L3                              18 MB
L1 per Xe-core                  256 KB
SLM per Xe-core                 128 KB
Max SLM per work-group          128 KB
Max work-group size            1024
Sub-group sizes                16, 32
```

Xe2-LPG (Lunar Lake iGPU) has the same 128/256 GRF dual mode and 512-bit registers. Xe-HPG Alchemist (A770) is 128 GRF only, 256-bit registers, 16 XVE per core.

Query helper in the guide (Xe-core count = slices * subslices_per_slice):

```
gpu_slices
gpu_subslices_per_slice
gpu_eu_count_per_subslice      // XVE per Xe-core
gpu_hw_threads_per_eu
global_mem_size
local_mem_size                 // SLM per work-group
max_work_group_size
sub_group_sizes
```

L1/L3 are not queryable; use the spec table.

## XVE vs XMX

XVE: SIMD ALU threads. Each hardware thread executes SIMD 16 or 32. ALUs cover FP64/32/16, BF16, INT64/32/16/8, etc.

XMX: matrix engine, DPAS. Guide programs it two ways:

1. SYCL joint_matrix (`sycl::ext::oneapi::experimental::matrix`). Requires `aspect ext_intel_matrix` (`sycl-ls --verbose`). No emulation. "DPAS is the name of the elementary operations done in Intel XMX."
2. ESIMD `xmx::dpas` (see `esimd-dpas-api.md`) when joint_matrix is gated off.

joint_matrix shape examples in the guide:

```
PVC-class (nsize==16):  int8  TM=8 TN=16 TK=32
                        bf16  TM=8 TN=16 TK=16
DG2-class (nsize==8):   int8  8x8x32
                        bf16  8x8x16
```

Tuning they recommend for XMX GEMM: subgroup doing 32x64x16 (via 16 mad calls reusing 4 A, 4 B, 16 C, or a 32x64x16 joint_matrix shape); work-group cache blocking e.g. 256x256x32; Large GRF; `joint_matrix_prefetch`.

BMG31 joint_matrix: issue 21741 / dkhaldi, needs 2026.0+. Until then ESIMD is the XMX door.

## GRF 128 vs 256

Xe2 (and PVC): each XVE has 64 KB register space.

```
Small / regular: 128 GRF x 64 B,  8 threads per XVE
Large:           256 GRF x 64 B,  4 threads per XVE
```

Occupancy halves in large mode. Campaign hail-mary 3 is this A/B.

Compiler switch (guide still names `pvc:` as the device key):

```
-ftarget-register-alloc-mode=pvc:default|small|large|auto
```

JIT SYCL:

```
icpx -fsycl -ftarget-register-alloc-mode=pvc:large  file.cpp
```

AOT:

```
icpx -fsycl -fsycl-targets=spir64_gen \
  -ftarget-register-alloc-mode=pvc:large \
  -Xsycl-target-backend "-device pvc"   # or the bmg device name
```

Older form also seen: `-ze-opt-large-register-file` on the backend.

`default` currently means small. `auto` lets IGC pick per kernel.

Guide's own GEMM advice: large GRF "gives the best results for the GEMM kernel". That is PVC-era advice, not a B70 finding.

## SLM

Per Xe-core, on-chip, visible to the work-group on that core. Lifetime = work-group.

Xe2-HPG: 128 KB SLM per Xe-core and per work-group max. Xe-LP was 64 KB.

Banks: "at the time of writing, 64 consecutive bytes stored in 16 consecutive banks at 4-byte granularity." Same-bank different addresses serialize.

Use: work-group managed cache, histogram bins, k-tile staging for GEMM. Barriers have a cost that grows with WG size.

Over-request: `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Debug:

```
export PrintDebugMessages=1
export NEOReadDebugKeys=1
```

prints `Size of SLM (N) larger than available (131072)` (128 KB).

2508.06753 fuses B abs-max into SLM as a GEMV knob.

## Sub-groups on Xe2

Supported sizes 16 and 32. ESIMD is subgroup size 1 (one HW thread per ESIMD work-item); the simd<> length is the vectorization.

## K2 / hail-mary mapping

- XMX = DPAS. XVE = everything else (quant, epilogue, GDN recurrent, conv1d).
- Dual GRF is real on Xe2. Measure occupancy vs blocking; do not pick a religion.
- SLM 128 KB/core is the staging budget for K-split and fused scales.
- joint_matrix may be off on this compiler; ESIMD is the documented escape hatch.
