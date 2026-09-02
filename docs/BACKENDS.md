# Backends

On this host a "backend" is not a serving engine. It is the path from
a kernel or collective down to the two B70s. Name the path in every
CONFIG. Do not mix paths in one comparison.

Default on Battlemage / Xe2: SYCL and PyTorch XPU sit on Level Zero,
and the Level Zero V2 adapter (immediate command lists only) is the
Intel default for Arc B-series. OpenCL is the other NEO surface.
Vulkan is a separate stack. Measure, do not assume they agree.

## Stack on b70s4dayz

```
  llama.cpp SYCL     vLLM XPU / sglang XPU     custom kernels
           \                |                      /
            \               |                     /
             SYCL (DPC++ / intel-llvm)     Triton-XPU
              |         \                      |
              |          \                     |
              |        oneDNN / XeTLA /      SPIR-V
              |        sycl-tla / oneCCL         |
              |               \                  |
              +-------- Level Zero -------- IGC --+
              |               |                  |
           OpenCL ICD      L0 loader          ocloc
              |               |                  |
              +------ Compute Runtime (NEO) ------+
                              |
                         xe KMD  (card0, card1)
```

KMD is `xe`. UMD at lab creation: Compute Runtime / NEO 26.22.38646.4.
The two cards are separate Level Zero / OpenCL devices, not tiles of
one device.

## Level Zero

Direct-to-metal. Device discovery, USM, command lists, P2P, IPC,
metrics. This is the native compute API for Intel GPUs and the
default SYCL GPU backend.

On Arc B / Xe2 Intel's L0 V2 adapter is the default. It only supports
immediate command lists. Regular queued command lists are a V1 path;
do not copy Max-series or A-series submission notes onto this host
without re-measuring.

Selectors and masks:

- `ZE_AFFINITY_MASK` -- which devices the UMD exposes
- `ZE_FLAT_DEVICE_HIERARCHY` -- FLAT vs COMPOSITE (one stack per B70
  card, so these should match here; still record the value)
- `sycl-ls` should show `[level_zero:gpu][level_zero:0]` and
  `[level_zero:1]` for the two B70s

P2P, IPC, and immediate-list behavior are first-class TP=2 questions.
See Intel's Level Zero spec and the immediate-command-list note in
docs/REFERENCES.md.

## SYCL

Khronos C++ for accelerators. On Intel GPUs the implementation is
DPC++ (`intel/llvm`). Host code plus device kernels, USM or buffers,
in-order / out-of-order queues.

SYCL is not a device driver. It rides a backend:

| SYCL backend     | What it talks to              | Use on this host              |
|------------------|-------------------------------|-------------------------------|
| `level_zero`     | L0 loader + NEO               | Default. Prefer this.         |
| `opencl`         | OpenCL ICD + NEO              | Control / comparison only.    |

Pin it:

```
ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

AOT target for these B70s (Battlemage G31) is `intel_gpu_bmg_g31`
(`ocloc` acronyms include `bmg-g31`). Do not AOT as `intel_gpu_bmg_g21`
(B580 / G21) and call it a B70 kernel.

`sycl::get_native<sycl::backend::ext_oneapi_level_zero>(...)` is the
escape hatch from a SYCL queue to a `ze_command_list` / device handle.
Use it when a collective or copy has to be Level Zero-explicit.

## OpenCL

Same NEO UMD, different API. `clinfo` on this host already sees both
B70s as Intel OpenCL Graphics devices. Useful as a compiler/runtime
control against SYCL/L0, and as the path `ocloc` historically speaks.
Not the preferred launch path for new kernels.

## IGC / ocloc

Intel Graphics Compiler lowers SPIR-V (from SYCL, OpenCL, Triton) to
Xe2 ISA. `ocloc` is the offline compiler. Kernel experiments should
keep the IGC version next to the number. A "faster kernel" that
changed IGC is not a kernel result.

## oneDNN, XeTLA, sycl-tla

Library kernels, not a device backend, but they own a lot of GEMM and
attention on XPU:

- oneDNN -- PyTorch XPU conv/GEMM and many fused ops
- XeTLA -- Intel template kernels used inside IPEX / XPU ops
- sycl-tla -- SYCL CUTLASS port; CMake target `intel_gpu_bmg_g31`
  for Battlemage G31

When a PyTorch or vLLM number moves, ask which of these fired before
claiming a "kernel win".

## oneCCL / XCCL

Collectives for TP=2. PyTorch distributed on XPU is XCCL built on
oneCCL. llama.cpp tensor-split is a different map (host-side split +
device copies / SYCL queues), not XCCL.

`CCL_TOPO_P2P_ACCESS=1` is not a default. This host's two B70s sit on
separate PCI bridges. P2P is a labeled control after collectives are
healthy with P2P off. See AGENTS.md.

## PyTorch XPU

`device="xpu"`. Aten ops via SYCL (`intel/torch-xpu-ops`), GEMM/conv
via oneDNN, `torch.compile` via Triton-XPU + IGC, distributed via
XCCL/oneCCL. Serving engines (vLLM XPU, sglang XPU) sit here.

Graph capture / piecewise vs full is a serving-tree question
(`b70_ai_things`). This lab cares about the kernels and collectives
underneath it.

## Triton-XPU

`intel/intel-xpu-backend-for-triton`. Triton IR -> LLVM -> SPIR-V ->
IGC. Runtime still needs Level Zero (or SYCL) to allocate, submit,
and sync. Use it for written kernels we can read; do not treat a
Triton dump as a Level Zero program.

## Vulkan

Mesa / ANV, not NEO. Separate compiler, separate memory, separate
queue model. Community reports exist of Vulkan surviving dual-B70
work when Level Zero / SYCL reports DEVICE_LOST. It is a control
backend, not the kernel research default.

## Serving engines (not backends)

These choose a backend. Record both.

| Engine            | Typical backend path              | Notes                          |
|-------------------|-----------------------------------|--------------------------------|
| llama.cpp SYCL    | SYCL -> Level Zero                | tensor-split TP=2 is not XCCL  |
| vLLM XPU          | PyTorch XPU -> SYCL/L0 + oneCCL   | TP=2 is the dangerous path     |
| sglang XPU        | PyTorch XPU                       | serving-tree primary           |
| OpenVINO          | own runtime, often L0/OpenCL      | Intel product; not yet a lab   |
|                    |                                   | default here                   |

## How to name a run

Minimum CONFIG fields:

1. Backend: `level_zero` | `sycl+l0` | `sycl+opencl` | `opencl` |
   `pytorch-xpu` | `triton-xpu` | `vulkan`
2. Device pin: card0, card1, or both, plus affinity/selector
3. L0 adapter if known (V1 vs V2) and command-list mode
4. Compiler: IGC / icpx / ocloc / triton version
5. Collectives: none | oneCCL | llama.cpp split | host copies
6. P2P: off | on (labeled control only)

Then COMMAND, RESULT, VERDICT as usual.
