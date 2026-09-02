# References

Peek list for this lab. Community trees under `refs/` are attached as
git submodules so we can read them locally. They are not vendored
code and they are not our evidence. A number from a reference stays
in that reference until we reproduce it on b70s4dayz.

Init or update:

```
git submodule update --init --depth 1
```

## Community (attached)

### Steve / neural.download

- Site: https://neural.download/
- Git: https://github.com/steveseguin/b70-optimization-lab
- Local: `refs/b70-optimization-lab/`

Unofficial Intel XPU optimization lab. Repro packets, patches, TP2
vLLM/llama.cpp notes, and the measured B70 scoreboard behind
neural.download. Start at that repo's README, `docs/`, `repro/`, and
`docs/pcie-topology-and-llm-inference.md`. Our host is two B70s on
separate bridges; Steve's packets mix 1x / 2x / 4x -- copy the map,
not the card count.

### XeCores

- Site: https://xecores.com/

Independent Battlemage cookbook index (B70, B65, B60, B50, B580).
Architecture notes and recipe pointers. Not affiliated with Intel.
Useful for BMG-31 layout (32 Xe-cores, 256 XMX, 608 GB/s) and for
which community recipes exist. Numbers still need a local CONFIG.

### SergiioB inference cookbook

- Git: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
- Local: `refs/intel-arc-pro-b70-inference-cookbook/`

vLLM XPU and llama.cpp SYCL recipes for Arc Pro B60/B70. Dual-B70
TP2 / PP2 notes live in that tree's `docs/DUAL-B70-TP2.md`. Image
digests and patch matrices are family-specific; do not mix them.
Xe2 wedge watchdog notes are serving-ops, not a kernel finding.

### flashnext-harness

- Git: https://github.com/bbeartheancient/flashnext-harness
- Local: `refs/flashnext-harness/`

Qwen3.8-Flash-Next on 2x B70 via llama.cpp SYCL, plus SYCL/ESIMD
kernels (`kernels/m1multitronic-sycl/`). `ONEAPI_DEVICE_SELECTOR`
is Level Zero. Recipe in `docs/flashnext_recipe.md`. Peek for
tensor-split launch flags and Battlemage SYCL kernel CMake presets,
not as a serving default.

## Direct from Intel / oneAPI

These are the first-party specs and trees. Prefer them over blog
paraphrases when a backend or compiler question is in play.

### Linux GPU stack (this hardware)

- Intel GPU on Linux (dGPU docs): https://dgpu-docs.intel.com/
- Install paths, including Intel OMIX for Arc Pro B-series:
  https://dgpu-docs.intel.com/installation-guides/index.html
- OMIX install (pinned compute stack for B50/B60/B65/B70):
  https://dgpu-docs.intel.com/installation-guides/installing-omix.html
- Client GPU / xe KMD overview:
  https://dgpu-docs.intel.com/driver/client/overview.html

OMIX is Intel's named Arc Pro B-series compute bundle (Level Zero,
OpenCL, SYCL compiler, oneDNN, oneMKL). This host is already on
kernel 7.1 + NEO 26.22; treat OMIX as a documented Intel path, not
something to install on top of a working stack without a P0 identity
freeze.

### Level Zero

- Spec: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html
- Loader + headers: https://github.com/oneapi-src/level-zero
- SYCL Level Zero backend (intel/llvm):
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
- Immediate command lists, L0 V2 default on Arc B:
  https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html
- oneAPI GPU optimization guide, Level Zero chapter:
  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/level-zero.html
- Multi-tile / multi-card under the L0 SYCL backend:
  https://intel.github.io/llvm/MultiTileCardWithLevelZero.html
- Device hierarchy / `ZE_FLAT_DEVICE_HIERARCHY`:
  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/exposing-the-device-hierarchy.html

### SYCL / DPC++

- Khronos SYCL 2020: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- Intel DPC++ (intel/llvm): https://github.com/intel/llvm
- Get started / runtimes: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
- Environment variables (`ONEAPI_DEVICE_SELECTOR`, L0 V2):
  https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
- Device architecture names, including `intel_gpu_bmg_g31`:
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc
- oneAPI GPU optimization guide (Xe architecture, kernels, USM):
  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html

### Compute runtime, compiler, KMD

- Compute Runtime (NEO; implements L0 + OpenCL):
  https://github.com/intel/compute-runtime
- Intel Graphics Compiler: https://github.com/intel/intel-graphics-compiler
- xe kernel driver (upstream):
  https://www.kernel.org/doc/html/latest/gpu/xe/index.html
  (also `Documentation/gpu/xe/` in the kernel tree)

### Libraries that own real kernels and collectives

- oneDNN: https://github.com/uxlfoundation/oneDNN
- oneCCL: https://github.com/oneapi-src/oneCCL
- sycl-tla (SYCL CUTLASS; `intel_gpu_bmg_g31` for B70-class G31):
  https://github.com/intel/sycl-tla
- PTI-GPU (Level Zero tracing / timing): https://github.com/intel/pti-gpu

### PyTorch XPU / Triton (Intel-maintained)

- PyTorch XPU ops: https://github.com/intel/torch-xpu-ops
- oneCCL bindings for PyTorch: https://github.com/intel/torch-ccl
- Triton XPU backend: https://github.com/intel/intel-xpu-backend-for-triton
- PyTorch prerequisites for Intel GPUs (lists Arc B / Battlemage):
  https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html
- IPEX-LLM Battlemage quickstart (B-series, not B70-specific):
  https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md

### Product

- Arc Pro B70 is BMG-31 / `8086:E223` on this host. Intel's public
  product pages move; keep PCI ID + `sycl-ls` / `clinfo` as identity,
  not a marketing name.

## How to use a reference

1. Read it.
2. If it claims a kernel, collective, TP=2, or PP=2 fact, write a
   local experiment that names backend + device pin.
3. Put the local CONFIG -> COMMAND -> RESULT -> VERDICT in JOURNAL.md.
4. Only then promote into FINDINGS.md.

Do not paste a neural.download or XeCores tok/s figure into FINDINGS
as if this host produced it.
