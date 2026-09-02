# intel/llvm#21741 -- B70 ESIMD DPAS wrong in large SYCL builds

Status: LANDED, still OPEN as of 2026-06-11 (issue updated; no close).
Not local evidence. Reproduction is on a B70, but a different host/stack than this lab.

## Source

- URL: https://github.com/intel/llvm/issues/21741
- API: https://api.github.com/repos/intel/llvm/issues/21741
- Opened: 2026-04-12 by PMZFX
- State: open. Label: esimd.
- Comments: 4 (through 2026-04-27). Last issue update 2026-06-11.

## Environment in the report (not this host)

- GPU: Intel Arc Pro B70 (BMG/Xe2, PCI 0xe223)
- Driver: libze-intel-gpu1 26.09.37435.1
- IGC: intel-igc-core-2 2.30.1
- Compiler: Intel oneAPI DPC++ 2025.3.3 (2025.3.3.20260319)
- Runtime: Level-Zero V2, 1.14.37435+1
- OS: Ubuntu 26.04, kernel 7.0.0-12-generic

This lab's baseline is kernel 7.1 + Compute Runtime 26.22. Do not assume the bug is gone or still present without a local standalone-vs-fat-tree A/B.

## Kernel that failed

Flash attention in ESIMD:

- `[[intel::sycl_explicit_simd]]`
- `xmx::dpas<8, 8, float>` for QK^T and PV (SystolicDepth=8, RepeatCount=8, float acc)
- HEAD_DIM 64 and 128
- Each work-item computes 8 output rows (RC=8)
- `block_load` / `gather` / `block_store` / `exp` / `convert`
- nd_range<3>: dim 0 = head/batch, dim 2 = query tiles

This is bf16/fp16-style DPAS, not the int2 mix. The bug is about ESIMD DPAS numeric identity inside a fat SYCL process, so it still gates every K2 number.

## Failure pattern

Deterministic:

- work-item 0 in `global_id(0)` always correct
- work-items 1+ in `global_id(0)` always wrong
- Wrong values look like plausible attention, not garbage
- ERR ~1.3-1.5 vs expected <0.0005
- Stable across runs

## What passes (same kernel source)

1. Standalone test binary
2. Standalone with fat-project flags (`-O3`, `-fPIC`, `-DNDEBUG`, GGML_BACKEND_*)
3. Small shared .so called from a separate executable
4. Small .so + 9 extra SYCL TUs
5. Small .so + oneMKL SYCL BLAS
6. Small .so with 64+ extra kernel call sites (grown offload payload)
7. Standalone `-O0`
8. Standalone Large GRF

## What fails

1. Shared library as part of a large SYCL backend (~50+ other SYCL sources)
2. Same large project, static lib (`BUILD_SHARED_LIBS=OFF`)
3. Large backend loaded via dlopen
4. Kernel moved to its own .so with its own device image, still linked/loaded by the large backend (ELF section split verified)
5. `queue.wait()` before launch (rules out leftover queue state)
6. `-fsycl-device-code-split=per_kernel` and `per_source`
7. `-O0` in the integrated build
8. Large GRF in the integrated build
9. `SYCL_ESIMD_FUNCTION`
10. Split HEAD_DIM templates into separate TUs

Reporter's one-liner:

```
Standalone binary with ESIMD DPAS kernel -> PASS (all work-items correct)
Same kernel source, compiled as part of large SYCL project -> FAIL (work-item 0 correct, 1+ wrong)
Same kernel source, in separate .so with own device image, loaded by large SYCL project -> FAIL
```

## Workarounds that did NOT work

Device-image split, per_kernel split, -O0, Large GRF, queue drain, SYCL_ESIMD_FUNCTION, template splitting, shared-vs-static. The trigger appears to be "process also compiled/loaded a large real-world SYCL backend (~50+ TUs)". A synthetic mini-repro with ~10 TUs plus a fat offload payload did not trip it.

There is no Intel-endorsed workaround in the comments. Practical lab workaround from the report itself: keep a standalone oracle binary. Campaign hail-mary 4 already says this.

## Intel / sibling comments

sarnex (2026-04-13): hard to investigate without a reproducer.

fineg74 (2026-04-14): maybe fast-math, or some other kernel setting a rounding mode / other state that then affects DPAS. Puzzling that WI0 is correct and the rest are not, and that it needs a large kernel set. Could be driver-side. Still wants a reproducer.

Matt-McPherson (2026-04-25): different stack, same chip family. OpenCL `intel_sub_group_f16_f16_matrix_mad_k16` compiles, ocloc emits real DPAS, runtime returns all zeros. `ext_oneapi_matrix` not listed on B70 (Level Zero IP version 0x0 for 0xe223, SYCL gates the matrix aspect on IP version). Their NEO 26.14.37833 / IGC 2.32.7 is newer than the issue's 26.09 and still zeros. Different wrong-value pattern (all zeros vs WI0-correct). Treat as a sibling hypothesis, not this issue.

dkhaldi (2026-04-27): BMG31 joint_matrix support landed end of 2025. If using < 2026.0, `ext_intel_matrix` false is expected. Use 2026.0 or intel/llvm OSS builds. This answers the aspect gate, not the ESIMD wrong-results bug.

## K2 implication

Every K2 numeric result needs a standalone ESIMD binary oracle. A fat SYCL tree (llama.cpp backend, vLLM plugin, XeTLA-in-process) can pass compile and still compute the wrong tile for WI != 0.

Do not treat "dpas compiled" as "dpas is numerically closed" unless the standalone path agrees.
