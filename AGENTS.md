# AGENTS.md -- standing rules for xe2x2

Keep this file short. Evidence lives under results/ and docs/.

## Charter

This lab is for Intel GPU kernels and 2x2 parallelism on the dual B70 host.

In scope:

- Xe2 / B70 kernels (SYCL, Level Zero, IGC, OpenCL, Triton-XPU).
- Tensor parallel = 2.
- Pipeline parallel = 2.
- Collectives, P2P, PCI topology, and the failure modes of the above.

Out of scope unless explicitly expanded:

- Production serving and model-shelf promotion (`b70_ai_things`).
- Single-card model sweeps that do not feed a kernel or 2x2 question.
- NVIDIA / CUDA work.

## Style and evidence

- ASCII only in repository files and terminal output.
- Record experiments as CONFIG -> COMMAND -> RESULT -> VERDICT.
- Do not claim speed or stability without matched configuration, coherence,
  identity, health, and teardown evidence.
- Preserve user changes in a dirty worktree.

## Host and GPU discipline

- Work locally as hotschmoe in `/mnt/vm_8tb/github/xe2x2`.
- Use `/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run` for every real GPU
  touch: serving, benchmarking, profiling, XPU compilation, collectives.
- Use `gpu-run --card N` for a one-card workload and pair it with the
  workload's device pin.
- Kernel 7.1.0-070100 is the host baseline. Do not downgrade it to imitate
  an older result.

## Multi-GPU safety

Kernel 7.1 plus Compute Runtime 26.22 cured the former GuC/BCS hardware
wedge. Software TP>1 / oneCCL / queue-handoff failures still exist.

- Do not run arbitrary `CCL_TOPO_P2P_ACCESS=1` TP>1 serves.
- Repeated TP>1 worker-init or graph-capture crashes can poison later
  collectives even with P2P disabled.
- Run per-card health, then two-rank collective health, around risky TP=2
  or PP=2 work. Teardown and re-check after failures.
- The two B70s are on separate PCI bridges. Treat P2P as a hypothesis,
  not a given.

## Change discipline

When updating drivers, PyTorch, IGC, oneCCL, or a backend:

1. Record host kernel, UMD, Level Zero, oneCCL, compiler, and image
   identity before changing anything.
2. Change one layer at a time.
3. Rebuild every ABI-specific extension from tracked source.
4. Re-run per-card and two-rank collective health before claiming TP=2
   or PP=2 progress.
