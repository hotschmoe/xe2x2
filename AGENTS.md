# AGENTS.md -- standing rules for xe2x2

Keep this file short. Evidence lives in JOURNAL.md, FINDINGS.md, results/,
and docs/. This file is the only agent briefing. Do not add CLAUDE.md
or other vendor agent files.

## Charter

This lab is for Intel GPU kernels and 2x2 parallelism on the dual B70 host.

In scope:

- Xe2 / B70 kernels. Native path is Level Zero. SYCL (DPC++) rides
  Level Zero by default, or OpenCL as a labeled control. IGC compiles.
  Triton-XPU and PyTorch XPU sit on that same L0/SYCL stack. See
  docs/BACKENDS.md.
- Tensor parallel = 2.
- Pipeline parallel = 2.
- Collectives (oneCCL / XCCL vs llama.cpp tensor-split), P2P, PCI
  topology, and the failure modes of the above.

Kernel / math campaign map: `docs/KERNEL_CAMPAIGN.md`. After P0,
one-card kernel jobs run two-wide (`gpu-run --card 0` || `--card 1`).
Launch pairing and landmines: `docs/AGENT_LAUNCH.md`.
New-session orchestrator prompt: `docs/ORCHESTRATOR.md`.
Intel/oneDNN/XeTLA numbers are floors to beat, not ceilings.
Serving-shaped kernels rank by wall time (us), not TOPS%.
TP=1 vs TP=2 is per-op; fusion that drops a collective is a latency
win. A 3rd/4th B70 is evidence-gated, not current hardware.

Out of scope unless explicitly expanded:

- Production serving and model-shelf promotion (`b70_ai_things`).
- Single-card model sweeps that do not feed a kernel or 2x2 question.
- NVIDIA / CUDA work.

## Style and evidence

- ASCII only in repository files and terminal output.
- Record experiments as CONFIG -> COMMAND -> RESULT -> VERDICT.
- Append new JOURNAL.md entries at the bottom.
- Promote a durable result into FINDINGS.md. Do not leave it only in
  the journal.
- Do not claim speed or stability without matched configuration, coherence,
  identity, health, and teardown evidence.
- A napkin, guess, or "this should lose" is a CONFIG prior. It is not
  a RESULT. Measure with code on these cards; surprises belong in
  FINDINGS.md.
- Name the backend in every CONFIG (`level_zero`, `sycl+l0`,
  `sycl+opencl`, `opencl`, `pytorch-xpu`, `triton-xpu`, `vulkan`).
- Community numbers from refs/ or neural.download / xecores.com are
  not local evidence. Reproduce on this host before FINDINGS.md.
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
