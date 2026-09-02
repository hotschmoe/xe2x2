# xe2x2

Kernel and 2x2 parallelism lab for the dual Intel Arc Pro B70 host.

This machine is two Battlemage (Xe2) GPUs. The charter is not generic
multi-GPU serving. It is:

1. Intel GPU kernels (Xe2 / B70). Level Zero is the native path; SYCL
   rides it (or OpenCL as a control). IGC compiles. See
   [docs/BACKENDS.md](docs/BACKENDS.md).
2. Tensor parallel = 2 (one shard per card).
3. Pipeline parallel = 2 (one stage per card).
4. The 2x2 combinations those two axes imply, including why they fail.

`b70_ai_things` remains the serving and model-shelf lab. This repo is the
arena: kernels, collectives, topology, and TP=2 / PP=2 experiments that
should not live in the serving tree.

## Host

- Machine: `b70s4dayz`
- Path: `/mnt/vm_8tb/github/xe2x2`
- CPU: AMD Ryzen Threadripper 1950X (16c/32t), ~121 GiB RAM
- GPUs: 2x Intel Arc Pro B70 (Battlemage G31, PCI `8086:E223`)
  - card0 `0000:0b:00.0` / `renderD128` -- 256 EU, 30.3 GiB
  - card1 `0000:44:00.0` / `renderD129` -- 256 EU, 30.3 GiB
- KMD: `xe`
- UMD snapshot at repo creation: Compute Runtime / NEO `26.22.38646.4`
- Host kernel snapshot at repo creation: `7.1.0-070100-generic`

The two cards sit on separate Intel PCI bridges (`09:00.0` and `42:00.0`).
That topology is a first-class research object for TP=2, not a footnote.

Full inventory: [docs/HOST.md](docs/HOST.md).

## Layout

```
kernels/     device kernels, microbenchmarks, IGC/SYCL/L0 traces
parallel/    TP=2, PP=2, and 2x2 experiment protocols
results/     small tracked evidence (large artifacts stay gitignored)
docs/        host, backends, references, topology
refs/        attached community trees (git submodules)
scripts/     xe2x2 harnesses; GPU lease stays in b70_ai_things
```

## Standing rules

Read [AGENTS.md](AGENTS.md) before touching a GPU.

GPU access on this host is leased through `b70_ai_things` (`bin/gpu-run`).
Do not bypass the lease from this repo.

## Start here

- [AGENTS.md](AGENTS.md): safety, scope, and GPU lease. This is the
  only agent file.
- [RESEARCH_TODO.md](RESEARCH_TODO.md): work order, P0 first.
- [docs/KERNEL_CAMPAIGN.md](docs/KERNEL_CAMPAIGN.md): open kernel,
  INT2/INT4/NVFP4, and dual-card campaign. Not a locked design.
- [docs/MODELS.md](docs/MODELS.md): what to serve after the math
  floor (27B dense, 35B-A3B MoE, Gemma, Flash-Next).
- [docs/AGENT_LAUNCH.md](docs/AGENT_LAUNCH.md): how not to collide
  once several agents run.
- [docs/ORCHESTRATOR.md](docs/ORCHESTRATOR.md): paste-ready prompt
  for a new session that runs the campaign.
- [FINDINGS.md](FINDINGS.md): current evidence ledger.
- [JOURNAL.md](JOURNAL.md): newest experiment window.
- [docs/HOST.md](docs/HOST.md): measured inventory of this machine.
- [docs/BACKENDS.md](docs/BACKENDS.md): Level Zero, SYCL, OpenCL,
  PyTorch XPU, Triton-XPU, oneCCL.
- [docs/REFERENCES.md](docs/REFERENCES.md): community labs plus
  first-party Intel / oneAPI trees.

Peek locally (after `git submodule update --init --depth 1`):

- `refs/b70-optimization-lab/` -- Steve, also https://neural.download/
- `refs/intel-arc-pro-b70-inference-cookbook/`
- `refs/flashnext-harness/`
- https://xecores.com/ (site only, not cloned)

Work order:

1. Record host identity (kernel, KMD, UMD, L0, oneCCL, PyTorch) before
   changing anything.
2. Per-card health, then two-rank collective health, then TP=2 / PP=2.
3. Log every run as CONFIG -> COMMAND -> RESULT -> VERDICT.
