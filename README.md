# xe2x2

Kernel and 2x2 parallelism lab for the dual Intel Arc Pro B70 host.

This machine is two Battlemage (Xe2) GPUs. The charter is not generic
multi-GPU serving. It is:

1. Intel GPU kernels (Xe2 / B70, Level Zero, SYCL, IGC).
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
docs/        host, topology, and standing notes
```

## Standing rules

Read [AGENTS.md](AGENTS.md) before touching a GPU.

GPU access on this host is leased through `b70_ai_things` (`bin/gpu-run`).
Do not bypass the lease from this repo.

## Start here

1. Record host identity (kernel, KMD, UMD, L0, oneCCL, PyTorch) before
   changing anything. See `docs/HOST.md`.
2. Per-card health, then two-rank collective health, then TP=2 / PP=2.
3. Log every run as CONFIG -> COMMAND -> RESULT -> VERDICT.
