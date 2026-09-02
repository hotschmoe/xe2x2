# RESEARCH_TODO.md -- xe2x2 work order

Updated 2026-09-02. Work top to bottom. Do not mix environment refresh,
kernel changes, and parallelism-map changes in one comparison.

## P0: freeze the host baseline

- Record kernel, KMD, firmware, UMD / Compute Runtime, Level Zero
  loader, IGC, oneCCL, and PyTorch XPU identities into docs/HOST.md
  (extend the creation snapshot; do not replace it blindly).
- Record `sycl-ls` (level_zero vs opencl) and whether L0 V2 is the
  live adapter. See docs/BACKENDS.md.
- Run b70_ai_things per-card health on both B70s through gpu-run.
- Run two-rank collective health with P2P disabled.
- Confirm neither card is display-held and no live serve holds the
  lease.

Exit gate: identities recorded, both health layers green, no live
server, JOURNAL entry written.

## P1: kernel microbench floor

- One-card copy, GEMM, and attention-shaped kernels on card0, then
  card1, same binary.
- Capture IGC / SYCL / Level Zero identity with the numbers.
- Repeat after a cold lease so the first-run compile is not the score.

Exit gate: a tracked result under results/ that names the kernel, the
card, the compiler, and the health state.

## P2: TP=2

- Synthetic all-reduce / all-gather / send-recv across the two cards.
- Then a tiny sharded matmul or tiny-model TP=2 map.
- P2P off first. P2P on only as a labeled control after collectives
  are healthy.

Exit gate: correctness, matched rank evidence, teardown, post-health.
Speed is secondary.

## P3: PP=2

- Two-stage synthetic activation handoff, one stage per card.
- Measure bubble, copy path (host vs device), and stage-memory split.
- Tiny-model PP=2 only after the synthetic handoff is correct.

Exit gate: same as P2.

## P4: mixed 2x2

- Only after P2 and P3 pass on this host with the current UMD.
- Start with TP=2 inside a two-stage pipeline, then the reverse, as
  separate experiments.

Exit gate: a written verdict on whether 2x2 is a real map on this PCI
tree or a paper map that dies in collectives.

## Standing bans

- Do not skip health to get a speed number.
- Do not reuse quarantined wheels, .so files, or compiler caches from
  b70_ai_things as proof that xe2x2 work is clean.
- Do not promote a serving wrapper from this repo. Hand findings back
  to b70_ai_things.
