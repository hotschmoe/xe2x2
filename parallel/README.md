# parallel

TP=2, PP=2, and 2x2 protocols for the dual B70 host.

Kernel math stays under `../kernels/`. This tree is the two-card
protocol. Campaign map: `../docs/KERNEL_CAMPAIGN.md`.

## Axes

- `tp2/` -- tensor parallel = 2. One shard per card. Collectives and
  weight-split correctness first, then speed. Push all-reduce, call
  count, and P2P are arms.
- `pp2/` -- pipeline parallel = 2. One stage per card. Activation
  handoff, bubble, and stage-memory split first, then speed.
- `2x2/` -- mixed maps. Only after both single-axis trees have a
  passing health + correctness run on this host.

## Dual-card vs one-card

A TP=2 / PP=2 / 2x2 job uses both cards. Pause the one-card kernel
matrix (`gpu-run --card 0` || `--card 1`) while a parallel protocol
runs. Do not steal one card out from under a collective.

## Required sequence

1. Per-card health.
2. Two-rank collective health.
3. Correctness on a tiny model or synthetic tensor.
4. Teardown and re-health.
5. Then scale.

PCI topology is documented in `docs/HOST.md`. P2P is a measured
property of this machine, not an environment variable you flip.
