# RESEARCH_TODO.md -- xe2x2 work order

Updated 2026-09-02. P0 is the only hard gate. After P0, kernel
workstreams (K0-K6) may run in any order. Do not mix environment
refresh, kernel changes, and parallelism-map changes in one
comparison.

Campaign map (open questions, not a locked path):
`docs/KERNEL_CAMPAIGN.md`.

## Dual-card scheduling

Two B70s. Independent one-card kernel/math jobs run two-wide
(`gpu-run --card 0` || `gpu-run --card 1`). Two-card collectives
pause that matrix. Split experiment matrices across cards, then swap
so every arm has both-card evidence.

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

P0 passed 2026-09-02g (`results/p0/SUMMARY.md`, docs/HOST.md freeze).
K0-K6 have both-card RESULTS. Held-clock scale-to-f16
M=1 (2026-09-02ap) beats W8A8 34 vs 44 us. Same RC=4
tile at M=64 (2026-09-02aq) is 243-247 us pipe vs
W8A8 46 us, both cards, cur=2800 act~2.7 GHz
throttle=1, cosine=1. Occupancy cut the 16x napkin
to ~7.4x M=1; still 5.3x the incumbent. Next: ngen
M=64 RC=8/GRF256/SLM on the f16 contract, or fuse
K5 into the M=1 GEMM. Loop every 20m.

## After P0: kernel workstreams (parallelizable)

Pick a directory, one question per run. Details in each README.

- K0 `kernels/roofline/` -- copy + GEMM + GEMV roofs
- K1 `kernels/onednn_isa/` -- dump incumbent Intel ops
- K2 `kernels/esimd_dpas/` -- hand s8 / s4 / s2 DPAS, light INT2
- K3 `kernels/precision_compose/` -- INT2/INT4 as INT8 or E2M1
- K4 `kernels/w8_compare/` -- FP8 W8A16 vs INT8 W8A16 vs W8A8
- K5 `kernels/epilogue_quant/` -- INT8 without ~160 quant launches
- K6 `kernels/nvfp4/` -- every NVFP4 spoof / LUT / split
- K7 `kernels/gdn/` -- GDN hybrid leftover (Qwen3.8 is not plain attn)

Launch pairing: `docs/AGENT_LAUNCH.md`. A literature agent may fetch
`docs/REFERENCES.md` campaign papers with no GPU lease.

Exit per workstream: JOURNAL entry, artifact under results/, promote
to FINDINGS.md only when both cards (or both ranks) support it.
Napkin priors (compose loses, INT2 is useless, ...) are CONFIG, not
RESULT. Measure.

## After a math floor: models

See `docs/MODELS.md`. Do not start a serve to skip K0-K3.

- Dense: Qwen3.8-27B (already on disk: BF16, FP8, W8A8, INT4, NVFP4).
- MoE body: Qwen3.6-35B-A3B when fetched; Ornith-1.5-35B-A3B is the
  on-disk stand-in of the same size class.
- Compact MoE: Gemma 4 26B A4B (fetch when needed).
- Stretch: Qwen3.8-Flash-Next (NVMe expert/PLE, not a 2xB70 resident).

Quants: INT8 W8A16 / W8A8, integer INT4, NVFP4 spoof. FP8 W8A16 is
the dense incumbent control.

## P2: TP=2  (`parallel/tp2/`)

Needs P0. Happier if K0 exists so collective GB/s has a roof.

- Synthetic all-reduce / all-gather / send-recv.
- P2P off first. P2P on only as a labeled control.
- Push all-reduce, fused RMSNorm+AR, and minimum call count are
  first-class arms, not footnotes.
- Tiny sharded matmul only after the synthetic collective is correct.

Exit gate: correctness, matched rank evidence, teardown, post-health.

## P3: PP=2  (`parallel/pp2/`)

- Two-stage synthetic activation handoff, one stage per card.
- Bubble, copy path, stage-memory split.
- Tiny-model PP=2 only after the synthetic handoff is correct.

Exit gate: same as P2.

## P4: mixed 2x2  (`parallel/2x2/`)

Only after P2 and P3 pass on this host with the current UMD.
TP=2 inside a two-stage pipeline, then the reverse, as separate
experiments.

## Standing bans

- Do not skip health to get a speed number.
- Do not reuse quarantined wheels, .so files, or compiler caches from
  b70_ai_things as proof that xe2x2 work is clean.
- Do not promote a serving wrapper from this repo. Hand findings back
  to b70_ai_things.
- Do not mix a slot-move topology A/B into a kernel matrix.
- Do not cite sibling-lab tok/s or ISA notes as FINDINGS until this
  host repeats them.
- Do not treat XeTLA / oneDNN / an Intel paper as a ceiling.
- Rank serving-shaped work by us, not TOPS% or BW%.
- Do not assume TP=2 for every op. Four B70s wait on evidence.
