# Agent launch brief

Read AGENTS.md, then this, then the one workstream README you own.
P0 is the only hard GPU gate. Docs/KERNEL_CAMPAIGN.md is the map.
Napkin is CONFIG, not RESULT. Intel published != unbeatable.
Latency (us) ranks above TOPS% for serving-shaped work.

## How not to step on each other

- One GPU-touching agent per card, or one agent that owns both cards
  for a two-card job. Never two agents on the same DRM node.
- Default one-card math: **different arms** on card0 and card1 at
  the same time (`gpu-run --card 0` || `--card 1`). Same binary on
  both cards only when the both-card rule below says so.
- A TP=2 / PP=2 / collective agent takes both cards. All one-card
  kernel agents pause. Only one agent is in charge of that job.
- Compile is CPU docker (`compile_extra.sh`). Two TUs may compile
  in one docker call (args are a list). Do not race one IGC cache
  dir across two compiles of the same stem.
- Read-only literature / ISA-reading agents do not need the lease.
  Run those in parallel with P0 or K-work.
- Append JOURNAL.md at the bottom. New date letter (2026-09-02e,
  then f, ...). Two parallel one-card arms get two letters. Do not
  rewrite someone else's entry.
- One question per experiment directory. Do not "also fix serving."
- Preserve a dirty worktree. Do not revert sibling agent files.
- Do not start a vLLM/sglang serve to answer a kernel question.
- Do not skip writing a kernel because XeTLA/oneDNN/a paper exists.
  Dump them, then beat them in us.
- Score TP=1 vs TP=2 per op. Fusion that removes a collective is a
  latency win. Not every op wants TP=2.

## Both-card rule (when to run the same arm twice)

Held-clock s8 W8A8 tiles on this host have matched within ~1% us
and have the same numeric on both cards. Dual-running every steal
is extra GPU time, not extra silicon.

Run **one card** (held 2800, named clock, `gpu-run --card N`) when:

- The tile family already has both-card held-clock evidence (the
  s8 scale-to-f16 8x2-N / 4x8 family does).
- The question is a schedule steal (WG map, unroll, A-db, k-block).
- Cosine/max_abs close and throttle=0 / clocks held.

Run **both cards** (same binary, card0 || card1) when:

- New dtype, ISA encoding, or numeric contract (s4, s2, NVFP4 LUT,
  first fuse, first producer, first scale-to-f16 of a family).
- First floor we might promote to FINDINGS.md (new us champion).
- Clocks disagreed last time, throttle=1, D3hot, or us spread >5%.
- Numeric not closed (max_abs or cosine off).
- Health, copy roof, or anything that previously disagreed.

Swap the winner of a one-card steal onto the other card later, not
every fire. Promote a new floor to FINDINGS only after the sibling
card has run it once, or after both-card was required above.

Two-card fabric (TP=2 / PP=2 / collectives / P2P): one agent owns
both cards. No sibling one-card GPU agent until that job teardown
and re-health.

## First GPU agents (suggested pairing, not a law)

After P0 health is green, split the **matrix**, do not clone:

| Pair | card0 | card1 |
|---|---|---|
| 1 | K0 copy roof | K0 s8 square GEMM |
| 2 | K2 `dpas<s8,s8>` | K2 `dpas<s4,s4>` |
| 3 | K2 `dpas<s2,s8>` mix (see hail mary) | K2 `dpas<s2,s2>` |
| 4 | K1 dump `fp8_gemm_w8a16` | K1 dump `int8_gemm_w8a16` |

Same-arm both-card only under the both-card rule, then swap.

A literature agent can run at any time: fetch the papers in
`docs/REFERENCES.md` (campaign literature) into notes, no GPU.

## Build landmine (reproduce, do not assume)

intel/llvm issue 21741: ESIMD `xmx::dpas` on B70 was bit-exact in a
standalone binary and wrong when compiled inside a large SYCL
project (~50 TUs), even if the kernel lived in its own .so. First
DPAS micros: tiny standalone `icpx -fsycl` binaries. Promote into
a big tree only after the standalone oracle matches.

## Clocks and TOPS

Datasheet 367 INT8 TOPS assumes a clock. A kernel at 60% of peak
with the card in a 160 W / low-clock state is not a kernel loss.
Record `freq` / power if readable (`xpu-smi` or sysfs), or at least
whether the card was idle-cold vs just after a heavy run. Same for
608 GB/s.

## What not to invent in week one

- A new serving engine.
- A slot move.
- `CCL_TOPO_P2P_ACCESS=1` inside a serve.
- Bitcast NVFP4 E2M1 onto s4 DPAS as if it were integer.
- Calling a sibling-lab tok/s a FINDING.
- Mixing OpenCL and SYCL/L0 in one comparison without labeling it.
