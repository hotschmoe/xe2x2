# Agent launch brief

Read AGENTS.md, then this, then the one workstream README you own.
P0 is the only hard GPU gate. Docs/KERNEL_CAMPAIGN.md is the map.
Napkin is CONFIG, not RESULT. Intel published != unbeatable.
Latency (us) ranks above TOPS% for serving-shaped work.

## How not to step on each other

- One GPU-touching agent per card, or one agent that owns both cards
  for a two-card job. Never two agents on the same DRM node.
- One-card math: pair as `gpu-run --card 0` || `gpu-run --card 1`.
- A TP=2 / PP=2 / collective agent takes both cards. All one-card
  kernel agents pause.
- Read-only literature / ISA-reading agents do not need the lease.
  Run those in parallel with P0 or K-work.
- Append JOURNAL.md at the bottom. New date letter (2026-09-02e,
  then f, ...). Do not rewrite someone else's entry.
- One question per experiment directory. Do not "also fix serving."
- Preserve a dirty worktree. Do not revert sibling agent files.
- Do not start a vLLM/sglang serve to answer a kernel question.
- Do not skip writing a kernel because XeTLA/oneDNN/a paper exists.
  Dump them, then beat them in us.
- Score TP=1 vs TP=2 per op. Fusion that removes a collective is a
  latency win. Not every op wants TP=2.

## First GPU agents (suggested pairing, not a law)

After P0 health is green:

| Pair | card0 | card1 |
|---|---|---|
| 1 | K0 copy roof | K0 s8 square GEMM |
| 2 | K2 `dpas<s8,s8>` | K2 `dpas<s4,s4>` |
| 3 | K2 `dpas<s2,s8>` mix (see hail mary) | K2 `dpas<s2,s2>` |
| 4 | K1 dump `fp8_gemm_w8a16` | K1 dump `int8_gemm_w8a16` |

Then swap cards so every arm has both-card evidence.

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
