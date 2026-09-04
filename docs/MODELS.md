# Models after the math floor

What to have on disk once K0-K3 (roofline, incumbent ISA, DPAS
micro, precision compose) have a local numeric floor. This is a
shelf for *kernel and 2x2 questions*, not a production catalog.
Serving promotion stays in `b70_ai_things`.

Weights live under the serving tree (`models/files/`) or a path named
in CONFIG. Do not copy 50 GiB checkpoints into xe2x2.

## Quant policy

Focus on XMX-native integer paths, plus NVFP4 as the day-one NVIDIA
dump playground.

| Quant | Role | Why |
|---|---|---|
| INT8 W8A16 | dense decode default to beat | Native XMX, no activation-quant launches. The honest rival to Steve FP8 W8A16. |
| INT8 W8A8 | prefill / fused-epilogue | Native s8x s8 DPAS. Only after K5 (quant launches) has an answer, or as a GEMM-only control. |
| INT4 GPTQ/AWQ/RTN | MoE default, dense capacity | Integer s4 is what ESIMD `dpas<s4,s4>` wants. Grouped expert GEMM is the Intel MoE story. |
| NVFP4 | crazy / day-one | Exporters target NVIDIA. Spoof, LUT, 4-bit resident, compose. Never bitcast. |
| FP8 W8A16 | incumbent control | Steve's well-optimized dense path. Baseline, not the kernel research target. |
| MXFP4 / GGUF | labeled controls | MXFP4 is OCP, not NVFP4. GGUF is llama.cpp SYCL, useful for Gemma and Flash-Next. |

Do not skip an INT4 or INT2 microbench because "MoE will use INT4
later." Micro first, then a real checkpoint.

## Fit on 2x 30.3 GiB (priors, measure at load)

Two cards, no XeLink, Gen3 x16 each. KV and GDN state eat the rest.
Numbers below are weight-disk priors from this host or sibling labs,
not a KV-capacity FINDING.

## Dense primary -- Qwen3.8-27B

The model we actually want to serve. Already on disk
(`b70_ai_things/models/files/qwen3.8-27b/`):

| Local tree | Disk | Notes |
|---|---|---|
| `bf16/` | 52G | correctness oracle |
| `fp8-official/` | 29G | Steve W8A16 incumbent |
| `w8a8-gptq/` | 35G | INT8 XMX, tight on one card, TP=2/PP=2 customer |
| `gptq-int4-mtp-bf16-9d189a60/` | 19G | integer INT4 + MTP |
| `nvfp4-radixark/` | 21G | K6 customer |

Use this family for: K4 W8 A/B, K5 epilogue-quant in a real layer
shape, K6 NVFP4 spoof vs resident 4-bit, TP=2 decode once collectives
are healthy. One-card INT4/NVFP4/FP8; two-card INT8 and long KV.

## MoE primary -- 30B/35B-A3B class (~3B active)

Two named bodies, same size class. Kernel work is K8 for Lightning
(Mamba-2 + MoE + rare attn) and grouped GEMM / EP vs PP for both.

### Nemotron 3.5 Lightning 30B-A3B

The NVFP4 MoE we actually want to trick onto Xe2. Hugging Face:
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` and
`...-NVFP4`. Hybrid 52-layer `nemotron_h`: 23 Mamba-2, 23 MoE
(128 routed top-6 + 1 shared, 1856 / 3712), 6 GQA (32/2, d=128),
hidden 2688. Official PTQ: W4A16 experts, FP8 mamba in/out + KV.
Kernel brief `kernels/nemotron/README.md`. Do not copy 66G BF16
into xe2x2; fetch NVFP4 or sibling GPTQ-INT4 G64 when a real
tensor is required. Sibling B70 already served GPTQ-INT4+DFlash
(refs/ cookbook, ~187 tok/s C1) -- dump, do not cite as FINDINGS.

Lightning makes **expert-parallel vs PP=2** a current 2-card
question (NVIDIA trains EP=8 TP=1). Mamba state wants replicate.
Do not start a serve in this repo.

### Qwen3.6-35B-A3B / Ornith stand-in

Named target for grouped GEMM, expert routing, and TP vs PP vs
batching on a dense-attn MoE (not Mamba). Sibling labs served
GPTQ-INT4 and Quark W8A8. Qwen3.6 is not in the current
`models/files/` live set; fetch when that name is required.
Until then, **Ornith-1.5-35B-A3B is already on disk** (BF16, GPTQ
INT4, NVFP4, W8A8 RTN). Use Ornith as the on-hand MoE body; do
not pretend it is Qwen3.6 or Lightning when reporting.

MoE is where PP=2 and batching get interesting: fewer cross-card
bytes than per-layer TP allreduce, stages stay busy at c>1, expert
placement per stage is an open map. TP=2 still shards wide
projections and can win decode if AR is fused (Steve). Measure both.

## MoE compact -- Gemma 4 26B A4B

Fits one B70 at Q8 in Steve's llama.cpp SYCL packet (~122 tok/s
community; reproduce before FINDINGS). Not on this host today. Fetch
when we want a MoE that does not need two cards for capacity, so
TP=1 vs TP=2 vs PP=2 is a speed question rather than a fit question.
INT8/Q8 here is XMX-native; INT4 is the capacity/speed A/B.

## Stretch -- Qwen3.8-Flash-Next

Too big to treat as a 2xB70 resident serve. Sibling recipe: 512
experts, ~65G q4 expert pool + ~40G PLE mmap'd from NVMe, hot experts
in VRAM, llama.cpp SYCL + ESIMD MoE, vLLM XPU unsupported. Keep it
as a kernel and tiering object (expert GEMM, NVMe cold path, PP=2
stage placement), not as P0-P4 serving. Do not block the 27B or
35B-A3B shelf on Flash-Next.

## How a math kernel graduates to a model

1. Numeric microbench on synthetic tiles (K0-K3).
2. Real-shape GEMM on one projection from the 27B (K4/K6).
3. One layer or one expert, still not a server.
4. Tiny forward (one request, greedy, health around it) through
   `gpu-run`. Hand the wrap back to `b70_ai_things` if it wants a
   shelf entry.

Skip steps only when the question is already a fabric question
(TP=2/PP=2) and the kernel floor exists.

## TP=1 vs TP=2 vs PP=2 vs batch (open)

Per-op, not per-model. A projection may want TP=2; a GDN recurrent
update or a tiny epilogue may want TP=1 because the allreduce is
longer than the math. Joining two linears (or linear+RMSNorm) so
one collective remains is the TP=2 reason to write mega-kernels.
Score us and collectives/token. Peak fabric GB/s is secondary.

Priors, all unmeasured in xe2x2:

- TP=2 can win single-stream decode when allreduce is fused / few
  calls (Steve dense FP8). This PCI tree is still cross-die Gen3.
- PP=2 moves one push per microbatch instead of tens of allreduces.
  Sibling 27B-W8A8 PP=2 won TTFT (~4.7x) and held c1-c8 aggregate;
  decode was gated by eager and no MTP. Graph+MTP PP is still open.
- MoE + continuous batch is the PP fill: both stages busy, comm is
  one activation handoff, experts can live on one stage.
- Do not pick a religion. Same model, same quant, TP=2 vs PP=2 vs
  TP=1 two-replica, at c1 and at c>1, with health.

## Dual-card

One-card model probes (INT4 27B, Gemma if fetched) still run
card0 || card1 as two independent loads only if VRAM and lease
allow; two full 27B loads will not. Default: one model job uses
the map under test (1 card or 2). Keep the kernel matrix two-wide
when the model job is not running.
