# K8 -- Nemotron 3.5 Lightning MoE (Mamba-2 + MoE + rare attn)

Question: which Lightning ops actually burn us on these two B70s,
and can the official NVFP4 expert weights ride Xe2 XMX without a
bitcast lie?

Open. This is not Qwen3.8 GDN and not a dense 5120 GEMM. 52-layer
hybrid, ~3B active / 30B total. A s8 GEMM win that leaves Mamba-2
SSU and grouped MoE in eager bf16 can still lose the model.

Do not start a vLLM/sglang serve to answer a kernel question.
Serving wraps stay in b70_ai_things. Sibling tok/s in refs/ are
floors to dump, not FINDINGS.

Checklist: `kernels/nemotron/TASKS.md`.
Campaign map: `docs/KERNEL_CAMPAIGN.md` (K8).
Work order: `RESEARCH_TODO.md` (Lightning section).

## Checkpoint names (do not mix)

| Role | Hugging Face id |
|---|---|
| BF16 reference | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| NVFP4 (PTQ) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| DSpark draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` |
| DFlash draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` |
| Sibling GPTQ-INT4 G64 | `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` |

Architecture `nemotron_h`. Config is the shape source
(`config.json` on the BF16 card). `moe_latent_size` is null:
this is not Super/Ultra LatentMoE unless a later dump says so.

## Layer stack (52)

23 Mamba-2, 23 MoE, 6 attention. Pattern from config
`layers_block_type`. MTP is one extra `attention` then `moe`.

| Field | Value |
|---|---|
| hidden | 2688 |
| vocab | 131072 |
| attn heads / kv / head_dim | 32 / 2 / 128 |
| packed qkv N | 4608 (4096+256+256) |
| o-proj | n=2688 k=4096 |
| routed experts | 128, top-6, n_groups=8, topk_group=1 |
| shared experts | 1, intermediate 3712 |
| routed intermediate | 1856, act relu2 |
| mamba heads / head_dim | 64 / 64 (inner 4096) |
| conv_kernel | 4 |
| ssm_state_size | 128 |
| chunk_size | 128 |
| mamba cache dtype (config) | float32 |
| native context | 262144 (1M claimed) |

KV bf16 is ~6 KiB/token (6 layers * 2 kv * 128 * 2 * 2).
Mamba f32 state is ~46 MiB for 23 layers. Long context is a
Mamba-state question, not a fat flash-attn question.

Official NVFP4 PTQ (model card, not a local RESULT):
W4A16 on routed and shared experts, FP8 per-tensor dynamic
on mamba in_proj/out_proj and KV. Spoof the expert path.
Do not pretend mamba linears are E2M1.

## Why (sibling, reproduce)

refs/intel-arc-pro-b70-inference-cookbook served GPTQ-INT4
Lightning + DFlash on one B70 (~187 tok/s C1, 120K context)
with native grouped-topk and SSU B8/W4 patches. Native MTP
accept was 0% on that stack. That is a community floor.
Reproduce ops here as micros before citing FINDINGS.

## Incumbents to dump (K1 first, then beat in us)

- oneDNN `int8_gemm_w8a8` / `w8a16` at 2688 x 1856 and 2688 x 3712
- oneDNN `nvfp4_gemm_w4a16` on the same expert tiles
- XPU grouped-topk (sibling patch exists)
- XPU Mamba SSU B8/W4
- integer GPTQ s4 grouped MoE (true INT4 XMX control)

## Standing bans for this workstream

- Do not bitcast NVFP4 E2M1 onto s4 DPAS as if it were integer.
  +-12 overflows s4 [-8,7]. That arm is an explicit negative
  unless cosine dies, in which case FINDINGS already holds.
- Do not rerun Qwen GDN mixer leftover tiles and call them
  Mamba-2. SSU is not Gated DeltaNet.
- Do not apply Qwen MTP patches to a Nemotron image, or Nemotron
  grouped-topk / SSU to a Qwen image (sibling landmine).
- Do not enable P2P. Do not start a serve.
- One question per fire. Card0 || card1 on different arms.
- Rank us. TOPS% is diagnostic.

## Record

Op, shape (M,N,K or T,C,state), dtype/spoof name, us, GB/s,
cosine vs BF16 or E2M1 oracle, clocks, backend. Expert count
and top-k in the CONFIG name. Packed E2M1 bytes vs s8 scratch.

## Exit

A map: Lightning op -> roof (launch / BW / XMX) and which NVFP4
spoof is least-bad at decode M=1 vs prefill M=64/256 on the
expert tiles. FINDINGS per closed arm or clean refusal.

Results land in `results/k8/`.
