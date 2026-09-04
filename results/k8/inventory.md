# K8 inventory -- Nemotron 3.5 Lightning (CONFIG until a checkpoint is local)

Date 2026-09-04. No Lightning BF16/NVFP4/GPTQ tree on this host
(`models/files/` has ornith, qwen3.6-27b, qwen3.8-27b). Histogram
and NVFP4 tensor layout stay blocked. Shapes from published
`config.json` on nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.
Napkin bytes are CONFIG, not RESULT.

## Architecture

model_type nemotron_h. 52 layers: 23 mamba, 23 moe, 6 attention.
MTP: 1 extra attention+moe. moe_latent_size=null (no d->l).
hidden=2688 vocab=131072.

## One-layer ops (decode T=1)

| op | dtype (official PTQ) | MNK / state | notes |
|---|---|---|---|
| mamba in_proj | FP8 W8A16 | M=1 k=2688 N~10304 if ngroups=8 | z/x + B/C + dt. Confirm N at runtime. |
| mamba conv1d | f16/f32 | T=1 C=4096 K=4 | same FIR length as GDN, different C |
| mamba SSU | f32 state | 64 heads, d_head=64, d_state=128 | not GDN delta |
| mamba out_proj | FP8 W8A16 | M=1 n=2688 k=4096 | |
| moe router | bf16/fp32 | 2688 -> 128, top-6, n_groups=8 | grouped_topk |
| moe routed up | NVFP4 W4A16 | M=1 n=1856 k=2688 x 6 experts | relu2 |
| moe routed down | NVFP4 W4A16 | M=1 n=2688 k=1856 x 6 | |
| moe shared | NVFP4 W4A16 | M=1 n=3712 k=2688 (and down) | every token |
| attn qkv packed | (not expert NVFP4) | M=1 n=4608 k=2688 | 6 layers only |
| attn o | | M=1 n=2688 k=4096 | |
| lm_head | | M=1 n=131072 k=2688 | wide |

## Bytes (CONFIG napkin, bf16 unless noted)

Routed experts 23*128*(up+down)*2688*1856*2 ~ 27.4 GiB.
Shared 23*(up+down)*2688*3712*2 ~ 0.92 GiB.
NVFP4 4-bit resident ~ half: experts ~14.2 GiB. Fits one 30.3 GiB
card with KV. s8 load-time LUT of all routed ~27.4 GiB: likely
does not fit with runtime+KV. Measure when a checkpoint lands.

KV bf16 ~6 KiB/token: 32K ~192 MiB, 120K ~720 MiB, 256K ~1.5 GiB.
Mamba f32 state ~23*64*64*128*4 ~ 46 MiB, independent of T.

## Blocked

- NVFP4 safetensor dtype map (which keys are E2M1+scale vs FP8)
- E2M1 code histogram / overflow 8 and 12 on real experts
- Runtime confirmation of mamba n_groups and in_proj N

Synthetic E2M1 with a documented histogram is allowed for spoofs
until the NVFP4 card is fetched (K6 README).
