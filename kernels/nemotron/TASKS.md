# K8 tasklist -- Lightning ops and NVFP4 spoofs

One checkbox is one fire (or a documented refusal). Split cards.
Mark STOP in the line when a mapping loses. Do not rerun STOP
tiles. Napkin is CONFIG, not RESULT.

Held K6 priors to re-test on Lightning expert shapes, not to skip:
NVFP4 can feed XMX via nibble->s8 (slow). Not as fast as s8 with
s8-A. Two-term s4 compose can beat s8 at decode 5120, not at
FFN-down or large-M vs W8A8. Never bitcast. iselect / product-LUT
GEMV already lost on 5120.

## 0. Inventory (no serve)

- [x] Dump one BF16 layer: op names, dtypes, MNK, bytes of
      mamba state vs expert weights vs KV at T=1 and T=256.
      CONFIG napkin in results/k8/inventory.md. No ckpt on host.
- [ ] Dump the NVFP4 checkpoint layout: which tensors are
      E2M1+FP8-scale, which are FP8, which stay BF16.
      BLOCKED: no NVFP4 tree on disk.
- [ ] Histogram E2M1 codes on routed experts vs shared expert
      (overflow rate of codes 8 and 12). Per-expert if cheap.
      BLOCKED: no NVFP4 tree on disk. Synthetic E2M1 OK for spoofs.
- [x] Confirm `moe_latent_size` is null at published config
      (no d->l). Runtime dump still open if a later ckpt appears.

## 1. Mamba-2 (new math, not GDN)

- [ ] SSU decode T=1, 64 heads x 64 dim, d_state=128, vs eager
      and vs sibling SSU B8/W4. Rank pipe_host.
- [ ] conv1d K=4 on mamba channels (C~4096, not GDN 10240).
- [ ] Fuse conv+SSU T=1 vs sequential.
- [ ] SSD / chunked prefill chunk=128, T=256 vs sequential
      conv+SSU. Combined wall time, not additive napkin.
- [ ] in_proj GEMM M=1 and M=64, k=2688, fat N (z/x/B/C/dt).
      s8 / W8A8 / FP8-W8A16. Official PTQ is FP8 here.
- [ ] out_proj GEMM n=2688. Same dtypes.
- [ ] State bytes and us at T=1 vs T=256 vs T=8192 (no 1M
      until a kernel floor exists).

STOP if a mapping is GDN leftover by another name.

## 2. MoE grouped GEMM (the compute body)

Shapes: routed 2688 x 1856 up, 1856 x 2688 down, relu2.
Shared 2688 x 3712, every token. top-6 of 128.

- [ ] Router 2688 -> 128, sigmoid, grouped top-6, expert bias.
      vs native XPU grouped-topk.
- [ ] Decode M=1, 6 routed experts, s8 vs W8A8 vs oneDNN
      nvfp4_gemm_w4a16 vs GPTQ s4. Launch tax named.
- [ ] Shared expert M=1, 2688 x 3712, same dtypes.
- [ ] Prefill M=64 grouped (not 128 launches).
- [ ] Prefill M=256 grouped.
- [ ] Packing: fused grouped DPAS vs gather-scatter into a
      dense 6-row tile vs persistent expert B.
- [ ] relu2 epilogue fused vs extra launch (K5).
- [ ] Integer GPTQ-s4 grouped as the true INT4 XMX control
      (sibling G64 checkpoint, not NVFP4).

## 3. Attention (6 layers only -- cheap, still measure)

- [ ] Packed qkv M=1 n=4608 k=2688 s8 vs W8A8.
- [ ] Packed qkv M=64 and M=256.
- [ ] o-proj M=1 n=2688 k=4096.
- [ ] GQA 32/2 decode attn vs Mamba SSU us (expect attn << SSU
      at long T; measure).

Do not start a flash-attn campaign. Six layers, 2 KV heads.

## 4. MTP head

- [ ] Extra attn+moe at M=2, M=4, M=6 (spec widths).
- [ ] lm_head 2688 x 131072 M=1 and MTP-M. Wide-N leftover.
- Native MTP accept 0% is a sibling prior, not a kernel skip.

## 5. NVFP4 spoofs on Lightning experts (whacky allowed)

Label every arm with the spoof name. Numeric vs E2M1 dequant
oracle. Packed nibbles stay in HBM unless the arm is load-time
repack. Both-card on first numeric of a new spoof.

### 5a. Floors (dump, then beat)

- [ ] oneDNN `nvfp4_gemm_w4a16` M=1 2688 x 1856 (decode expert).
- [ ] same, 1856 x 2688 (down).
- [ ] same, 2688 x 3712 (shared).
- [ ] same at M=64 and M=256.
- [ ] Load-time s8 LUT of all 128+1 experts. Fits 30.3 GiB?
      Bytes vs resident NVFP4. If it does not fit, FINDINGS.

### 5b. Held K6 spoofs, Lightning shapes (do not skip)

- [ ] Nibble LUT -> s8 DPAS (merge) M=1 expert up/down.
- [ ] Closed-form nibble->s8 (exp/mant) M=1 expert.
- [ ] Two-term s4 compose `w_lo + 8*w_hi` A=s4 M=1 expert down.
- [ ] Two-term at M=64 and M=256 expert (prior: loses large-M).
- [ ] Dyadic s2/s4 planes {0.5,1,2,4} plus residual {1.5,3,6}.
- [ ] Mixed s8-A x E2M1-as-s8-B (LUT) vs W8A8.
- [ ] GPTQ s4 on the same MNK as control.

STOP if us > 4x the W8A8 floor with no new numeric trick.

### 5c. Whacky / hail-mary (label hail-mary in CONFIG)

- [ ] Bitcast E2M1 onto s4 DPAS. Expect cosine death. Keep as
      explicit negative on this histogram.
- [ ] Lo-only s4, drop codes 8 and 12. Live only if overflow
      histogram is tiny on Lightning experts (Qwen FFN was ~25%,
      DEAD). Per-expert lo-only if the tail is a few experts.
- [ ] Dual-path: DPAS on in-range codes, scalar/XVE fixup on
      8 and 12. Count the fixup density.
- [ ] u4 magnitude + sign plane, then sparse 8/12 correction.
- [ ] Shared expert load-time s8 (every token, worth VRAM),
      routed experts stay packed E2M1 + LUT (capacity).
- [ ] Dequant-to-bf16 GEMV at M=1 1856. Tiny-N may beat LUT+XMX.
- [ ] 16-code product LUT GEMV on expert down (lost at 5120;
      maybe wins at 1856). Kill or keep with us.
- [ ] iselect 16-entry table (lost 6x at 5120). One fire at
      1856 then STOP if still a loss.
- [ ] W4A8 QServe-style: NVFP4 B, s8 A, dequant in the XMX
      pipe, static A scales. Protective range.
- [ ] Karatsuba two-term vs schoolbook two-term on 1856.
- [ ] Group-16 NVFP4 scale landmine (`g16_scale_landmine.py`
      pattern) on a real Lightning tensor, not synthetic.
- [ ] Unpack E2M1 to a GPTQ-looking s4 packing (wrong map,
      measure cosine). If it accidentally closes, FINDINGS.
- [ ] MXFP4 e8m0 group-32 as a labeled third format (0 layers
      expected on this NVFP4 card).
- [ ] Persist selected-6 expert tiles in L2, not SLM (1856*2688
      nibble ~2.5 MiB/expert, SLM will refuse).
- [ ] Prefetch next expert B while current DPAS (grouped).
- [ ] s2xs8 mix on a dyadic-only subset of codes.
- [ ] FP8 mamba in/out as W8A16 incumbent, not an E2M1 spoof.
      Separate CONFIG.
- [ ] KV FP8 (official PTQ) vs bf16 vs s8. 6 KiB/token class.
- [ ] Expert-wise scale fuse: apply NVFP4 block scale in the
      DPAS epilogue with the existing 0.02-style f16 scale path.

### 5d. Fit and bytes

- [ ] Resident NVFP4 expert pool bytes vs s8 repack vs GPTQ s4
      vs two-term s4 planes, one card 30.3 GiB, KV at 32K and
      120K (sibling capacity prior).
- [ ] Decode GB/s of packed E2M1 vs compute us. M=1 should be
      bytes; if LUT tax >> HBM, FINDINGS (K6 already saw this
      at 5120).

## 6. Fabric (pause one-card matrix)

- [ ] TP=1 vs TP=2 on hidden 2688 attn/mamba proj. Prior: AR
      99-137 us may exceed the GEMM.
- [ ] Mamba state: replicate (TP=1), do not shard.
- [ ] EP=2: experts 0-63 vs 64-127, top-6 may miss a card.
      Identity + us vs replicate-all.
- [ ] PP=2: experts pinned per stage, one activation handoff.
      Bubble vs decode 77 us host-staged prior.
- [ ] Mixed 2x2 decode on Lightning hidden 2688 (not 5120).
- One-shot XCCL AG >=2.5 MiB still hangs. Use chunked 64h or
  sendrecv 2641 us. P2P off.

## First two-wide pair (after inventory)

| card0 | card1 |
|---|---|
| Mamba-2 SSU T=1 vs eager | MoE grouped s8 M=1 6-of-128, 2688 x 1856 vs W8A8 |

Then swap. Then NVFP4 w4a16 dump vs nibble-LUT on the same
expert tile (both-card: new numeric).

## Parked (do not start here)

Flash-attn sweep. DFlash/DSpark CUDA drafts as xe2x2 kernels.
Qwen packed-qkv 10240 x 5120 leftover. GDN mixer-slmhtc / pipe.
A Lightning serve in this repo.
