# K8 tasklist -- Lightning ops and NVFP4 spoofs

One checkbox is one fire (or a documented refusal). Split cards.
Mark STOP in the line when a mapping loses. Do not rerun STOP
tiles. Napkin is CONFIG, not RESULT.

Held K6 priors to re-test on Lightning expert shapes, not to skip:
NVFP4 can feed XMX via nibble->s8 (slow). Not as fast as s8 with
s8-A. Two-term s4 compose can beat s8 at decode 5120, not at
FFN-down or large-M vs W8A8. Lightning expert-up M=1 U=14 is
15.518 us (04ao), 16-class vs s8 16.060. Never bitcast.
iselect / product-LUT GEMV already lost on 5120
and at Lightning 1856 (STOP).

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

- [x] SSU decode T=1, 64 heads x 64 dim, d_state=128.
      card0 80.064 us pipe_host at 2800 (04ah). spin=0 was
      190 us ramp, do not freeze 190. Sibling card1 in flight.
- [x] conv1d K=4 on mamba channels C=4096. card1 4.355 us at
      2800 (04ai), wash vs C=2048 4.4.
- [x] Fuse conv+SSU T=1 vs sequential.
      card0 81.075 us pipe_host at 2800
      (04av) vs 84.419 napkin (~0.96x).
      Two in-order launches. One-card
      first; sibling not yet.
- [x] SSD / chunked prefill chunk=128, T=256 vs sequential
      conv+SSU. Combined wall time, not additive napkin.
      card0 21114.404 us pipe_host
      (04az) vs 256*80.064=20496
      (~1.03x). Serial T, not a
      beat. One-card first;
      sibling not yet.
- [x] in_proj GEMM M=1 s8 n=10304 k=2688.
      s8 U=14 23.504 us pipe_host
      card0 at 2800 (04bl). Left
      16-class vs packed qkv
      16.609 (~1.42x) / expert
      16.060 (~1.46x). Not
      napkin N-linear ~89.
      Official PTQ is FP8; s8
      is the beat-me control.
- [x] in_proj GEMM M=64 s8. Same N.
      s8 U=14 211.257 us
      pipe_host card0 (04ca).
      throttle=1 timed
      act=2633. Do not
      freeze as 2800. vs M=1
      23.504 (~8.99x), not
      M-linear. vs qkv M=64
      106.287 (~1.99x) vs
      expert M=64 31.198
      (~6.77x). Fat N=10304
      is 1.22x of N-linear
      173; 0.89x of qkv
      N-linear 238. W8A8
      M=64 still open. One-
      card first; sibling
      later (throttle=1).
- [x] in_proj GEMM M=1 FP8-W8A16
      n=10304 k=2688. 51.036 us
      card1 (04bm). Loses to s8
      23.504 (~2.17x). Fat N
      left 44-class (~1.15x of
      W8A8 44.285). 543 GB/s.
      Official PTQ is FP8.
- [x] in_proj GEMM M=1 W8A8
      n=10304 k=2688. 41.159 us
      card1 (04cb). 41-class
      like qkv 41.320, not
      N-linear ~246. Beats
      FP8 51.036 (~0.81x).
      Loses to s8 23.504
      (~1.75x). 673 GB/s.
      Official PTQ is FP8.
- [ ] in_proj GEMM M=64 W8A8.
- [ ] in_proj GEMM M=64 FP8-W8A16.
- [x] out_proj GEMM n=2688. Same dtypes.
      W8A8 M=1 n=2688 k=4096 is
      44.081 us card1 (04bk),
      same MNK as attn o-proj.
      44-class launch like
      expert-up 44.285.
      FP8-W8A16 43.694 us
      card1 (04bo), wash vs
      W8A8 (~0.99x). Loses
      to s8 o-proj 23.115
      (~1.89x). 252 GB/s.
      s8 still open.
      Official PTQ is FP8.
- [ ] State bytes and us at T=1 vs T=256 vs T=8192 (no 1M
      until a kernel floor exists).

STOP if a mapping is GDN leftover by another name.

## 2. MoE grouped GEMM (the compute body)

Shapes: routed 2688 x 1856 up, 1856 x 2688 down, relu2.
Shared 2688 x 3712, every token. top-6 of 128.

- [x] Router 2688 -> 128, sigmoid, grouped top-6, expert bias.
      vs native XPU grouped-topk.
      global_top6 27.333 us pipe_host
      card1 at 2800 (04aw). Same
      top-6 set as host, cosine=1.
      Grouped topk_group=1 later.
- [x] Decode M=1, 6 routed experts s8 vs W8A8 (one expert).
      s8 U=14 one-expert 16.060 us beats W8A8 44.285 (launch
      class, not N-linear). Grouped-6 UP 165.223 us card1
      (04ag) beats 6x W8A8 ~266, ~1.72x vs 6x16 napkin
      (k64 loop not U=14). Grouped-6 DOWN 120.502 us
      pipe_host card0 at 2800 (04at), K-linear vs UP
      165 (~1.06x of 114 napkin), ~1.25x vs 6x16.
      nvfp4_gemm_w4a16 39.255 us card1
      (04as), 37-class launch. GPTQ s4
      U=14 16.224 us card1 (04ba),
      16-class vs s8 16.060.
- [x] Shared expert M=1, 2688 x 3712, same dtypes.
      W8A8 42.273 us card1 (04am), 44-class
      launch like routed-up 44.285 not N-linear.
      s8 U=14 16.541 us pipe_host card0 at
      2800 (04ar). Launch class vs expert-up
      16.060 / packed qkv 16.609, not napkin
      32. Beats W8A8 42.273. nvfp4 / GPTQ
      still open.
- [x] Prefill M=64 grouped (not 128 launches).
      grouped-6 UP 231.179 us
      pipe_host card0 (04bs) vs
      M=1 grouped 165.223 (~1.40x)
      vs 6*31.198=187 napkin
      (~1.23x) vs 6*W8A8 M=64
      6*39.907=239 (~0.97x).
      k64 loop not U=14. mean
      38.4 us/expert vs U=14
      31.198. W8A8 one-expert
      M=64 39.907 card1 (04bt),
      40-class like nvfp4 40.184.
      throttle=1 timed act=2783.
      One-card first; sibling
      not yet. Not 64 one-expert
      launches.
- [x] Prefill M=256 grouped.
      grouped-6 UP 654.497 us
      pipe_host card0 (04by) vs
      M=1 grouped 165.223 (~3.96x)
      vs M=64 grouped 231.179
      (~2.83x) vs 6*31.198*4=749
      napkin (~0.874x) vs
      6*W8A8 M=64 M-linear
      6*39.907*4=958 (~0.683x).
      k64 loop not U=14. mean
      107.1 us/expert vs U=14
      M=64 31.198. throttle=1
      timed act=2683. Do not
      freeze as 2800. One-card
      first; sibling not yet.
      Not 256 one-expert
      launches.
- [ ] Packing: fused grouped DPAS vs gather-scatter into a
      dense 6-row tile vs persistent expert B.
- [ ] relu2 epilogue fused vs extra launch (K5).
- [ ] Integer GPTQ-s4 grouped as the true INT4 XMX control
      (sibling G64 checkpoint, not NVFP4).

## 3. Attention (6 layers only -- cheap, still measure)

- [x] Packed qkv M=1 n=4608 k=2688 s8 vs W8A8.
      s8 U=14 16.609 us pipe_host
      card0 at 2800 (04al). W8A8
      41.320 us card1 (04aq).
      Launch class vs expert-up
      16.060 / 44.285, not
      napkin 40 / 110.
- [x] Packed qkv M=64.
      s8 U=14 106.287 us
      pipe_host card0 at 2800
      (04bj). Left launch-class
      vs M=1 16.609 (~6.40x),
      not M-linear. vs expert
      M=64 31.198 (~3.41x) vs
      two-term 29.637 (~3.59x).
      Fat N=4608 is 1.37x of
      N-linear 77.5. W8A8 M=64
      still open. One-card
      first; sibling not yet.
- [x] Packed qkv M=256.
      s8 U=14 303.121 us
      pipe_host card0 (04bw).
      throttle=1 timed
      act=2600->2583. Do not
      freeze as 2800. vs M=1
      16.609 (~18.3x) vs M=64
      106.287 (~2.85x), not
      M-linear. Fat N MN-linear
      from expert M=64 ~310
      (~0.978x). W8A8 M=256
      still open. One-card
      first; sibling later
      (throttle=1).
- [x] o-proj M=1 n=2688 k=4096.
      s8 stock U=16 23.115 us
      pipe_host card0 at 2800
      (04ap). Tracks 4 K-blocks
      vs packed qkv 16.609, not
      launch-class 16. Beats
      Qwen NT1 SK 44 / W8A8 47
      and qkv W8A8 41.320.
      W8A8 44.081 us card1
      (04bk), 44-class launch
      like expert-up 44.285 /
      qkv 41.320, not K-linear.
      Loses to s8 23.115
      (~1.91x). Same MNK as
      mamba out_proj.
- [x] GQA 32/2 decode attn vs Mamba SSU us (expect attn << SSU
      at long T; measure).
      T=1 card0 pytorch SDPA
      54.944 us (04bn) vs SSU
      80.064 (~0.69x). Faster
      not <<. T=256 card0
      48.560 us (04bp) vs SSU
      80.064 (~0.61x) vs T=1
      54.944 (~0.88x) vs SSD
      21114 (~0.0023x, 435x).
      Not << vs decode SSU;
      IS << vs SSD prefill.
      Not T-linear. Clocks
      not held. Not flash-
      attn. Six layers.

Do not start a flash-attn campaign. Six layers, 2 KV heads.

## 4. MTP head

- [ ] Extra attn+moe at M=2, M=4, M=6 (spec widths).
- [x] lm_head 2688 x 131072 M=1.
      s8 U=14 1012.237 us
      pipe_host card0 at 2800
      (04br). Left 16-class vs
      packed qkv 16.609
      (~60.9x) / expert 16.060
      (~63.0x). Tracks napkin
      N-linear
      16.060*(131072/1856)~1134
      (~0.89x). Wide-N leftover
      IS N-linear, not launch-
      class. Host cosine closed.
      One-card first.
- [ ] lm_head MTP-M. Wide-N
      leftover M=1 closed
      1012.237 (04br).
- Native MTP accept 0% is a sibling prior, not a kernel skip.

## 5. NVFP4 spoofs on Lightning experts (whacky allowed)

Label every arm with the spoof name. Numeric vs E2M1 dequant
oracle. Packed nibbles stay in HBM unless the arm is load-time
repack. Both-card on first numeric of a new spoof.

### 5a. Floors (dump, then beat)

- [x] oneDNN `nvfp4_gemm_w4a16` M=1 2688 x 1856 (decode expert).
      39.255 us card1 (04as), 37-class
      launch like square 5120 37, not
      N-linear ~13. Beats W8A8 44.285,
      loses to s8 16.060 / two-term
      15.518, beats LUT 83.659. Clocks
      not held. No E2M1 cosine.
- [x] same, 1856 x 2688 (down).
      39.224 us card0 (04be), 37-class
      launch like UP 39.255, not
      N-linear. Beats W8A8 44.285,
      wash vs LUT down 38.351, loses
      to two-term down 11.008 / s8
      16.060. Clocks not held. No
      E2M1 cosine.
- [x] same, 2688 x 3712 (shared).
      38.266 us card1 (04bi), 37-class
      launch like UP 39.255 / DOWN
      39.224, not N-linear ~78. Beats
      shared W8A8 42.273, loses to
      shared s8 16.541. Clocks not
      held. No E2M1 cosine.
- [x] same at M=64.
      40.184 us card1 (04bq), 40-class
      launch like M=1 39.255 (~1.02x),
      not M-linear. Loses to s8 M=64
      31.198 (~1.29x) / two-term
      29.637 (~1.36x). Beats W8A8
      M=1 44.285 (~0.91x); wash vs
      matched W8A8 M=64 39.907
      (04bt). Clocks not held. No
      E2M1 cosine.
- [x] same at M=256.
      57.140 us card1 (04bz), 57-class
      like M=1 39.255 (~1.46x) / M=64
      40.184 (~1.42x), not M-linear.
      Beats two-term M=256 100.811
      (~0.567x); loses to W8A8 M=64
      39.907 (~1.43x). W8A8 M=256
      still open. Clocks not held.
      No E2M1 cosine.
- [ ] Load-time s8 LUT of all 128+1 experts. Fits 30.3 GiB?
      Bytes vs resident NVFP4. If it does not fit, FINDINGS.

### 5b. Held K6 spoofs, Lightning shapes (do not skip)

- [x] Nibble LUT -> s8 DPAS (merge) M=1 expert up.
      Stock U=16 REFUSED 04ak. U=14 83.659 us
      pipe_host card0 at 2800 (04an). Loses to
      s8 16.060 (~5.21x) and W8A8 44.285 (~1.89x).
      K-linear vs 5120 158, not launch-class.
- [x] Nibble LUT -> s8 DPAS (merge) M=1 expert down.
      Stock U=14/16 cannot divide k=1856.
      k64 38.351 us pipe_host card0 at 2800
      (04bb). Beats LUT up 83.659 (~0.46x)
      and W8A8 44.285 (~0.87x). Loses to s8
      16.060 (~2.39x). Never bitcast.
- [x] Closed-form nibble->s8 (exp/mant) M=1 expert.
      U=14 71.715 us pipe_host
      card0 at 2800 (04ax). Beats
      merge LUT 83.659 (~0.857x).
      Loses to s8 16.060 (~4.47x),
      two-term 15.518 (~4.62x),
      W8A8 44.285 (~1.62x).
      K-linear vs 5120 134.8, not
      launch-class. Never bitcast.
- [x] Two-term s4 compose `w_lo + 8*w_hi` A=s4 M=1 expert up.
      U=14 15.518 us pipe_host card1 at
      2800 (04ao). Launch class vs s8
      16.060, beats W8A8 44.285. Never
      bitcast.
- [x] Two-term s4 compose `w_lo + 8*w_hi` A=s4 M=1 expert down.
      k64 11.008 us pipe_host card1 at
      2800 (04bc). Beats two-term UP
      15.518 (~0.71x), K-linear not
      launch-class 16. Beats grouped
      down s8/6 20.084. Beats GPTQ
      16.224 / W8A8 44.285. Never
      bitcast.
- [x] Two-term at M=64 expert (prior: loses large-M at 5120).
      U=14 29.637 us pipe_host card0 at
      2800 (04bh). Beats s8 M=64 31.198
      (~0.95x) and W8A8 M=1 44.285
      (~0.67x). Left launch-class vs
      M=1 15.518 (~1.91x). 8x2-N does
      NOT lose vs W8A8 at this shape
      (unlike 5120 217.92). Never
      bitcast.
- [x] Two-term at M=256 expert (prior: loses large-M).
      U=14 100.811 us pipe_host card1 at
      2800 (04bx). Loses to M=64 two-term
      29.637 (~3.40x) and s8 M=64 31.198
      (~3.23x) and W8A8 M=1 44.285
      (~2.28x) / M=64 39.907 (~2.53x).
      Left launch-class vs M=1 15.518
      (~6.50x). 8x2-N DOES lose vs
      W8A8 at M=256 (5120 prior holds
      here; unlike M=64). Never
      bitcast.
- [x] Dyadic s2/s4 planes {0.5,1,2,4} plus residual {1.5,3,6}.
      hail-mary s2xs2 one-plane
      padM=8 event 41.615 us
      card1 (04bu). cosine=1
      max_abs=0 ok=1 (s2xs2,
      not E2M1). M=1 RC=8
      unaligned. 4-plane napkin
      166.460 CONFIG, not a
      fused RESULT. Loses to
      two-term 15.518 (~2.68x
      / ~10.7x). Beats W8A8
      44.285 one-plane (~0.94x).
      s2xs4 COMPILE_REFUSED
      (K6). residual pairwise.
      No STOP (166 < 177).
      Never a floor.
- [x] Mixed s8-A x s4-B integer control
      M=1 expert up. U=14 13.458 us
      pipe_host card0 at 2800 (04bf).
      Beats two-term 15.518 / s8
      16.060 / GPTQ 16.224 / W8A8
      44.285. 13.5-class, below
      the 16-class launch floor.
      Not E2M1-as-s8 LUT (that is
      nibble_lut 83.659 already).
      Never bitcast. Sibling not
      yet.
- [x] GPTQ s4 on the same MNK as control.
      U=14 16.224 us pipe_host
      card1 at 2800 (04ba).
      Launch class vs s8 16.060
      / two-term 15.518. Beats
      nvfp4 39.255 / W8A8 44.285.
      Synthetic g128, no ckpt.

STOP if us > 4x the W8A8 floor with no new numeric trick.

### 5c. Whacky / hail-mary (label hail-mary in CONFIG)

- [x] Bitcast E2M1 onto s4 DPAS. Expect cosine death. Keep as
      explicit negative on this histogram.
      hail-mary padM=8 n=1856 k=2688
      cosine=0.663320 max_abs=14784
      ok=0 card1 (04bg). M=1 RC=8
      unaligned. event 45.104 us
      clocks not held. vs two-term
      15.518 / s8 16.060. Never a
      floor.
- [ ] Lo-only s4, drop codes 8 and 12. Live only if overflow
      histogram is tiny on Lightning experts (Qwen FFN was ~25%,
      DEAD). Per-expert lo-only if the tail is a few experts.
- [ ] Dual-path: DPAS on in-range codes, scalar/XVE fixup on
      8 and 12. Count the fixup density.
- [ ] u4 magnitude + sign plane, then sparse 8/12 correction.
- [ ] Shared expert load-time s8 (every token, worth VRAM),
      routed experts stay packed E2M1 + LUT (capacity).
- [x] Dequant-to-bf16 GEMV at M=1 1856. Tiny-N may beat LUT+XMX.
      hail-mary 26.962 us pipe_host card1 at
      2800 (04ay). Beats LUT 83.659 (~0.32x),
      loses to s8 16.060 (~1.68x), beats W8A8
      44.285 (~0.61x). 27-class like router
      27.333, not LUT 84 / XMX 16. No STOP
      (us << 4x W8A8 177).
- [x] 16-code product LUT GEMV on expert down (lost at 5120;
      maybe wins at 1856). Kill or keep with us.
      hail-mary 448.180 us pipe_host card0
      timed 2800 spin=0 (04bv). Loses to
      LUT 83.659 (~5.36x), s8 16.060
      (~27.9x), GEMV bf16 26.962
      (~16.6x). Over STOP 177 (~2.53x).
      cosine=0 max_abs=0 ok=0 (all-zero
      C; N%16 synthetic). STOP.
- [x] iselect 16-entry table (lost 6x at 5120). One fire at
      1856 then STOP if still a loss.
      hail-mary 537.031 us pipe_host card1 at
      2800 (04bd). Loses to merge LUT 83.659
      (~6.42x), closed-form 71.715 (~7.49x),
      s8 16.060 (~33.4x), W8A8 44.285
      (~12.1x). K-linear vs 5120 1022.
      STOP.
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
- [x] FP8 mamba in as W8A16 incumbent, not an E2M1 spoof.
      51.036 us card1 (04bm).
      Loses to s8 23.504
      (~2.17x). Fat N left
      44-class. Separate
      CONFIG. out 43.694
      (04bo).
- [x] FP8 mamba out as W8A16 incumbent, not an E2M1 spoof.
      43.694 us card1 (04bo).
      Wash vs W8A8 o-proj
      44.081 (~0.99x). Loses
      to s8 23.115 (~1.89x).
      Same MNK as attn o-proj.
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
