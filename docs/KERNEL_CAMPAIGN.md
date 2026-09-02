# Kernel and math campaign

Open research map for Xe2 / B70 kernels, integer DPAS, NVFP4 spoofing,
and later TP=2 / PP=2 fabric. Updated 2026-09-02.

This file is a question list, not a locked design. Agents may pick any
workstream after P0 (host freeze + health). Do not treat a suggested
arm as the required implementation. Record what was actually tried.

## Napkin is not evidence

A guess, a schoolbook count, a datasheet roof, or "this should lose"
is a hypothesis. It does not go in FINDINGS.md and it does not skip
an arm.

- Write the guess in CONFIG as a prior, with the formula if any.
- Run actual code on actual tensors on these two B70s.
- Report the measured us, TOPS, GB/s, and numeric error.
- If the guess was right, the FINDING is the measurement, not the
  napkin. If it was wrong, that surprise is the FINDING.

This applies to precision compose, decode-vs-prefill roofs, "INT2 is
useless", "PP=2 cannot win decode", "NVFP4 cannot feed XMX", "we
cannot beat XeTLA/oneDNN", and every other prior in this file.

## Attitude: no one has tried a real SYCL / L0 kernel

Intel papers, XeTLA, oneDNN JIT, sycl-tla, and "the published DPAS
mix" are incumbents. They are floors to dump and beat, not ceilings.
The CUDA-community line is "no one has tried a real CUDA kernel."
Same here for SYCL, ESIMD, and Level Zero. A published int2xint8
recipe is a starting dump (K1/K2), then we write a real kernel.

Do not skip an arm because Intel already shipped it. Do not call a
match of their number a win. Win is lower wall time on this host.

## Latency ranks first

For serving-shaped work, **wall time and launch latency beat TOPS%
and bandwidth%.** A kernel at 20% of 367 TOPS that returns in 40 us
beats a 90% TOPS kernel that takes 200 us with extra launches or an
allreduce. Decode is a latency problem. Prefill can care about TOPS.
Always report us (and us/token when a shape implies a token). TOPS
and GB/s are diagnostic columns, not the ranking key.

This is why fusing ops for TP=2 matters: each extra collective is
latency, not just bytes.

## TP=1 vs TP=2 is per-op

Some ops want TP=1 (replicate, no comm). Some want TP=2 (shard,
allreduce). Joining adjacent ops in TP=2 is beneficial when it
removes a P2P/allreduce that would have sat between them. Count
collectives per token as a first-class score. Do not assume "TP=2
everywhere" or "fuse everything." Measure the same op TP=1 vs TP=2
vs fused-neighbors-TP=2. Latency of the comm, not peak GB/s of a
256 MiB allreduce, decides.

Sibling prior (unmeasured here): this fabric is push-fast / read-slow
and allreduce-hostile. Fusion and PP=2 exist to cut that latency.

Sibling-lab notes (b70_ai_things, flashnext, Steve) are hypotheses to
reproduce here. They are not xe2x2 FINDINGS until this host repeats
them with named backend, card pin, compiler identity, and health.

Language default: C++ SYCL / ESIMD for device code, Python for the
harness. OpenCL, Triton-XPU, and oneDNN-only wrappers are labeled
controls, not bans. AOT target is `intel_gpu_bmg_g31`. Backend name
in every CONFIG (`sycl+l0` default).

## Dual-card scheduling

Two B70s. One-card kernel and math work runs two-wide.

- Independent one-card jobs: `gpu-run --card 0` and `gpu-run --card 1`
  at the same time, same binary when the question is "does card1 match
  card0", different arms when the question is a matrix.
- Never mix a two-card collective with a one-card job. Either two
  independent one-card runs, or one two-card run.
- Per-card compile caches. Do not race one IGC cache directory across
  two cards.
- Pairing suggestion, not a law: even workstream ids on card0, odd on
  card1, then swap once so every arm has both-card evidence.
- Collectives, P2P, TP=2, PP=2 take both cards and pause the one-card
  matrix.

Datasheet roofs (Intel, not yet a local FINDING): 367 INT8 TOPS, 608
GB/s, 256 XMX. Record percent-of-roof as a diagnostic after a local
copy/GEMM floor exists. Rank serving-shaped kernels by us. Decode
M=1 and prefill large-M are different questions; do not average them
into one TOPS number.

## What we already believe (reproduce, do not cite as ours)

From b70_ai_things / flashnext, to re-measure:

- Xe2 XMX DPAS lights INT8 natively. Portable APIs (oneDNN,
  joint_matrix, Triton tl.dot) floor at INT8.
- ESIMD `dpas<s4,s4>` compiled, disassembled as `dpas.s4.s4`, numeric
  0/128, about 2x INT8 MAC rate. Prefill / W4A4 lever, not decode.
- Silicon also has INT2 matrix modes. No xe2x2 measurement yet.
- oneDNN s4/u4 GEMM is weight decompress, not INT4 DPAS.
- NVFP4 E2M1 x2 is exact int8 `{0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}`.
  Load-time s8 repack served 8B; 27B s8 repack did not fit one card.
- E2M1 is a float LUT. It is not two's-complement s4. +-12 overflows
  s4 [-8,7]. oneDNN s4 is the wrong contract.
- `nvfp4_gemm_w4a16` keeps packed E2M1 in VRAM (decode bytes), not
  INT4 XMX.
- M=1 decode sits 30-300x under the compute roof. Extra launches hurt
  decode. Extra TOPS help prefill.
- W8A8 serving paid ~160 per-token activation-quant launches. FP8
  W8A16 skipped them. Native s8 GEMM alone did not close that gap.
- Push all-reduce plus fused RMSNorm was a useful fabric prototype
  (`xpu_push_ar_fused_rmsnorm.cpp`). Minimum AR count is still open.
- Hardware VNNI transform (`lsc_load_2d` Transformed) was bit-exact;
  host-prepacked flat VNNI was not, on BMG-G31.

Reproduce or refute each bullet in this repo before FINDINGS.md.

## Precision compose: can INT2 / INT4 stand in for INT8?

Prior (unmeasured): schoolbook compose of s8 from s4/s2 is likely
slower than native s8 if MAC rate scales with bitwidth. That prior
is why the arm exists, not why it can be skipped. A second prior:
the compose that may *win* is representing NVFP4 E2M1 as a short sum
of integer DPAS terms, because E2M1 is not an integer type.

Schoolbook split. An s8 value as two base-16 s4 digits:

```
a = a1 * 16 + a0
b = b1 * 16 + b0
a*b = a1*b1*256 + (a1*b0 + a0*b1)*16 + a0*b0
```

That is four s4 MACs plus shifts/adds per s8 MAC (three with
Karatsuba). Four s2 digits per s8 is sixteen schoolbook terms.

If MAC rate scaled with bitwidth (s4 ~2x s8 is a sibling-lab
hypothesis; s2 ~4x s8 is a guess), four s4 MACs would cost about 2x
one s8 MAC, and sixteen s2 MACs about 4x. Native s8 DPAS exists, so
the prior is "compose loses." Measure it. Reasons the prior can die:

- DPAS K-depth, repeat count, and packing may not scale as 1/bits.
- Karatsuba / odd-even / residue splits change the term count.
- Epilogue shift+add may be free or may kill issue rate.
- s2/s4 may have better register blocking for some MNK.
- Unsigned u2/u4 plus a sign plane may beat signed s2.

Open measurements (do not skip the dumb ones):

1. Does `dpas<s2,s2>` / `dpas<u2,u2>` compile on BMG-G31?
2. What encoding does IGC emit? Keep the disasm.
3. Numeric closure on random s2/u2 tiles vs a host oracle.
4. MAC/s vs s4 vs s8 vs mixed (s8 x s4, s4 x s2, ...).
5. Schoolbook and Karatsuba s8-from-s4 and s8-from-s2 vs native s8
   at the same MNK. Report terms, shifts, extra bytes, TOPS, and
   GB/s separately.
6. Bitplane / binary GEMM (popcount or s2 planes) as a control.

NVFP4-shaped compose, more likely to matter:

E2M1 magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Each is a single
dyadic or a sum of two. After x2 the int codes are
`{0,1,2,3,4,6,8,12}`. s4 holds `[-8,7]`, so 8 and 12 overflow.

Ideas to try, none required:

- Two-term `w = w_lo + 8*w_hi` with `w_hi` in `{0,1}` and `w_lo` in
  s4. Two s4 DPAS plus add vs one s8 LUT.
- Sparse correction only on overflowing codes.
- Dyadic planes `{0.5,1,2,4}`: at most two scaled binary/s2 GEMMs
  per E2M1 weight.
- Keep packed nibbles in HBM, LUT in registers to s8, one s8 DPAS
  (on-the-fly spoof). This is the decode-bandwidth play.
- True integer s4/s2 GEMM only on GPTQ/AWQ/RTN integer checkpoints,
  not on NVFP4.

MXFP4 (e8m0, group 32) is a third format. Do not mix it into an NVFP4
arm without labeling it.

## Workstreams

Each workstream has a directory under `kernels/` or `parallel/`.
READMEs there are the agent brief. Pick one question per experiment
directory run. Dual-wide where the work is one-card.

### K0  Roofline floor          `kernels/roofline/`

Copy / STREAM-class, then s8 GEMM, then M=1 GEMV. Percent of 608 GB/s
and 367 INT8 TOPS. Card0 || card1.

### K1  Incumbent ISA           `kernels/onednn_isa/`

Dump IGC / L0 traces of current well-optimized Intel ops:
`fp8_gemm_w8a16`, `int8_gemm_w8a16`, `int8_gemm_w8a8`,
`int4_gemm_*`, `nvfp4_gemm_w4a16`, XeTLA / sycl-tla if reachable.
The dump is a floor to beat, not a ceiling. The question is what
IGC already emits, then whether a real ESIMD/L0 kernel is faster
in *us*.

### K2  ESIMD DPAS micro        `kernels/esimd_dpas/`

Hand tiles. s8, s4, s2, mixed. Transformed LSC 2D vs prepack. RC / K
/ N sweeps. This is how we light INT2 silicon even if we do not yet
know a model that wants it.

### K3  Precision compose       `kernels/precision_compose/`

s8-from-s4, s8-from-s2, Karatsuba, bitplanes, E2M1 two-term and
dyadic splits. Mathematician bait. Numeric oracle first, speed second.

### K4  W8 kernel-to-kernel     `kernels/w8_compare/`

`fp8_gemm_w8a16` vs `int8_gemm_w8a16` vs `int8_gemm_w8a8` at Qwen3.8
TP=2 shapes. GEMM-only and, for W8A8, GEMM+quant as two rows. M =
1,2,4,64,256,1024. Scale granularity is part of the arm name.

### K5  Kill the quant launches `kernels/epilogue_quant/`

How to move to INT8 without ~160 standalone activation-quant kernels.
Producer epilogue, residual dedup, fusedq, static scales, mega-kernel.
Do not start from a serving wrap.

### K6  NVFP4 everything        `kernels/nvfp4/`

Spoof yes, bitcast no, and then try every spoof. Load-time s8 LUT,
resident 4-bit JIT unpack, on-the-fly nibble->s8 DPAS, s4 overflow
split, dyadic planes, MXFP4 as a labeled third format, integer s4
on a non-NVFP4 checkpoint as the true INT4 XMX control.

### K7  GDN hybrid path         `kernels/gdn/`

Qwen3.8-27B and 35B-A3B-class models are GDN hybrids. Inventory plus
micros for conv1d / delta update / qkvz. A GEMM win that leaves GDN
eager is not a model win.

## Hail-mary arms (still measure)

These are allowed. They are not the first binary. Label them
hail-mary in CONFIG. Kill or keep with numbers.

1. **int2 x int8 DPAS, not just int2 x int2.**
   arXiv 2508.06753 (Intel/XeTLA) shows Xe2 native
   `int2 x int8 -> int32` DPAS with VNNI16-packed int2 weights and
   an IGC dump containing `dpas.8x8 ... r:s2 ... r:b`. Our first
   INT2 mix should include `s2 x s8` / `u2 x u8`. The s2xs2 arm
   stays. The mix is the one the literature actually lights.

2. **E2M1 is 16 codes. Product is a 256-entry LUT.**
   Signed E2M1 has 16 values; A*B is a closed table of exact
   products, then scale. For M=1 GEMV this might beat a systolic
   tile (or lose; measure). It is also the cheapest numeric oracle
   for NVFP4. Try it as a decode-shaped kernel, not as a prefill
   TOPS play.

3. **Large GRF (256) vs regular (128).**
   Xe2 threads have a dual GRF mode. Occupancy vs register blocking
   is an A/B, not a religion.

4. **Standalone ESIMD vs fat SYCL tree.**
   intel/llvm#21741: B70 DPAS was correct standalone and wrong in a
   large multi-TU build. Every K2 number needs a standalone oracle.

5. **Integer RMSNorm / stay-in-s8 residual.**
   Hard numeric contract. Only after K5 producer-epilogue exists as
   the sane path.

6. **PP=2 + microbatch + push-only handoff**, MoE experts pinned per
   stage. Fabric hail mary. Pause kernel matrix. Health around it.

7. **Persistent kernels / software pipelined mainloop** (load next K
   while DPAS current K). oneDNN may already do this; dump first
   (K1) before rewriting.

8. **W4A8 progressive quant (QServe-style)** so dequant does not
   leave the XMX pipe. Protective range, subtract-after-multiply.
   NVIDIA paper; the Xe2 analogue is an experiment.

## Literature to fetch (read-only agent is fine)

Put notes under `results/` or a short JOURNAL pointer. Do not paste
foreign tok/s into FINDINGS.

Xe2 / DPAS / GEMM:

- Intel oneAPI GPU optimization guide, Xe architecture chapter
  (XVE vs XMX, GRF 128/256, SLM).
- ESIMD DPAS API (`sycl_ext_intel_esimd`, `xmx::dpas`).
- XeTLA: https://intel.github.io/xetla (int8 GEMM templates,
  epilogues).
- sycl-tla / CUTLASS Xe, target `intel_gpu_bmg_g31`.
- arXiv 2508.06753 -- Xe2 int2xint8 GEMM, VNNI16, XeTLA autotune.
- intel/llvm#21741 -- B70 ESIMD DPAS wrong in large SYCL projects.
- PTI-GPU / unitrace: https://github.com/intel/pti-gpu

Formats:

- OCP MX spec v1.0 (MXFP4 = E2M1 + E8M0 / 32).
- NVIDIA NVFP4 blog (E2M1 + E4M3 / 16 + tensor scale). Not MXFP4.
- QServe / QoQ arXiv 2405.04532 -- W4A8, keep math on the systolic,
  dequant in-register.
- QuaRot / FlatQuant / PrefixQuant -- only when W4A4 / Hadamard is
  the question (`docs/quant` in the serving tree).

Model / hybrid:

- Gated DeltaNet papers (the Qwen3.8 linear-attn path).
- FlashAttention on XPU is a control; do not start there.

Fabric:

- Sibling `b70_ai_things/docs/P2P_GPU.md` J.13-J.19 (PP=2 vs TP=2
  on this push-fast/read-slow box). Reproduce, do not cite as ours.

### P2  TP=2 fabric             `parallel/tp2/`

After one-card kernels have a floor. Allreduce / allgather /
send-recv. P2P off first. Push all-reduce is an arm, not the answer.
Minimum call count is a first-class question.

### P3  PP=2 handoff            `parallel/pp2/`

Activation bubble, host vs device copy, stage-memory split.

### P4  mixed 2x2               `parallel/2x2/`

Only after P2 and P3 each have a passing correctness + health run.

## Roofline split (do not collapse)

Rank by wall time first.

Decode / M=1: us, launch count, then bytes. TOPS is a lie here if
XMX is idle.

Prefill / large M: us for the shape, then TOPS toward 367, plus
whether INT4/INT2 rate multipliers show up.

Fabric / TP=2: us per collective and collectives per token, then
GB/s of the payload, health after teardown. Joining ops that delete
an allreduce is a latency win even if the fused kernel is "less
efficient."

A kernel may win one column and lose the others. That is a valid
verdict. Latency is the ranking key for anything that will sit on
a decode path.

## Safety

P0 health before GPU touch. `gpu-run` for every real device use.
Per-card then two-rank collective health around TP=2 / PP=2. No
arbitrary `CCL_TOPO_P2P_ACCESS=1` in a serve. Slot moves are a
separate topology A/B (`docs/HOST.md`); do not mix them into a
kernel matrix.

ASCII only. CONFIG -> COMMAND -> RESULT -> VERDICT. Promote durable
results to FINDINGS.md. Community tok/s stay in refs/.

After a math floor exists, model shelf and quant policy:
`docs/MODELS.md`. Dense 27B first, then a 35B-A3B-class MoE, Gemma
26B-A4B as compact MoE, Flash-Next as stretch.

## Four B70s (evidence-gated)

The operator will add a 3rd and 4th Arc Pro B70 to this machine if
xe2x2 produces evidence that more cards pay. Do not treat 4x as the
current map. Do not buy-the-cards as an experiment.

This board (ASRock Fatal1ty X399 Professional Gaming) is nominally
x16/x8/x16/x8 in four-card mode, dual-slot coolers eat the pitch,
and the two extra slots are the x8 partners on each die. Four cards
change cooling, power, BDF count, `gpu-run` / health / collective
probes (today two-card), and rank maps (TP=4, DP=2xTP=2, PP=2xTP=2,
PP=4). Prior that would warrant it, all unmeasured at 4x:

- TP=2 decode on two cards is healthy and comm-latency-bound such
  that two more ranks would cut shard width enough to win us/token.
- PP=2 + batch saturates two stages and wants two more.
- Independent replicas: two more 27B/MoE copies beat one fatter
  parallel map.

Until then: two cards, dual-wide one-card kernels, TP=2/PP=2 as
written. Note 4x implications in a JOURNAL verdict if a result
clearly would scale past two, but do not block on hardware we do
not have.
