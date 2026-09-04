# JOURNAL

Append new entries at the bottom. Format: CONTEXT, CONFIG, COMMAND,
RESULT, VERDICT. ASCII only.

Active window starts at 2026-09-02a.

### 2026-09-02a - lab created

CONTEXT -> Operator asked for a lab and research arena for Intel GPUs,
  kernels, and TP=2 / PP=2 on this machine. Named xe2x2. Sibling
  serving tree remains b70_ai_things.

CONFIG -> Host b70s4dayz, kernel 7.1.0-070100-generic, KMD xe,
  two Arc Pro B70 (8086:E223) at 0000:0b:00.0 and 0000:44:00.0,
  NEO 26.22.38646.4, 256 EU and 30.3 GiB per card. Default branch
  master. Visibility public.

COMMAND ->
  ```
  gh repo create xe2x2 --public --description "Xe2 dual-B70 kernel and 2x2 parallelism lab (TP=2, PP=2)"
  ```

RESULT -> https://github.com/hotschmoe/xe2x2 created. Local path
  /mnt/vm_8tb/github/xe2x2. Host inventory written to docs/HOST.md.
  PCI tree shows the two cards on separate Intel e2ff bridges.

VERDICT -> Lab open. Next work is RESEARCH_TODO P0: freeze identities
  and run per-card plus two-rank collective health. Do not start mixed
  2x2 before both single-axis maps have a passing run.

### 2026-09-02b - references and backends

CONTEXT -> Operator: attach Steve (neural.download git), the SergiioB
  cookbook, flashnext-harness, and xecores.com. Drop CLAUDE.md. Write
  Level Zero / SYCL backends and first-party Intel refs.

CONFIG -> Submodules under refs/, shallow. Agent file is AGENTS.md
  only.

COMMAND ->
  ```
  git rm CLAUDE.md
  git submodule add --depth 1 \
    https://github.com/steveseguin/b70-optimization-lab.git \
    refs/b70-optimization-lab
  git submodule add --depth 1 \
    https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git \
    refs/intel-arc-pro-b70-inference-cookbook
  git submodule add --depth 1 \
    https://github.com/bbeartheancient/flashnext-harness.git \
    refs/flashnext-harness
  ```

RESULT -> docs/BACKENDS.md and docs/REFERENCES.md added. Intel OMIX,
  L0 spec, intel/llvm SYCL, compute-runtime, IGC, oneDNN, oneCCL,
  sycl-tla (intel_gpu_bmg_g31), torch-xpu-ops, Triton-XPU listed.
  Default Xe2 path is SYCL on Level Zero V2 immediate lists.

VERDICT -> Peek list is local. Do not import foreign tok/s into
  FINDINGS. CLAUDE.md stays gone.

### 2026-09-02c - kernel / NVFP4 / INT2 campaign written

CONTEXT -> Operator asked to record the kernel-to-kernel INT8 vs FP8
  plan, how to drop ~160 activation-quant launches, close-to-metal
  XMX work, NVFP4 spoofing in full, INT2 silicon, and dual-card
  parallel runs, without locking an implementation path. Also asked
  whether INT2/INT4 DPAS terms can reconstruct an INT8 MAC.

CONFIG -> Docs only. No GPU touch. Host still the 2026-09-02a
  snapshot (two B70s, kernel 7.1, NEO 26.22). Language default C++
  SYCL/ESIMD + Python harness, already agreed.

COMMAND -> write docs/KERNEL_CAMPAIGN.md; kernel workstream READMEs
  K0-K6; parallel/tp2, pp2, 2x2 READMEs; point RESEARCH_TODO,
  FINDINGS, AGENTS, repo README at the campaign.

RESULT -> Campaign is a question map. Napkin on precision compose:
  s8-from-s4 is four schoolbook digit-MACs (three Karatsuba); if s4
  MAC rate is ~2x s8, compose likely loses to native s8. INT2 is the
  same story with more terms. The compose that may matter is E2M1 as
  a short sum of integer terms (overflow split of codes 8 and 12,
  dyadic planes). INT2 still gets a first-class K2/K3 arm so the
  unit is exercised even without a model. NVFP4 arms list load-time
  s8 LUT, resident 4-bit JIT, on-the-fly nibble LUT, s4 split,
  dyadic planes, MXFP4 as a third format, and integer s4 as the true
  INT4 XMX control. One-card work is two-wide on card0 || card1.
  TP=2 keeps push all-reduce and minimum call count as open arms.

VERDICT -> P0 remains the GPU gate. After that, agents may pick any
  K-workstream. Sibling-lab bullets stay hypotheses until reproduced
  here. Slot moves stay out of the kernel matrix.

### 2026-09-02d - measure napkins; post-math model shelf

CONTEXT -> Operator: every guess / "napkin says compose loses" must
  be verified with code and values; surprises are findings. After
  pure math, which models to keep for serving research: Qwen3.8-27B,
  Flash-Next, Qwen3.6-35B-A3B, Gemma-26B-A4B. Quants: XMX-native
  INT8/INT4 plus NVFP4. TP=2 decode and PP=2 MoE/batching both in
  play.

CONFIG -> Docs only. Local weights under
  b70_ai_things/models/files: qwen3.8-27b (bf16 52G, fp8 29G, w8a8
  35G, gptq-int4 19G, nvfp4 21G), qwen3.6-27b nvfp4, ornith-1.5-35b-a3b
  (bf16, gptq-int4, nvfp4, w8a8). No Qwen3.6-35B-A3B and no Gemma 4
  26B A4B in the live files tree.

COMMAND -> add docs/MODELS.md; standing "napkin is not evidence"
  rule in docs/KERNEL_CAMPAIGN.md and AGENTS.md; point RESEARCH_TODO
  / FINDINGS / README.

RESULT -> Dense primary is Qwen3.8-27B (already complete quant set).
  MoE primary is Qwen3.6-35B-A3B when fetched; Ornith-1.5-35B-A3B is
  the on-disk size-class stand-in. Gemma 4 26B A4B is the compact
  one-card MoE fetch. Flash-Next stays stretch (NVMe expert/PLE, not
  2xB70 resident). Quant focus INT8 W8A16/W8A8, integer INT4, NVFP4
  spoof; FP8 W8A16 is the dense incumbent control. TP=2 vs PP=2 vs
  c>1 batch is an A/B on the same model, not a religion.

VERDICT -> Do not skip K3 because the napkin says compose loses.
  Do not start Flash-Next serving to skip 27B. Fetch 35B-A3B / Gemma
  when MoE fabric work starts.

### 2026-09-02e - launch brief, GDN, hail marys, literature

CONTEXT -> Operator: anything to add before agents; instructions,
  hail-mary plays, extra research / literature.

CONFIG -> Docs only. No GPU.

COMMAND -> write docs/AGENT_LAUNCH.md; kernels/gdn/ (K7); hail-mary
  and literature sections in docs/KERNEL_CAMPAIGN.md; campaign
  papers in docs/REFERENCES.md.

RESULT -> Under-weighted GDN (Qwen3.8 is hybrid). Hail mary 1 is
  literature-backed: Xe2 native int2xint8 DPAS (arXiv 2508.06753,
  VNNI16), so K2 must try s2xs8 not only s2xs2. Hail mary 2 is the
  16-code E2M1 product LUT. Landmine: intel/llvm#21741 B70 DPAS
  wrong in fat SYCL trees; first micros are standalone binaries.
  Record clocks before quoting TOPS percent. Launch rule: one
  GPU agent per card; literature agents need no lease.

VERDICT -> Set agents loose on P0, then the card0||card1 pairs in
  AGENT_LAUNCH.md. One read-only literature agent in parallel is
  useful. Do not start six GPU agents on one lease.

### 2026-09-02f - beat Intel; latency; per-op TP; four-card gate

CONTEXT -> Operator: (1) published Intel/XeTLA/oneDNN is not a
  ceiling; motto like "no one has tried a real CUDA kernel" but for
  SYCL/L0. (2) Some ops want TP=1; joining ops in TP=2 cuts P2P
  latency. (3) Latency/time almost outranks BW% and TOPS. (4) 3rd
  and 4th B70s if evidence warrants.

CONFIG -> Docs only.

COMMAND -> standing rules in docs/KERNEL_CAMPAIGN.md, AGENTS.md,
  AGENT_LAUNCH.md, parallel/tp2, HOST.md four-card note.

RESULT -> Incumbents are floors. Rank serving-shaped kernels by us.
  TP=1 vs TP=2 vs fused-neighbors is per-op. Four-card is
  x16/x8/x16/x8 on this board, gated on measured two-card wins,
  not a current experiment.

VERDICT -> Agents dump Intel, then try to beat it in wall time.
  Do not block on hardware we do not have.

### 2026-09-02g - P0 host freeze and health

CONTEXT -> Orchestrator session. P0 is the only hard GPU gate:
  identities, per-card health, two-rank collective with P2P off,
  no live serve, JOURNAL.

CONFIG -> backend mix named below. Kernel 7.1.0-070100-generic,
  KMD xe, two B70 8086:E223 at 0000:0b:00.0 and 0000:44:00.0.
  No display (all DP/HDMI disconnected). No live serve (grafana /
  prometheus / open-webui only). gpu-run lease was free. Host
  oneAPI DPC++ 2026.1.1.20260724 (relocated steve-repro tree).
  Health images: vllm-xpu-env:int8g-v0251 (per-card pytorch-xpu)
  and b70-sglang-xpu-int8-runtime:20260826-mtp6
  sha256:adc915d266ea... (two-rank xccl).

COMMAND ->
  ```
  # identities: uname, dpkg, clinfo, journalctl firmware, sysfs clocks
  # sycl-ls + per-card health, two-wide:
  gpu-run --card 0 bash -lc 'source setvars; ZE_AFFINITY_MASK=0 sycl-ls --verbose;
    xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180'
  gpu-run --card 1 bash -lc 'source setvars; ZE_AFFINITY_MASK=1 sycl-ls --verbose;
    xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180'
  # then both cards:
  gpu-run xpu-collective-health --p2p 0 --timeout 240
  ```

RESULT ->
  Identities: NEO 26.22.38646.4, IGC 2.36.3, L0 loader 1.28.2-2,
  libze_intel_gpu 1.15.38646, GuC 70.58.0, HuC 8.2.10, DMC 2.6.
  sycl-ls live adapter is Unified Runtime over Level-Zero V2 on
  both cards, architecture intel_gpu_bmg_g31, ext_intel_esimd and
  ext_intel_matrix present. OpenCL platform is the NEO control.
  card0 UUID ...0b00..., card1 UUID ...4400...
  GT0 idle 400 MHz, max 2800 MHz, power cap 230 W, D3hot at rest.
  xpu-health card0 HEALTHY; card1 HEALTHY (gpu-run, ~26s each).
  COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10
  p2p=0 (gpu-run both cards, 28s).
  Host has no g++; icpx AOT needs a container g++ (CPU docker).

VERDICT -> P0 gate is green. Kernel workstreams may run two-wide.
  Default kernel backend is sycl+l0 on L0 V2. Do not enable P2P.
  Next: AGENT_LAUNCH pair 1 (K0 copy card0 || K0 s8 square GEMM
  card1), then swap. Artifacts: results/p0/SUMMARY.md, docs/HOST.md
  P0 freeze. L0 V2 fact promoted to FINDINGS.md.

### 2026-09-02h - K0 copy roof and boring s8 GEMM, both cards

CONTEXT -> AGENT_LAUNCH pair 1 then swap. Napkin 608 GB/s / 367
  INT8 TOPS is CONFIG. Boring SYCL tile is labeled XVE, not DPAS.

CONFIG -> backend sycl+l0, icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  NEO 1.15.38646+4, ZE_AFFINITY_MASK per card, gpu-run --card N.
  GT0 cur=2800 MHz, throttle 0. copy_roof USM event-profiled,
  5 warmup + 20 iters. s8_square_gemm 16x16 local tile, 2 warmup
  + 8 iters, host s32 oracle at n=64.

COMMAND ->
  ```
  gpu-run --card 0 .../copy_roof
  gpu-run --card 1 .../s8_square_gemm --n 256,1024,2048 --check-n 64
  # swap
  gpu-run --card 1 .../copy_roof --bytes 4k..256MiB
  gpu-run --card 0 .../s8_square_gemm --n 256,1024,2048 --check-n 64
  ```

RESULT ->
  D2D 256 MiB: 551-553 GB/s card0, 550 GB/s card1 (~90% of 608).
  D2D 16 MiB: 622-637 GB/s both cards (above datasheet; cache-sized).
  D2D 32-128 MiB: 561-569 GB/s.
  H2D 256 MiB: 13.5-14.2 GB/s. D2H 256 MiB: 4.66-4.67 GB/s both.
  s8 n=2048: 9049 us / 1.899 TOPS card0, 9053 us / 1.898 TOPS
  card1 (0.52% of 367). n=64 max_abs=0 both. Copies byte-exact.

VERDICT -> Local HBM copy floor is ~550 GB/s at 256 MiB, not the
  16 MiB 622. D2H is the slow host direction. XVE s8 tile is a
  1.9 TOPS / 9.05 ms floor; K2 must beat it in us. Promoted to
  FINDINGS. Next pair: K2 dpas s8 || dpas s4 standalone.

### 2026-09-02i - campaign literature notes (no GPU)

CONTEXT -> Always-on read-only agent. Papers in docs/REFERENCES.md.

CONFIG -> no GPU. Notes under results/literature/. No tok/s.

COMMAND -> fetch arXiv 2508.06753, intel/llvm#21741, ESIMD
  xmx::dpas, XeTLA, NVFP4 vs OCP MX, QServe 2405.04532, Gated
  DeltaNet, oneAPI Xe guide, IGC DPAS.md.

RESULT -> INDEX.md plus one file per source. All eight priority
  items landed. Paper ISA (not silicon): Xe2 native int2xint8
  DPAS encoding `dpas.8x8 ... rW:s2 rA:b`, VNNI16, same systolic
  rate as s8 (OPS_PER_CHAN=4, K=32) not 4x; IGC s2 range [-2,1].
  #21741 still OPEN (standalone exact, fat SYCL tree WI>=1 wrong).

VERDICT -> Literature only. Do not put paper MAC rates in
  FINDINGS. K2 s2xs8 mix is the literature arm; standalone
  oracle is mandatory.

### 2026-09-02j - K0 M=1 s8 GEMV both cards

CONTEXT -> K0 suggested arm: decode-shaped GEMV at Qwen-like K,N.
  Rank us, then bytes. TOPS is diagnostic.

CONFIG -> backend sycl+l0, naive per-N loop, M=1, k=5120 n=5120
  and n=17408, gpu-run --card 0 || --card 1, GT0 cur=2800.

COMMAND ->
  ```
  gpu-run --card N .../s8_gemv --k 5120 --n 5120
  gpu-run --card N .../s8_gemv --k 5120 --n 17408
  ```

RESULT -> 5120x5120: 989 us card0 / 990 us card1, 26.5 GB/s,
  max_abs=0. 5120x17408: 2241 us card0 / 2235 us card1, ~40 GB/s,
  max_abs=0. ~14x slower than a 550 GB/s copy of the weights.

VERDICT -> Decode floor is 1-2 ms on this boring kernel. Promote
  us to FINDINGS. A DPAS/oneDNN GEMV that returns in a few tens
  of us would be the actual serving win.

### 2026-09-02k - K2 ESIMD DPAS s8/s4/s2/s2xs8 both cards

CONTEXT -> AGENT_LAUNCH pairs 2 and 3, then swap. First DPAS
  micros are standalone binaries (intel/llvm#21741 still open).
  Literature prior: s2xs8 same systolic rate as s8; sibling s4
  ~2x s8. Napkin, then measure.

CONFIG -> backend sycl+l0, icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  xmx::dpas, lsc_load_2d Transformed=true, gpu-run --card N,
  1024^3, 5 warmup + 20 iters (s2 repeat 10+30). s2 packed
  [-2,1]. Host s32 oracle on check and timed shapes.

COMMAND ->
  ```
  gpu-run --card 0 .../dpas_s8 --m 1024 --n 1024 --k 1024
  gpu-run --card 1 .../dpas_s4 --m 1024 --n 1024 --k 1024
  # swap, then
  gpu-run --card 0 .../dpas_s2xs8 ...
  gpu-run --card 1 .../dpas_s2 ...
  # swap plus s2xs2 clock repeat
  ```

RESULT -> Compile: all four yes. Numeric: max_abs=0 every arm
  both cards, including s2xs8 and s2xs2. us at matched ~583 MHz:
  s8 374, s4 250 (1.49x), s2xs8 278 (faster than s8 despite the
  same-rate paper prior -- fewer B bytes), s2xs2 223 on card1.
  s2xs2 on card0 hit 49-89 us when cur_freq was 1950-2800; TOPS
  followed the clock, not a second algorithm. DPAS s8 374 us vs
  K0 XVE 2200 us at 1024^3.

VERDICT -> INT2 is real on these cards. Promote compile + numeric
  closure + us ranking to FINDINGS. Do not promote a 367-TOPS
  percent: short kernels droop to 583 MHz. s4 is 1.49x s8 here,
  not 2x. Next: hold clocks, dump IGC encoding, K1 oneDNN floor,
  K3 compose vs this native s8/s4.

### 2026-09-02l - K1 dump fp8_gemm_w8a16 and int8_gemm_w8a8

CONTEXT -> AGENT_LAUNCH pair 4. Dump incumbents, then beat them
  in us. Loop scheduler armed 20m.

CONFIG -> backend pytorch-xpu on sycl+l0. fp8:
  b70-local/vllm-openai-xpu:qwen38-fp8-mtp1-...-r50-s01.
  int8: sglang int8 runtime mtp6. gpu-run --card N,
  ZE_AFFINITY_MASK=N. Synthetic Qwen-ish M=1,64,256.

COMMAND ->
  ```
  run_dump.sh 0 fp8 <fp8-image>
  run_dump.sh 1 int8 vllm-xpu-env:int8g-v0251   # MISSING w8a16
  run_dump.sh 1 int8 <sglang>                   # HAS w8a8, no w8a16
  run_dump.sh 0 int8a8 <sglang>
  run_dump.sh 1 fp8 <fp8-image>                 # swap
  run_dump.sh 1 int8a8 <sglang>
  ```

RESULT -> int8_gemm_w8a16 absent in three images. fp8_gemm_w8a16
  M=1 5120: 57.5 us card0 / 56.2 us card1. int8_gemm_w8a8 GEMM-only
  M=1 5120: 45.0 / 45.5 us. M=64 5120: fp8 92-94 us, int8 61-62 us.
  IGC dump of the fp8 call is gemm_zero_fill, not the mainloop.

VERDICT -> Promote GEMM-only INT8 vs FP8 us to FINDINGS. W8A16
  INT8 is a patched op, not stock here. Next: clock-held DPAS,
  K3 compose measure, beat 45 us at M=1 5120.

### 2026-09-02m - clock-held DPAS occupancy

CONTEXT -> Short K2 micros sat at ~583 MHz. Repeat with 20 warmup
  + 80 iters and sysfs clocks around the run.

CONFIG -> same standalone dpas_s8 / dpas_s4, 1024^3, gpu-run
  --card N, sycl+l0.

COMMAND -> kernels/esimd_dpas/run_clock_hold.sh N s8|s4 then swap.

RESULT -> Numeric still 0. us tracks start clock: s8 257 us
  (start 867 MHz) vs 110 us (card1). s4 131 us (start 1883) vs
  69 us / 31 TOPS (start 2800). 31 TOPS is ~8% of 367 on this
  untuned tile.

VERDICT -> Do not freeze the 1.49x s4/s8 ratio. Clock is a first-
  class column. Occupancy helps but this tile is not a roof. K3
  compose must compare at matched clocks.

### 2026-09-02n - int8_gemm_w8a16 exists as ref_matmul

CONTEXT -> K1 card1 agent found the W8A16 INT8 op in
  b70-sglang-xpu-int8-w8a16:20260828-2dd55f3 after stock images
  missed it. Repeat on card0.

CONFIG -> backend pytorch-xpu on sycl+l0, same dump_incumbent.py
  --op int8, gpu-run --card 0 and 1, ZE_AFFINITY_MASK per card.

COMMAND -> run_dump.sh N int8 b70-sglang-xpu-int8-w8a16:20260828-2dd55f3

RESULT -> HAS int8_gemm_w8a16=True. M=1 5120: 2027 us both cards.
  M=64 5120: ~12.2 ms. IGC: OpenCL ref_matmul, SRC_DT_F16
  WEI_DT_S8 DST_DT_F16, inner loop convert+mad, no dpas.

VERDICT -> Calling the symbol is not lighting XMX. Floor remains
  int8_gemm_w8a8 at 45 us. Promote: W8A16 INT8 in that image is a
  scalar reference. FINDINGS updated.

### 2026-09-02o - K3 compose measured; napkin loses

CONTEXT -> K3 binaries compiled CPU-only. Prior: schoolbook
  compose ~2.7x slower than native s8. Codex plan said E2M1
  two-term is the better first bet. Measure both.

CONFIG -> sycl+l0, standalone compose_s8_from_s4 and
  compose_e2m1_two_term, 1024^3, 5 warmup + 20 iters, gpu-run
  --card N then swap. Karatsuba host-only.

COMMAND ->
  ```
  gpu-run --card 0 .../compose_s8_from_s4 --m 1024 --n 1024 --k 1024
  gpu-run --card 1 .../compose_e2m1_two_term ...
  # swap
  ```

RESULT -> Numeric max_abs=0. Schoolbook vs native s8 in-binary:
  card0 207 vs 272 us (compose faster); card1 112 vs 115 us.
  E2M1 two-term vs s8 LUT: card1 222 vs 374 us; card0 83 vs 94 us.
  Karatsuba skip: (a0+a1) not in s4; host identity closed.

VERDICT -> Napkin "compose loses" is dead on this tile. Promote.
  Absolute us still clock-limited. Next: K4 W8 A/B vs 45 us
  W8A8 / 56 us FP8 floors.

### 2026-09-02p - fp8_gemm_w8a16 JIT is bf16 DPAS

CONTEXT -> K1 fp8 agent: IGC_ShaderDumpEnable was only
  gemm_zero_fill. Re-dump with ONEDNN_JIT_DUMP + iga Xe2.

CONFIG -> backend pytorch-xpu on sycl+l0, fp8 image r50-s01,
  card0, ngen gemm_kernel bins.

COMMAND -> ONEDNN_JIT_DUMP=1 dump_incumbent fp8; iga disasm
  dnnl_dump_gpu_gemm_kernel.*.bin

RESULT -> dpas.8x8 rW:bf rA:bf rAcc:f. E4M3 unpack is shl+mul
  then bf16 DPAS. No dpas.s8. xe_hp_systolic skipped for f8.

VERDICT -> Sibling "no native FP8 XMX" is now a local ISA
  FINDING. 56 us is decompress+bf16, not FP8 systolic. Promote.

### 2026-09-02q - K4 W8 A/B both cards plus git milestone

CONTEXT -> P0-K3 committed 8caea59 and pushed. K4 unmixes GEMM
  from serving quant.

CONFIG -> bench_w8.py, gpu-run --card N, M=1,2,4,64,256,1024
  at 5120, plus M=1 17408. Cosine vs host.

COMMAND ->
  ```
  run_w8.sh 0 fp8 <fp8-image>
  run_w8.sh 1 int8a8 <sglang>
  # swap
  git commit / git push origin master
  ```

RESULT -> INT8 W8A8 GEMM-only faster at every M. M=1 5120:
  42-46 vs 70-72 us. M=64: 46-49 vs 382-430 us. Cosine ~1.
  Milestone: https://github.com/hotschmoe/xe2x2/commit/8caea59

VERDICT -> Kernel ranking is INT8 W8A8, then FP8 W8A16 emulate.
  K5 must add quant launches before claiming a serve win. Promote
  the GEMM table.

### 2026-09-02r - K2 ocloc/IGA encodings

CONTEXT -> Clock-held DPAS already both-card. Runtime
  IGC_ShaderDumpEnable on AOT binaries dumped only SIP/caps.

CONFIG -> CPU ocloc disasm of unbundled sycl-spir64_gen zebin
  from the four standalone AOT bins. Tiny GPU tiles on card0
  (s8, s2xs8) and card1 (s4, s2) for numeric, backend sycl+l0.

COMMAND ->
  ```
  clang-offload-bundler --type=o --targets=sycl-spir64_gen --unbundle
  ocloc disasm -file device.elf -device bmg-g31
  gpu-run --card N kernels/esimd_dpas/run_igc_dump.sh N ARM
  ```

RESULT -> IGA inner loop is `dpas.8x8 (16|M0)` acc `:d`.
  s8 `rW:b rA:b`; s4 `rW:s4 rA:s4`; s2 `rW:s2 rA:s2`;
  s2xs8 `rW:s2 rA:b`. Tiny tiles max_abs=0. has_dpas true,
  GRF 128.

VERDICT -> Promote encodings. s8 vs s4 is the operand type on
  the same opcode. Runtime IGC dumps of AOT ESIMD are the wrong
  tool; disasm the zebin.

### 2026-09-02s - K5 naive RMSNorm-epilogue both cards

CONTEXT -> W8A8 serving mixed ~160 quant launches into the
  kernel ranking. Unmix: 2-launch vs fused producer epilogue.

CONFIG -> sycl+l0, standalone rmsnorm_epilogue, one WI/row,
  gamma=1 eps=1e-6, symmetric s8 qmax=127, gpu-run --card N.

COMMAND ->
  ```
  gpu-run --card 0 kernels/epilogue_quant/run_k5.sh 0
  gpu-run --card 1 kernels/epilogue_quant/run_k5.sh 1
  ```

RESULT -> M=1 5120: fused 830 us both cards vs two-launch
  1361/1252 us. M=1 17408: 2820 vs 3769 us. max_abs<=1 (f16
  vs float host). Fusion ~30-40% by dropping a launch.

VERDICT -> Launch fusion is real and closed enough. 830 us is
  not a serving epilogue next to 45 us W8A8 GEMM. Promote the
  launch-count result, not the us as a floor.

### 2026-09-02t - K6 nibble LUT spoof both cards

CONTEXT -> One NVFP4 spoof arm. Never bitcast E2M1 onto s4.
  K3 already measured two-term s4; this is nibble LUT to s8.

CONFIG -> sycl+l0, standalone nibble_lut_s8, packed E2M1 in
  HBM, LUT to s8, then K2 s8 DPAS. 1024^3, gpu-run --card N.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6.sh 0
  gpu-run --card 1 kernels/nvfp4/run_k6.sh 1
  ```

RESULT -> max_abs=0 both cards. Host LUT DPAS 271 us (card0
  550 MHz) / 75 us (card1). Device unpack+DPAS 305 / 84 us.
  Unpack tax ~12% at this tile.

VERDICT -> Nibble LUT spoof is numerically closed. Promote.
  Absolute us is clock. In-register fused LUT still open.

### 2026-09-02u - K5 WG-256 bandwidth epilogue both cards

CONTEXT -> Naive fused RMSNorm-quant was 830 us at M=1 5120.
  That was one WI looping K. Rewrite with WG=256 per row.

CONFIG -> sycl+l0, standalone rmsnorm_epilogue_bw, same
  contract, gpu-run --card N, GT0 cur=2800.

COMMAND ->
  ```
  gpu-run --card 0 kernels/epilogue_quant/run_k5_bw.sh 0
  gpu-run --card 1 kernels/epilogue_quant/run_k5_bw.sh 1
  # card0 repeat
  ```

RESULT -> Fused M=1 5120: 36 us card0 first, 13 us repeat,
  7 us card1. max_abs<=1. ~20-100x vs naive 830 us. Fusion
  still ~1.5x two-launch. Short kernels swing; do not freeze
  7 us.

VERDICT -> 830 us was the 1-WI loop. Producer epilogue is now
  tens of us, next to 45 us W8A8 GEMM. Promote with the swing.

### 2026-09-02v - K6 in-register nibble LUT VNNI4

CONTEXT -> Two-launch unpack tax was ~12%. Try one-launch
  packed load + GRF LUT + s8 DPAS. Never bitcast s4.

CONFIG -> sycl+l0, nibble_lut_reg, packs raw/vnni4/kmajor,
  1024^3, gpu-run --card N, GT0 cur=2800.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_reg.sh 1
  gpu-run --card 0 kernels/nvfp4/run_k6_reg.sh 0
  ```

RESULT -> VNNI4 max_abs=0 both cards (check and 1024^3). Raw
  and kmajor refuse. Timed VNNI4 2316-2317 us vs two-launch
  84-305 us.

VERDICT -> VNNI4 is the Transformed s8 B layout. This scalar
  LUT loses in us. Keep two-launch as the fast spoof. Promote
  the layout, not the 2316 us as a win.

### 2026-09-02w - serving-shape DPAS vs 45 us W8A8

CONTEXT -> Beat oneDNN int8_gemm_w8a8 at Qwen shapes. Existing
  8x16 tile, M=8 padded decode (RC=8). Napkin: pad M=1 to 8
  and maybe beat 45 us.

CONFIG -> sycl+l0, dpas_s8 and dpas_s4, gpu-run --card N,
  shapes M=8/64/256 x 5120 x 5120 and M=8 x 17408 x 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_serving.sh 0 s8
  gpu-run --card 1 kernels/esimd_dpas/run_serving.sh 1 s4
  # swap, plus M=8 s4 warmup20/iters40 repeat
  ```

RESULT -> max_abs=0. M=64 5120: s8 274-373 us vs W8A8 46-49.
  M=256: 691-1064 vs 75. Padded M=8 s4 34-120 us (repeat
  64-87); s8 56-230 us.

VERDICT -> This tile loses. Do not promote the 34 us one-off.
  Floor stays 45 us. Next is a real GEMM schedule.

### 2026-09-02x - K6 vectorized in-register LUT

CONTEXT -> Scalar in-register LUT was 2316 us. Vectorize
  nibble decode and VNNI4 pack.

CONFIG -> sycl+l0, nibble_lut_simd, 1024^3, gpu-run --card N.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_simd.sh 0
  gpu-run --card 1 kernels/nvfp4/run_k6_simd.sh 1
  ```

RESULT -> max_abs=0 both cards. 304 us card0 (633 MHz start)
  / 406 us card1 (2800 start). ~6-8x vs scalar 2316 us.

VERDICT -> simd LUT is a us win vs scalar in-register. Not a
  matched-clock beat of two-launch unpack. Promote the ratio,
  not a single us.

### 2026-09-02y - blocked s8 NT=2/4 A-reuse vs 45 us

CONTEXT -> 8x16 tile lost 6-12x to W8A8 at M=64/256. Prior:
  reuse A across more N so A loads drop.

CONFIG -> sycl+l0, dpas_s8_block NT=2 (8x32) and NT=4 (8x64),
  gpu-run --card N, M=8/64/256 x 5120 x 5120.

COMMAND ->
  ```
  gpu-run --card 0 run_block.sh 0 2
  gpu-run --card 1 run_block.sh 1 4
  # swap
  ```

RESULT -> max_abs=0. M=64: NT=2 269-352 us, NT=4 318-474 vs
  8x16 274-373 vs W8A8 46-49. M=256: NT=4 694-876 vs 8x16
  892-1064 vs W8A8 75.

VERDICT -> A-reuse does not close the gap. Promote the miss.
  Floor stays 45 us. Next: SLM/GRF256/prefetch or W8A8 ngen dump.

### 2026-09-02z - int8_gemm_w8a8 ngen ISA both cards

CONTEXT -> Hand 8x16 and NT=2/4 lost 6x+ to W8A8. Dump the
  incumbent schedule like fp8 (ONEDNN_JIT_DUMP + IGA Xe2).

CONFIG -> pytorch-xpu on sycl+l0, sglang int8 mtp6,
  gpu-run --card N, ONEDNN_JIT_DUMP=1. CPU libiga64 Xe2 disasm.

COMMAND ->
  ```
  gpu-run --card 0 kernels/onednn_isa/run_w8a8_jitdump.sh 0
  gpu-run --card 1 kernels/onednn_isa/run_w8a8_jitdump.sh 1
  python3 kernels/onednn_isa/iga_disasm.py dnnl_dump_gpu_gemm_kernel.{0,2,4}.bin
  ```

RESULT -> Bins md5-identical across cards. M=1: 64x dpas.8x4
  wg 8x2 k64. M=64: 64x dpas.8x8 grf256 wg 4x2x4 + SLM.
  M=256: 384x dpas.8x8 grf256 k128 no SLM. Native s8 `:b`.

VERDICT -> Steal RC=4 for decode and GRF256 for prefill. Promote.

### 2026-09-02aa - ESIMD RC=4 tile; GRF256 request refused

CONTEXT -> W8A8 ngen M=1 is dpas.8x4. Hand tile was RC=8.
  Try RC=4 decode pad M=4, and request GRF256 for M=64.

CONFIG -> sycl+l0, dpas_s8_rc4 and dpas_s8_grf256, gpu-run
  --card N. GRF via intelex::grf_size<256> and separately
  -ftarget-register-alloc-mode=pvc:large.

COMMAND ->
  ```
  gpu-run --card 0 run_rc4_grf.sh 0 rc4
  gpu-run --card 1 run_rc4_grf.sh 1 grf256
  # swap
  ocloc disasm AOT zebin
  ```

RESULT -> RC=4 IGA is dpas.8x4, max_abs=0. M=4 5120: 77-168 us
  vs W8A8 M=1 45 us. M=64: 614-894 vs RC=8 274-373. GRF256
  zebin still grf_count 128 (property and pvc:large).

VERDICT -> RC=4 lights, does not beat 45 us. GRF256 request
  did not take. Promote the encoding + the miss.

### 2026-09-02ab - SLM A share WG=16 RC=4

CONTEXT -> W8A8 M=1 is dpas.8x4 wg 8x2 + SLM. Try sharing A
  in SLM across 16 threads (N=256 per WG).

CONFIG -> sycl+l0, dpas_s8_slm, RC=4, slm_block_store/load A,
  barrier per K-chunk, gpu-run --card N.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_slm.sh 0
  gpu-run --card 1 kernels/esimd_dpas/run_slm.sh 1
  ```

RESULT -> max_abs=0. M=4 5120: 372-463 us vs RC=4 no-SLM 77-168
  vs W8A8 45. M=64: 525-1083. M=256: 1296-1373.

VERDICT -> SLM A + per-K barrier loses. Promote the miss.
  Floor stays 45 us. Need the ngen unroll/k64, not just SLM.

### 2026-09-02ac - K2 k64 NT=2/4 vs 45 us W8A8

CONTEXT -> SLM A lost. W8A8 M=1 ngen is 64x dpas.8x4 k64.
  Try RC=4, two K=32 chunks per step, NT=2/4 A-reuse, no SLM.

CONFIG -> sycl+l0, standalone dpas_s8_k64, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Shapes 4/64/256 x 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_k64.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_k64.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> max_abs=0 both cards both NT. IGA is dpas.8x4 rW:b
  rA:b, GRF 128. NT=2 has 4 static dpas (K loop remains).
  NT=4 has 16. Not ngen's 64. M=4: 92-396 us (D3hot first
  pair 278/396, warm swap 92/111) vs RC=4 77-168 vs W8A8 45.
  M=64: 293-728 vs W8A8 46-49 vs RC=8 274-373. M=256:
  599-962 vs W8A8 75 vs RC=4 1028-1069. Clocks swing us.

VERDICT -> k64 blocking with 4-16 dpas is not the 45 us
  kernel. Promote encoding + miss. Next: land ~64 static
  dpas.8x4 like ngen, not another SLM/NT micro.

### 2026-09-02ad - K2 64 static dpas.8x4 unroll

CONTEXT -> k64 blocking left 4/16 dpas. ngen M=1 is 64x
  dpas.8x4. Unroll k64 steps so 2*NT*UNROLL=64.

CONFIG -> sycl+l0, standalone dpas_s8_u64, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. NT=4 U=8 (innerK 512)
  and NT=2 U=16 (innerK 1024). Shapes 4/64/256 x 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_u64.sh 0 4
  gpu-run --card 1 kernels/esimd_dpas/run_u64.sh 1 2
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: both kernels 64x dpas.8x4 rW:b rA:b, GRF
  128. max_abs=0 both cards both NT. M=4: NT=2 53-69 us
  (warm) / NT=4 107-316 (D3hot 316). M=64: 314-570 vs W8A8
  46-49. M=256: 594-1198 vs 75. Warm NT=2 decode is closer
  than k64 92-396; still above 45.

VERDICT -> 64 static dpas.8x4 lights and is not the 45 us
  kernel. Do not freeze 53 us (pad M=4, clocks). Promote
  encoding + miss. Next: ngen wg 8x2 / ska / prefetch, not
  another unroll count.

### 2026-09-02ae - K2 lsc_prefetch_2d on 64-dpas

CONTEXT -> ngen M=1 catalog has ff (null-dest UGM prefetch)
  around 64x dpas.8x4. u64 had the count, no prefetch.

CONFIG -> sycl+l0, standalone dpas_s8_pf, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Same NT=2 U=16 / NT=4
  U=8 as u64, plus lsc_prefetch_2d cached/cached of next k64.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_pf.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_pf.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4 plus null-dest
  load_block2d.ugm.d8 rd:0 (ff). max_abs=0. M=4: NT=2 83-208
  vs u64 53-69 vs W8A8 45. M=64: 229-677 vs u64 314-570.
  M=256: 602-952. Prefetch-before-load taxes decode.

VERDICT -> ff lights and is not a 45 us beat. Warm NT=2
  decode is slower than no-pf u64. Promote encoding + miss.
  Next: overlap prefetch with dpas (ngen order), not more
  pre-load sends.

### 2026-09-02af - K2 prefetch overlapped with dpas

CONTEXT -> pf-before-load taxed decode (83-208 vs u64
  53-69 vs W8A8 45). ngen M=1 is load, then ff, then
  dpas.8x4. Steal that order: prologue ff of k=0, then
  load, first dpas, then lsc_prefetch_2d of next k64.

CONFIG -> sycl+l0, standalone dpas_s8_ov, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Same NT=2 U=16 /
  NT=4 U=8 as u64/pf.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_ov.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_ov.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4 rW:b rA:b, GRF 128, null-dest
  load_block2d.ugm.d8 rd:0. ff count 34 (NT=2) / 18 (NT=4)
  = unroll*2+prologue, vs pf 131/99. max_abs=0 both cards
  both NT. M=4: NT=2 100-264 us (warm 100 at 2083 MHz) /
  NT=4 110-371 (D3hot first 371). M=64: 350-970 vs u64
  314-570 vs W8A8 46-49. M=256: 513-1010 vs 75. Inner IGA
  is prologue ff then load/dpas; later unrolls load-ff-dpas.

VERDICT -> ngen-order ff lights and is not a 45 us beat.
  Warm NT=2 decode (100 us) still loses to no-pf u64 53-69.
  Do not freeze 100 us. Floor stays 45 us. Next: ngen wg
  8x2 / ska double-buffer loads, not another ff placement.

### 2026-09-02ag - K2 A double-buffer software pipeline

CONTEXT -> ov-ff still lost decode to no-pf u64. ngen M=1
  catalog has ska. Steal a real A ping-pong: prologue load
  A[k=0], issue A[k+64] before dpas of current A.

CONFIG -> sycl+l0, standalone dpas_s8_ska, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Same NT=2 U=16 /
  NT=4 U=8 as u64. No extra ff. No SLM.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_ska.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_ska.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4 rW:b rA:b, GRF 128, ff=0.
  A d8 loads 34/18 (prologue + next-k) vs B d8v 64. max_abs=0
  both cards both NT. Warm clocks 2600-2800. M=4: NT=2 79-81
  us / NT=4 92-98 vs u64 NT=2 53-69 vs W8A8 45. M=64: 271-650
  vs u64 314-570. M=256: 965-1100 vs 75. Inner IGA mixes A
  loads with dpas.

VERDICT -> A double-buffer lights and is not a 45 us beat.
  Warm NT=2 decode (79-81 us) still loses to no-pf u64 53-69.
  Do not freeze 79 us. Floor stays 45 us. Next: ngen wg 8x2
  2D launch, not another K-pipe micro.

### 2026-09-02ah - K2 ngen wg 8x2 2D launch

CONTEXT -> u64 1D local=16 is still the decode floor (53-69).
  ngen M=1 is wg 8x2. Steal launch only: nd_range<2> local
  {8,2}=(N,M), same 64 dpas tile, no SLM, no ff.

CONFIG -> sycl+l0, standalone dpas_s8_wg, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. NT=2 U=16 / NT=4 U=8.
  M=4 has m_blocks=1 so half the WG idles.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_wg.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_wg.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4 rW:b rA:b, GRF 128, no barrier,
  no ngen ff (null rd:0 is store+EOT). max_abs=0 both cards
  both NT. M=4: NT=2 69-225 (warm 69 at 1550 MHz after NT=4;
  D3hot first 225) / NT=4 122-316. M=64: 468-963 vs u64
  314-570. M=256: 891-1161 vs 75.

VERDICT -> 8x2 2D launch lights and is not a 45 us beat.
  Warm NT=2 (69 us) ties the slow end of 1D u64, does not
  beat it. Idle M-lanes at M=4 are a decode tax. Do not
  freeze 69 us. Floor stays 45 us. Next: 8x2 along N only
  (no idle) or ngen SLM plus 64 dpas together.

### 2026-09-02ai - K2 wg 8x2 along N (no idle)

CONTEXT -> wg 8x2 (N,M) idled half the WG at M=4 (warm 69).
  Steal: local {8,2} both index N, 16 live N-groups, M is
  group(1). Same 64 dpas tile as u64.

CONFIG -> sycl+l0, standalone dpas_s8_wgn, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. NT=2 U=16 / NT=4 U=8.
  No SLM, no extra ff.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_wgn.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_wgn.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4 rW:b rA:b, GRF 128, no barrier,
  ff_prefetch=0. max_abs=0 both cards both NT. Warm ~2800
  MHz. M=4: NT=2 47-50 us / NT=4 73 vs 1D u64 NT=2 53-69 vs
  (N,M) 8x2 69 vs W8A8 M=1 42-46. M=64: 304-611 vs u64
  314-570. M=256: 1184-1285 vs 75.

VERDICT -> 8x2 along N is a decode win vs 1D u64 (47-50 vs
  53-69) on both cards. It is not a 45 us W8A8 M=1 beat
  (pad M=4). Do not freeze 47 us. New hand-tile decode
  floor is ~47-50 us. Next: ngen SLM plus 64 dpas together.

### 2026-09-02aj - K2 SLM A share plus 64 dpas 8x2-N

CONTEXT -> Old SLM (per-k32 barrier, no unroll) lost 372-463.
  wgn 8x2-N is 47-50 us with 16 N-lanes reloading the same A.
  ngen M=1 bundles SLM + 64 dpas + wg 8x2. Steal that bundle:
  lid0 stores A[k64] to SLM, two barriers per k64.

CONFIG -> sycl+l0, standalone dpas_s8_slm64, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Same 8x2-N as wgn.
  NT=2 U=16 / NT=4 U=8.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_slm64.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_slm64.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4, GRF 128, slm_size 1024,
  barriers in the unroll (64/32). max_abs=0 both cards both
  NT. M=4: NT=2 101-140 us (D3hot first 140) / NT=4 111-114
  vs wgn no-SLM 47-50 vs old SLM 372-463 vs W8A8 45. M=64:
  422-634 vs wgn 304-611. M=256: 899-1184.

VERDICT -> SLM A-broadcast plus 64 dpas lights and is not
  a 45 us beat. Faster than old per-k32 SLM, slower than
  no-SLM wgn at decode. Do not freeze 101 us. Hand floor
  stays 47-50 us. Next: ngen SLM is pack, not A-broadcast.

### 2026-09-02ak - K2 M=1 pad to RC=4 plus ngen SLM read

CONTEXT -> ngen M=1 catalog has 14 SLM ops. IGA of the
  M=1 bin is store/load/fence.slm.d32, not A-tile pack.
  wgn floor 47-50 was pad M=4 vs W8A8 M=1 42-46. Measure
  M=1 zero-padded to RC=4 on the wgn 8x2-N tile.

CONFIG -> sycl+l0, standalone dpas_s8_dec, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. Same 8x2-N 64 dpas
  as wgn. Host oracle on real M rows.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_dec.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_dec.sh 1 4
  # swap NT
  # CPU: grep send.slm dnnl_dump_gpu_gemm_kernel.0.bin.xe2.asm
  ```

RESULT -> ngen M=1 SLM is 14x d32 store/load/fence, not
  block2d A pack. max_abs=0 both cards. Within-run M=1 us
  tracks M=4 (same RC=4 work). Warm D0/2800 card1 NT=2:
  M=1 49 us / M=4 81 vs W8A8 M=1 42-46 vs wgn M=4 47-50.
  D3hot card0 NT=2: M=1 97 / M=4 131. NT=4 M=1 75-113.

VERDICT -> Zero-pad M=1 is closed and is not a 45 us beat.
  It is not 1/4 of M=4. Do not freeze 49 us (clocks 49 vs
  97). Hand floor stays 47-50 us. ngen SLM steal is d32
  ska remainder, not A-pack.

### 2026-09-02al - K2 clock-held M=1 NT=2 both cards

CONTEXT -> M=1 pad was 49 vs 97 across cards. Heat 1024^3
  s8 DPAS (80 timed iters) then time dec NT=2 at D0.

CONFIG -> sycl+l0, dpas_s8 heat then dpas_s8_dec NT=2,
  gpu-run --card N. Sample gt0 cur_freq every 0.2s.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_hold_dec.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_hold_dec.sh 1 2
  ```

RESULT -> After heat cur=1167, not 2800. max_abs=0. Heat
  141 us card0 / 160 us card1. M=1 5120: 61.5 / 65.4 us.
  M=4: 132 / 141. Freq log ~717-750 MHz during the hold.
  End cur=2800 is after the kernels, not during.

VERDICT -> Do not quote 46-48 us from this fire. Clocks
  drooped. Hand floor stays warm wgn M=4 47-50 us. The
  97 us D3hot vs 49 us warm still stands. Next: hold
  clocks during the timed loop, not after a long heat.

### 2026-09-02am - K2 clocks during the timed decode loop

CONTEXT -> Heat-then-decode dropped cur to ~750 MHz
  (61-65 us). Prior 47-50 us wgn quoted start/end
  cur=2800, not act during the event. Hold clocks
  DURING the timed loop. min_freq is root-only.

CONFIG -> sycl+l0, standalone dpas_s8_clk (wgn 8x2-N
  64 dpas, pad M to RC=4). No 1024^3 heat. gpu-run
  --card N. Sample .freq every 50 ms. Arms: prime=0/3
  with in-loop sysfs (duty-cycle miss), then spin=4000
  batched same-kernel occupancy, no in-loop sysfs.
  Event us + per-iter wait host us + pipelined host us
  (matched to W8A8 us_bench). NT=2. Both cards.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_clk.sh 0 2 0 4000
  gpu-run --card 1 kernels/esimd_dpas/run_clk.sh 1 2 0 4000
  ```

RESULT -> In-loop sysfs: M=1 77-180 us at 467-1250 MHz.
  Batched spin holds act/cur=2800 at timed_begin/end.
  max_abs=0. M=1 5120: event 35.96/35.80 us, wait-host
  50.4/50.0, pipe-host 36.44/36.43. M=4 tracks M=1
  (36.15/36.04 event, 36.70/36.65 pipe). 40-iter event
  min-max 34.4-37.2. W8A8 K4 host 42.1/46.1 includes
  scales; this stores s32.

VERDICT -> Long heat dumps clocks. Per-iter sysfs waits
  also dump duty cycle. Batched same-kernel spin holds
  2800. New hand decode floor is ~36 us event / ~36.4 us
  pipelined host at 2800, not 47-50. Do not call a
  serving beat of 42-46 (no scale epilogue). Next: d32
  ska flag broadcast, then scale to match W8A8.

### 2026-09-02an - K2 ngen d32 flag broadcast

CONTEXT -> ngen M=1 SLM is 14x store/load/fence.slm.d32
  plus send.gtwy barrier, not A-pack. Steal the prologue
  once per launch on the wgn 64-dpas tile.

CONFIG -> sycl+l0, standalone dpas_s8_d32, icpx 2026.1.1
  AOT intel_gpu_bmg_g31, gpu-run --card N. NT=2 U=16 /
  NT=4 U=8. Token=0 kept live in k0. No 2800 spin.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_d32.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_d32.sh 1 4
  # swap NT
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4, store.slm.d32,
  fence.slm.none.group, send.gtwy barrier, load.slm.d32.
  GRF 128, slm_size 1024, barrier_count 1. max_abs=0.
  M=1: NT=2 90-194 / NT=4 91-104 vs held-clock wgn 36
  vs no-hold wgn 47-50. M=4 133-197. M=64 552-1143.

VERDICT -> The ngen d32 encoding lights and is not a
  36 us or 45 us beat. Dummy flag+barrier is a decode
  tax. Hand floor stays no-SLM 36 us at 2800. Next:
  scale epilogue to match W8A8's 42-46 host contract.

### 2026-09-02ao - K2 in-kernel GEMM repeats at 2800

CONTEXT -> Batched one-shot spin (am) is ~36 us at 2800.
  Confirm with R fused GEMMs in one launch (zero acc,
  store last) so sysfs samples land inside the kernel.

CONFIG -> sycl+l0, standalone dpas_s8_rep, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card N. NT=2. M=1 R=4096,
  M=4 R=2048.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_rep.sh 0 2
  gpu-run --card 1 kernels/esimd_dpas/run_rep.sh 1 2
  ```

RESULT -> max_abs=0 both cards. Freq: 8 samples act=2800
  cur=2800 each card during the kernel. M=1 us_per 34.46 /
  34.36. M=4 us_per 34.51 / 34.47. Event ~141 ms / 4096.
  Matches clk min 34.4 us. One-shot held floor stays 36 us.

VERDICT -> Fused-repeat body is 34 us at 2800, same band
  as clk min. Not a serving one-shot. Scale epilogue is
  still the W8A8-contract gap.

### 2026-09-02ap - K2 W8A8 scale epilogue to f16

CONTEXT -> Raw s32 at 2800 is 36 us. W8A8 stores f16
  with A_scale (M) * B_scale (N). ngen epilogue is
  mov :hf then store_block2d.ugm.d16. Steal that
  contract on the wgn tile.

CONFIG -> sycl+l0, standalone dpas_s8_sc, icpx 2026.1.1
  AOT intel_gpu_bmg_g31, gpu-run --card N. NT=2. Fill
  s8 [-64,64], a_s=b_s=0.02, out f16. Spin=4000 then
  40 timed. Control: int8_gemm_w8a8 M=1 after M=64 heat,
  same image, warmup 30 iters 40, both cards.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc.sh 0 2 4000
  gpu-run --card 1 kernels/esimd_dpas/run_sc.sh 1 2 4000
  gpu-run --card N kernels/w8_compare/run_w8_m1hold.sh N
  ```

RESULT -> max_abs=0 cosine=1.0 both cards. timed
  act/cur=2800. M=1: event 33.29/33.39 us, pipe_host
  34.12/34.34. M=4 tracks (33.47/33.45 event, 34.11/
  34.39 pipe). ocloc: 64x dpas.8x4, store_block2d.ugm.d16.
  W8A8 M=1 after M=64 heat: 44.55/43.82 us host
  pipelined (cosine 1.0). K4 first-shape floor was
  42.1/46.1. Cold first-shape W8A8 this session was
  79/85 (not the floor).

VERDICT -> Scale-to-f16 on the wgn tile is a real us
  beat of same-session W8A8 M=1 (34 vs 44 pipe host)
  at held 2800, same s8/scale/f16 contract, both cards.
  Do not quote tok/s. Pad M=1 still does RC=4 work.
  Next: M=64 GRF256/SLM, or fuse K5 producer into this.

### 2026-09-02aq - K2 scale-to-f16 M=64 at held clock

CONTEXT -> ap beat W8A8 at decode M=1 (34 vs 44 pipe).
  Same RC=4 wgn tile at prefill M=64 is 16 M-blocks.
  Napkin: 16x the M=1 pad work (~528 us) vs W8A8
  M=64 46-49. Occupancy may cut that; ngen M=64 is
  RC=8 GRF256 64x dpas.8x8, not this tile.

CONFIG -> backend sycl+l0, standalone dpas_s8_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31, gpu-run
  --card N. NT=2 spin=4000 warmup=20 iters=20.
  Fill s8 [-64,64], a_s=b_s=0.02, out f16.
  Control: same-day int8_gemm_w8a8 M=64 from
  results/k2/w8a8_hold_card{0,1}.txt (46.17/46.45).

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc_m64.sh 0 2 4000
  gpu-run --card 1 kernels/esimd_dpas/run_sc_m64.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0 both cards. timed
  cur=2800, act 2683-2750, throttle=1 (sysfs during
  the 4000-spin and 20+20 timed). M=64 5120:
  event 246.88/244.83 us, pipe_host 247.16/242.53,
  median 246.77/242.14, min-max 239.8-255.7 /
  235.4-273.8. W8A8 M=64 same-day hold 46.17/46.45.
  ~5.3x the incumbent. ~7.4x this tile's M=1 33 us,
  not 16x (M=1 is 10 WGs; M=64 is 160 WGs).

VERDICT -> RC=4 scale-to-f16 does not beat W8A8 at
  M=64 (245 vs 46 us), both cards, clocks explained.
  Occupancy helped vs the 528 us napkin. Do not
  quote TOPS% (throttle=1, act ~2.7 GHz). Next steal
  is ngen M=64 RC=8/GRF256/SLM on this contract, or
  fuse K5 into the M=1 GEMM that already wins.

### 2026-09-02ar - K2 RC=8 dpas.8x8 f16 at M=64

CONTEXT -> RC=4 scale-to-f16 M=64 is 245 vs W8A8 46.
  ngen M=64 is 64x dpas.8x8 RC=8 grf256. Steal RC=8
  + 64 unroll on the f16 contract. Request GRF256.

CONFIG -> sycl+l0, standalone dpas_s8_sc8, icpx 2026.1.1
  AOT intel_gpu_bmg_g31, gpu-run --card N. NT=2 U=16
  (64 dpas). intelex::grf_size<256>. spin=512 warmup=10
  iters=20. Fill [-64,64] scales 0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8 rW:b rA:b, store_block2d
  d16, grf_count 128 (request refused). cosine=1.0
  max_abs=0. timed cur=2800 act~2767-2783 throttle=1.
  M=64: event 120.73/119.90 us, pipe_host 119.63/121.01
  vs RC=4 245 vs W8A8 46.17/46.45.

VERDICT -> RC=8 lights and is ~2x RC=4 (half the
  M-blocks), not a 46 us beat. GRF256 still refused.
  Next: ngen wg 4x2x4 / SLM pack, not another RC.

### 2026-09-02as - K2 ngen wg 4x2x4 + SLM A pack M=64

CONTEXT -> RC=8 8x2-N f16 is 120 us vs W8A8 46.
  ngen M=64 catalog is wg 4x2x4 kr grf256 k64 + 53
  SLM, 64x dpas.8x8. Steal 32-thread 4x2x4, 4x RC=8
  (32 rows/thread), SLM A share per k64.

CONFIG -> sycl+l0, standalone dpas_s8_sc84, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=4 (64 dpas). slm 4096. grf_size<256>.
  spin=512 warmup=10 iters=20. Fill [-64,64]
  scales 0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc84.sh 0 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc84.sh 1 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, slm_size 4096,
  barrier_count 1, 32 store.slm.d64x32t + 32 load
  + 8 fence.slm + 8 barriers, store_block2d.d16,
  grf_count 128 (request refused). cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0.
  M=64: event 135.17/135.94 us, pipe_host
  136.10/136.00 vs sc8 120 vs W8A8 46.17/46.45.

VERDICT -> 4x2x4 + SLM A pack lights and is
  numeric-closed, ~13% slower than no-SLM sc8
  at the same 2800. Not a 46 us beat. SLM
  A-share plus barrier-per-k64 is a tax on this
  tile (same class as decode slm64). Next: fuse
  K5 into the M=1 GEMM that already wins, or
  steal ngen kr/double-buffer without that barrier.

### 2026-09-02at - K5 scalar RMSNorm-quant inside GEMM

CONTEXT -> GEMM-only f16 is 34 us. K5 WG-256 is 7-36 us
  extra launch. Steal: one ESIMD launch, each WG
  RMSNorms f16 A, quant to s8, 64 dpas.8x4, scale f16.

CONFIG -> sycl+l0, standalone dpas_s8_fuse, icpx 2026.1.1
  AOT intel_gpu_bmg_g31, gpu-run --card N. NT=2 spin=4000.
  Scalar rint/sqrt in the k-loop. Pad M=1 to RC=4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_fuse.sh 0 2 4000
  gpu-run --card 1 kernels/esimd_dpas/run_fuse.sh 1 2 4000
  ```

RESULT -> timed act=cur=2800. M=1 event 314.2/312.9 us,
  pipe 312.6/313.5 vs GEMM-only 34 vs two-launch ~41-70.
  cosine 0.73 max_abs 50, not closed (ok=0). Same class
  as K6 scalar in-register LUT (2316 us).

VERDICT -> Naive scalar quant in the DPAS loop is not
  a launch win. ~9x the 34 us GEMM and not numeric-
  closed. Keep two-launch K5+GEMM. Next: vectorize the
  quant, or leave producer as its own WG-256 kernel.

### 2026-09-02au - K5 vectorized RMSNorm-quant inside GEMM

CONTEXT -> Scalar fuse (at) was 313 us, cosine 0.73.
  f16 lsc_load_2d width/pitch was passed as elements
  (API is bytes), so K>~surface/2 was OOB. Inner loop
  was scalar rint/sqrt. Steal: simd convert/reduce/
  hmax/rnde, pitch in bytes, same 64 dpas.8x4 f16.

CONFIG -> sycl+l0, standalone dpas_s8_fusev, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 spin=4000 warmup=50 iters=40. Fill f16 A, s8 B
  [-64,64], b_scale=0.02, out f16. Pad M=1 to RC=4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_fusev.sh 0 2 4000
  gpu-run --card 1 kernels/esimd_dpas/run_fusev.sh 1 2 4000
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x4, rnde (32|M0) x128,
  math.rsqt x4 + math.inv, load_block2d.d16 A /
  d8v B, store_block2d.d16, grf 128, no SLM.
  cosine=1.0 max_abs=0.015625 (1 f16 ulp) both
  cards. timed act=cur=2800 throttle=0.
  M=1: event 71.67/71.78 us, pipe_host 72.47/72.28.
  M=4 tracks (72.44/72.32 event, 72.80/72.76 pipe).
  vs scalar 313 vs GEMM-only 34 vs two-launch ~41-70.

VERDICT -> Vectorizing + byte pitch closes the fuse
  and is ~4.3x the scalar arm. It is not a 34 us
  GEMM beat (~2.1x) and loses to two-launch when the
  WG-256 producer is ~7 us. Every GEMM thread still
  re-reads f16 A. Keep two-launch K5+GEMM as the
  decode path. Next: ngen kr/double-buffer without
  barrier-per-k64, or stop re-reading A in the GEMM.

### 2026-09-02av - K2 sc8 A double-buffer M=64 no SLM

CONTEXT -> sc8 RC=8 8x2-N f16 is 120 us vs W8A8 46.
  sc84 4x2x4+SLM is 136 (barrier-per-k64 tax). ngen
  M=64 is kr xaf: A/B software pipeline, not SLM
  share. Steal ska-style A ping-pong on the sc8 tile.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=16 (64 dpas). prologue A[k=0], issue A[k+64]
  before dpas. No SLM. grf_size<256>. spin=512
  warmup=10 iters=20. Fill [-64,64] scales 0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8db.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8db.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, store_block2d.d16,
  grf_count 128, no slm_size, no barrier_count.
  cosine=1.0 max_abs=0. timed act=2783 cur=2800
  throttle=1. M=64: event 97.93/103.40 us, pipe_host
  96.64/100.44 vs sc8 120 vs sc84 136 vs W8A8 46.

VERDICT -> A double-buffer without SLM is a real
  ~17-19% us win vs sc8 at the same 2800/throttle=1.
  Not a 46 us beat (~2.1x). Do not put A in SLM to
  chase this. Next: ngen 4-acc M-tile (32 rows/
  thread) without SLM, or B pipeline + ca.ca.

### 2026-09-02aw - K2 4-acc M-tile no SLM M=64

CONTEXT -> sc8db A-db is 97-100 us vs W8A8 46. ngen
  M=64 keeps 4 acc along M (32 rows/thread) and
  reuses B. sc84 did that with SLM and lost. Steal
  4x RC=8, own A, no SLM, A ping-pong, NT=2 U=4
  (64 dpas).

CONFIG -> sycl+l0, standalone dpas_s8_sc8m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  32 rows/thread, 8x2 along N, no SLM. grf_size<256>.
  spin=512 warmup=10 iters=20. Fill [-64,64] scales
  0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8m4.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8m4.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8 {Atomic}, store_block2d
  d16, grf_count 128, no slm_size. cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0.
  M=64: event 118.99/118.75 us, pipe_host
  119.83/119.69 vs sc8db 97-100 vs sc8 120 vs
  sc84 136 vs W8A8 46.

VERDICT -> 4-acc without SLM lights and matches sc8
  (~120), not sc8db (~98). Fewer WGs (32-row tiles)
  gave up occupancy; B reuse did not pay. Keep the
  8-row A-db tile. Next: B pipeline + ca.ca on
  sc8db, or ngen null-dest prefetch.

### 2026-09-02ax - K2 sc8db B pipeline + ca.ca M=64

CONTEXT -> sc8db A-db is 97-100 us vs W8A8 46. ngen
  M=64 loads A/B ca.ca and keeps B in GRF (xaf).
  Steal: B ping-pong + cached/cached on the 8-row
  A-db tile. No SLM, no prefetch.

CONFIG -> sycl+l0, standalone dpas_s8_sc8bp, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=16 (64 dpas). A+B prologue, issue next
  k64 before dpas. lsc_load_2d L1/L2 cached.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8bp.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8bp.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64-68x dpas.8x8, load_block2d
  .ca.ca, store_block2d.d16, grf_count 128, no
  slm_size. cosine=1.0 max_abs=0. timed act=2783
  cur=2800 throttle=1. M=64: event 106.16/105.70
  us, pipe_host 105.46/106.51 vs sc8db 97-100 vs
  W8A8 46.

VERDICT -> B-db + ca.ca lights and is ~6-9% slower
  than A-db only at the same 2800/throttle=1. Extra
  B GRF is a tax. Keep sc8db as the M=64 hand floor.
  Next: ngen null-dest prefetch on sc8db (no extra
  GRF), or stop M=64 load-path chasing.

### 2026-09-02ay - K2 sc8db + ngen null-dest prefetch M=64

CONTEXT -> sc8db A-db is 97-100 us vs W8A8 46. B-db
  + ca.ca was a GRF tax. ngen M=64 issues
  load_block2d to null (ff) then real loads. Steal:
  lsc_prefetch_2d cached/cached of next k64 A and B
  on the A-db tile. No extra B GRF.

CONFIG -> sycl+l0, standalone dpas_s8_sc8ff, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=16. A ping-pong plus prefetch next k64.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8ff.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8ff.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, 32x null dest
  load_block2d.d8.ca.ca (rd:0), grf_count 128, no
  slm_size. cosine=1.0 max_abs=0. timed act~2770
  cur=2800 throttle=1. M=64: event 127.04/127.45
  us, pipe_host 125.83/128.07 vs sc8db 97-100 vs
  W8A8 46.

VERDICT -> ngen ff prefetch lights and is ~30%
  slower than A-db only. Extra null sends are a
  tax on this tile. Stop M=64 load-path chasing;
  sc8db 97-100 us is the hand floor (~2.1x W8A8).
  Next: ngen M=256 k128, or K5 producer that does
  not re-read A.

### 2026-09-02az - K2 k128 A-db M=256

CONTEXT -> M=64 load-path closed at sc8db 97-100 us.
  ngen M=256 is k128, 384x dpas.8x8, no SLM, W8A8
  75 us. Steal k128 (4x k32) + A ping-pong on the
  8-row tile. Napkin: 4x M=64 ~392 us.

CONFIG -> sycl+l0, standalone dpas_s8_sc8k128, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=8 (64 dpas), k128, A-db, no SLM.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8k128.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8k128.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, store_block2d.d16,
  grf_count 128, no slm_size. cosine=1.0 max_abs=0.
  timed act=2550-2600 cur=2800 throttle=1.
  M=256: event 447.92/431.53 us, pipe_host
  442.58/439.44 vs napkin 392 vs W8A8 75.

VERDICT -> k128 A-db lights and tracks ~4.5x the
  M=64 A-db floor (clocks lower, throttle=1). Not
  a 75 us beat (~5.9x). ngen's 384 unroll / wg 4x8
  / GRF256 still open. Next: K5 producer that does
  not re-read A, or ngen wg 4x8.

### 2026-09-02ba - K5 WG-256 producer then s8 GEMM

CONTEXT -> fusev one-launch was 72 us because every
  GEMM thread re-scanned f16 A. GEMM-only is 34 us.
  K5 WG-256 extra 7-36 us. Steal: producer writes
  s8 A + scale once, then the 64 dpas.8x4 GEMM
  loads s8. Two kernels, one in-order queue.

CONFIG -> sycl+l0, standalone dpas_s8_prod, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 spin=4000 warmup=50 iters=40. f16 A, s8 B
  [-64,64], b_scale=0.02, out f16. Pad M=1 to RC=4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_prod.sh 0 2 4000
  gpu-run --card 1 kernels/esimd_dpas/run_prod.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0.015625. timed
  act=cur=2800 throttle=0. M=1: prod 10.46/10.40
  us, gemm 33.06/33.20, pair_event 43.66/43.75,
  pipe_host 44.30/44.43. M=4 tracks. vs fusev 72
  vs GEMM-only 34 vs two-launch 41-70.

VERDICT -> Producer that does not re-read A is a
  real us win vs fusev (44 vs 72) and matches the
  best two-launch. Extra 10 us over GEMM-only 34.
  Keep two-kernel producer+GEMM as the decode
  quant path. Next: ngen wg 4x8 at M=256, or stop
  decode-quant chasing.

### 2026-09-02bb - K2 ngen wg 4x8 k128 M=256

CONTEXT -> k128 A-db 8x2-along-N is 440 us at M=256
  vs W8A8 75. ngen is wg 4x8 k128. Steal launch
  geometry: 4 along N, 8 along M. Same 8-row tile,
  k128 A-db, 64 dpas, no SLM.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=8, wg 4x8 NxM. grf_size<256>. spin=512
  warmup=10 iters=20. Fill [-64,64] scales 0.02
  out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, store_block2d.d16,
  grf_count 128, no slm_size. cosine=1.0 max_abs=0.
  timed act=2733-2750 cur=2800 throttle=1.
  M=256: event 228.53/230.15 us, pipe_host
  229.54/227.72 vs k128 8x2-N 440 vs W8A8 75.

VERDICT -> wg 4x8 (M on Y) is a real ~1.9x vs
  packing local dims along N. Not a 75 us beat
  (~3.0x). Geometry matters more than k128 here.
  Next: steal 4x8 on the M=64 A-db tile, or 384
  dpas unroll.

### 2026-09-02bc - K2 wg 4x8 on M=64 A-db

CONTEXT -> sc8db 8x2-along-N A-db is 97-100 us vs
  W8A8 46. wg 4x8 (M on Y) was ~1.9x at M=256.
  Steal that geometry on the M=64 A-db tile.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=16 (64 dpas), wg 4x8 NxM, k64 A-db, no SLM.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=64 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8db48.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8db48.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8, store_block2d.d16,
  grf_count 128, no slm_size. cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64: event 74.00/74.35 us, pipe_host 75.49/75.61
  vs sc8db 97-100 vs W8A8 46.17/46.45.

VERDICT -> wg 4x8 is a real ~1.3x vs 8x2-N A-db
  at M=64, new hand floor 75 us (~1.63x W8A8).
  Throttle=0 at 2800. Geometry is the M=64 steal.
  Next: 384 dpas unroll at M=256, or stop M=64
  geometry chasing.

### 2026-09-02bd - K2 4-acc + wg 4x8 k128 M=256

CONTEXT -> wg 4x8 8-row k128 is 228 us at M=256 vs
  W8A8 75. ngen is 384x dpas.8x8. Steal 4x RC=8
  (32 rows/thread) on that geometry: 256 dpas,
  B reuse across M-tiles, no A-db.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=8, 4 M-tiles, wg 4x8 NxM, k128, no SLM.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48m4.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48m4.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 256x dpas.8x8 {Atomic}, store_block2d
  d16, grf_count 128, no slm_size. cosine=1.0
  max_abs=0. timed act=2767 cur=2800 throttle=1.
  M=256: event 126.89/127.89 us, pipe_host
  128.39/128.57 vs 4x8 8-row 228 vs W8A8 75.

VERDICT -> 4-acc on wg 4x8 is a real ~1.8x vs 8-row
  at M=256. New hand floor 128 us (~1.7x W8A8).
  256 dpas not 384. Next: 384 unroll, or 4-acc on
  the M=64 4x8 tile.

### 2026-09-02be - K2 384 dpas 6-acc wg 4x8 M=256

CONTEXT -> 4-acc wg 4x8 k128 is 128 us at M=256 vs
  W8A8 75. ngen is 384x dpas.8x8 (16 acc = 4M x 4N
  x 24 k32; inner K=768 does not divide 5120).
  Steal the 384 count that does: 6x RC=8 (48
  rows/thread) NT=2 U=8 -> 384 dpas. Pad M 256->288.
  No A-db. Rank pipe_host vs 128 and 75.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m6, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=8, 6 M-tiles, wg 4x8 NxM, k128, no SLM.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48m6.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48m6.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 384x dpas.8x8 (NT=2 192x {Atomic}),
  store_block2d d16, grf_count 128, no slm_size.
  IGC spill 768 B (NT=2). cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=256: event 209.48/209.60 us, pipe_host
  210.06/209.95 vs 4-acc 128 vs W8A8 75.

VERDICT -> 384-count via 6-acc is a real loss vs
  4-acc (210 vs 128, ~1.64x). This arm is ~2.8x
  W8A8. Floor stays 4-acc 128 us (~1.7x W8A8).
  Matching ngen's dpas *count* is not the 75 us
  kernel. Next: A-db on 4-acc M=256, or 4-acc on
  M=64 wg 4x2 (4x8 would idle).

### 2026-09-02bf - K2 k32 A-db on 4-acc M=256

CONTEXT -> 4-acc wg 4x8 k128 is 128 us at M=256 vs
  W8A8 75. A-db won at M=64 (75 vs 97). 384-count
  6-acc lost (210). Steal ska-style k32 A ping-pong
  on the 4-acc tile: prologue A[k=0], issue A[k+32]
  before dpas. Keep 256 dpas. Not full k128 A-db.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4db, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=8, 4 M-tiles, wg 4x8 NxM, k128, k32 A-db,
  no SLM. grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48m4db.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48m4db.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 256x dpas.8x8 (192x {Atomic}),
  store_block2d d16, grf_count 128, no slm_size.
  NT=2 no spill (NT=4 4608 B). cosine=1.0 max_abs=0.
  timed act=2783 cur=2800 throttle=1.
  M=256: event 135.52/135.86 us, pipe_host
  135.13/134.88 vs 4-acc no A-db 128 vs W8A8 75.

VERDICT -> k32 A-db on 4-acc is a real small tax
  (~135 vs 128, ~1.05x). Floor stays 128 us
  (~1.7x W8A8). A-db that won at M=64 does not
  transfer to this M=256 tile. Next: 4-acc on
  M=64 wg 4x2.

### 2026-09-02bg - K2 4-acc wg 4x2 M on Y M=64

CONTEXT -> sc8m4 4-acc wg 8x2-along-N is 120 us vs
  wg 4x8 A-db 75 vs W8A8 46. M-on-Y won at M=256
  4-acc and at M=64 8-row. Steal wg 4 along N x 2
  along M on the 4-acc tile. 32 rows * 2 = 64, no
  idle. Same 64 dpas, k64 A-db, no SLM.

CONFIG -> sycl+l0, standalone dpas_s8_sc8m42, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=4, 4 M-tiles, wg 4x2 NxM, k64 A-db, no SLM.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill [-64,64] scales 0.02 out f16. M=64 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8m42.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8m42.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8 (33x {Atomic}),
  store_block2d d16, grf_count 128, no slm_size.
  IGC spill 1792 B (NT=2). cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64: event 114.71/115.12 us, pipe_host
  115.33/115.97 vs sc8m4 120 vs 4x8 A-db 75 vs
  W8A8 46.

VERDICT -> M-on-Y is a real small win vs 8x2-N
  (~115 vs 120, ~1.04x). Floor stays 8-row 4x8
  A-db 75 us (~1.53x this arm, ~1.63x W8A8).
  8-thread WG is not the 75 us kernel. Next:
  4-acc wg 4x2x4 no SLM (32 threads), or stop
  M=64 4-acc chasing.

### 2026-09-02bh - K2 4-acc wg 4x2x4 no SLM M=64

CONTEXT -> 4-acc wg 4x2 (8 thr) is 115 us vs 4x8
  A-db 75 vs W8A8 46. sc84 4x2x4+SLM was 136.
  ngen M=64 is wg 4x2x4. Steal 32-thread 4x2x4
  without SLM on the same 4-acc k64 A-db tile.

CONFIG -> sycl+l0, standalone dpas_s8_sc8m424, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=4, 4 M-tiles, wg 4x2x4 NxMxN, k64 A-db,
  no SLM. grf_size<256>. spin=512 warmup=10
  iters=20. Fill [-64,64] scales 0.02 out f16.
  M=64 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8m424.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_sc8m424.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 64x dpas.8x8 (33x {Atomic}),
  store_block2d d16, grf_count 128, no slm_size.
  IGC spill 1792 B (NT=2). cosine=1.0 max_abs=0.
  timed act=cur=2800 throttle=0.
  M=64: event 131.91/131.82 us, pipe_host
  132.62/132.94 vs 4x2 115 vs sc84 SLM 136 vs
  4x8 A-db 75 vs W8A8 46.

VERDICT -> 32-thread 4x2x4 no SLM is a real loss
  vs 8-thread 4x2 (~133 vs 115, ~1.15x). Slightly
  beats SLM 4x2x4 (136). Occupancy was not the
  leftover. Stop M=64 4-acc chasing. Floor stays
  75 us. Next: s4 on the M=64 4x8 A-db tile.

### 2026-09-02bi - K2 s4 on M=64 4x8 A-db

CONTEXT -> s8 wg 4x8 A-db is 75 us at M=64 vs
  W8A8 46. s4 was 1.49x s8 at 1024^3 / ~583 MHz,
  not 2x. New dtype on the winning tile: packed
  s4 A/B, one dpas per k64, wg 4x8 A-db. Both-card
  this fire.

CONFIG -> sycl+l0, standalone dpas_s4_db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card N.
  NT=2 U=16 (32 s4 dpas), wg 4x8 NxM, k64 A-db,
  no SLM, pack=2 along K. grf_size<256>. spin=512
  warmup=10 iters=20. Fill s4 [-8,7] scales 0.02
  out f16. M=64 5120. Never E2M1 bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_db48.sh 0 2 512
  gpu-run --card 1 kernels/esimd_dpas/run_s4_db48.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 32x dpas.8x8 rW:s4 rA:s4,
  store_block2d d16, grf_count 128, no slm_size.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64: event 33.010/33.042 us, pipe_host
  33.608/33.735 vs s8 4x8 A-db 75 vs W8A8 46.

VERDICT -> s4 on this tile is a real ~2.24x vs
  s8 75 us and under W8A8 46 us in wall time.
  New s4 hand floor 33.6 us at 2800. Different
  dtype than W8A8; INT8 s8 floor stays 75. The
  1.49x at 1024^3 was not this shape. Rank us.
  Do not quote tok/s. Next: split s4 M=256 4-acc
  and s4 M=1 decode, one arm per card.

### 2026-09-02bj - K2 s4 4-acc wg 4x8 M=256 card0

CONTEXT -> s8 4-acc wg 4x8 is 128 us at M=256 vs
  W8A8 75. s4 4x8 A-db is 33.6 us at M=64
  (~2.24x s8). Steal native s4 onto the 4-acc
  tile. One-card schedule steal; sibling later
  if this is a new floor.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  NT=2 U=8 (128 s4 dpas), 4 M-tiles, wg 4x8 NxM,
  k128, two s4 dpas per k128, no SLM, pack=2.
  grf_size<256>. spin=512 warmup=10 iters=20.
  Fill s4 [-8,7] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_w48m4.sh 0 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 128x dpas.8x8 rW:s4 rA:s4,
  store_block2d d16, grf_count 128, no slm_size.
  NT=2 no spill (NT=4 spill 3584 B). cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0.
  M=256 card0: event 48.266 us, pipe_host 48.650
  vs s8 4-acc 128 vs W8A8 75.

VERDICT -> One-card s4 4-acc is a real ~2.63x vs
  s8 128 us and under W8A8 75 us in wall time.
  Do not promote 48.7 us as a floor until card1
  runs it. Rank us. Next: sibling swap.

### 2026-09-02bk - K2 s4 RC=4 8x2-N M=1 card1

CONTEXT -> s8 RC=4 8x2-N scale-to-f16 is 34 us
  at M=1 vs W8A8 44. s4 4x8 A-db is 33.6 us at
  M=64. Steal native s4 onto the decode tile.
  One-card; sibling later if a new floor.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=16 (32 s4 dpas.8x4), wg 8x2 along N,
  pad M to RC=4, pack=2, no SLM. spin=4000
  warmup=50 iters=40. Fill s4 [-8,7] scales 0.02
  out f16. M=1 and M=4 5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_sc.sh 1 2 4000
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 32x dpas.8x4 rW:s4 rA:s4,
  store_block2d d16, grf_count 128, no slm_size.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 15.958 us, pipe_host 16.576
  vs s8 34 vs W8A8 44. M=4 tracks (pipe 16.346).

VERDICT -> One-card s4 decode is a real ~2.05x
  vs s8 34 us and under W8A8 44 us. Pad still
  does RC=4 work. Do not promote 16.6 us as a
  floor until card0 runs it. Rank us. Next:
  sibling swap of bj and bk.

### 2026-09-02bl - K2 s4 RC=4 M=1 sibling card0

CONTEXT -> card1 s4 RC=4 8x2-N was 16.6 us at
  M=1 2800, numeric closed. Sibling swap to
  close the decode floor.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=16 pack=2 spin=4000 as bk.
  Fill s4 [-8,7] scales 0.02 out f16. M=1 and
  M=4 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_sc.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 16.013 us, pipe_host 16.411
  vs card1 16.576 vs s8 34 vs W8A8 44.
  M=4 pipe 16.345 vs card1 16.346. Spread ~1%.

VERDICT -> Sibling matches. New s4 decode floor
  16.5 us at 2800 both cards. ~2.05x s8 34.
  Pad still RC=4 work. Rank us. Do not quote
  tok/s.

### 2026-09-02bm - K2 s4 4-acc M=256 sibling card1

CONTEXT -> card0 s4 4-acc wg 4x8 was 48.7 us at
  M=256 2800, numeric closed. Sibling swap to
  close the M=256 s4 floor.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  Same NT=2 U=8 pack=2 spin=512 as bj.
  Fill s4 [-8,7] scales 0.02 out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_w48m4.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=256 card1: event 48.724 us, pipe_host 48.471
  vs card0 48.650 vs s8 128 vs W8A8 75.
  Spread ~0.4%.

VERDICT -> Sibling matches. New s4 M=256 floor
  48.6 us at 2800 both cards. ~2.63x s8 128
  and under W8A8 75. Different dtype than W8A8.
  Rank us. Next: split s4 A-db on this 4-acc
  tile vs s4 decode N=17408.

### 2026-09-02bn - K2 s4 k64 A-db on 4-acc M=256 card0

CONTEXT -> s4 4-acc no A-db is 48.6 us at M=256.
  s8 k32 A-db on this tile was a tax (135 vs
  128). Steal k64 A ping-pong on s4. One-card.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4db,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. NT=2 U=8, 4 M-tiles, wg 4x8,
  k128, k64 A-db, pack=2, no SLM. spin=512
  warmup=10 iters=20. Fill s4 [-8,7] scales 0.02
  out f16. M=256 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_w48m4db.sh 0 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: 128x dpas.8x8 rW:s4 rA:s4,
  store_block2d d16, grf_count 128, no slm_size.
  NT=2 no spill (NT=4 spill 4608 B). cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0.
  M=256 card0: event 50.740 us, pipe_host 51.937
  vs no A-db 48.6 vs W8A8 75.

VERDICT -> k64 A-db on s4 4-acc is a real tax
  (~51.9 vs 48.6, ~1.07x). Same miss as s8.
  Floor stays 48.6 us. Rank us.

### 2026-09-02bo - K2 s4 decode N=17408 card1

CONTEXT -> s4 RC=4 8x2-N is 16.5 us at M=1
  N=5120. Qwen3.8-ish FFN-up is N=17408.
  Napkin N-linear 16.5*17408/5120 ~56 us.
  Same binary, one-card shape steal.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=16 pack=2 spin=4000. Fill s4 [-8,7]
  scales 0.02 out f16. M=1 and M=4, N=17408
  K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_sc_wide.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 29.039 us, pipe_host 29.754
  vs N=5120 16.5 vs napkin 56. M=4 tracks
  (pipe 29.739). Ratio 29.8/16.5 ~1.80x, not
  3.40x.

VERDICT -> Wide N is a real 29.8 us at 2800,
  better than N-linear. One-card. Do not freeze
  as a floor until card0. Rank us. Next: sibling
  N=17408 vs s4 M=1 K=17408 (down-proj).

### 2026-09-02bp - K2 s4 decode N=17408 sibling card0

CONTEXT -> card1 s4 RC=4 N=17408 was 29.8 us at
  M=1 2800, numeric closed. Sibling swap to
  close the wide-N floor.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=16 pack=2 spin=4000 as bo.
  Fill s4 [-8,7] scales 0.02 out f16. M=1 and
  M=4, N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_sc_wide.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 29.552 us, pipe_host 29.235
  vs card1 29.754 vs N=5120 16.5 vs napkin 56.
  M=4 pipe 29.537 vs card1 29.739. Spread ~1.8%.

VERDICT -> Sibling matches. New s4 wide-N floor
  29.5 us at 2800 both cards. ~1.80x N=5120,
  not 3.4x. Rank us.

### 2026-09-02bq - K2 s4 decode K=17408 card1

CONTEXT -> s4 M=1 N=5120 K=5120 is 16.5 us.
  N=17408 K=5120 is 29.5 us. FFN-down is
  N=5120 K=17408. Same B bytes as wide N.
  Napkin K-linear 16.5*17408/5120 ~56 us.
  One-card shape steal.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=16 pack=2 spin=4000. Fill s4 [-8,7]
  scales 0.02 out f16. M=1 and M=4, N=5120
  K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_sc_k17408.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 52.922 us, pipe_host 53.367
  vs N=5120 16.5 vs N=17408 29.5 vs napkin 56.
  M=4 tracks (pipe 53.303). Ratio 53.4/16.5
  ~3.24x, near K-linear 3.40x.

VERDICT -> Down-proj K=17408 is a real 53.4 us,
  near K-linear, slower than wide-N at the same
  B bytes (29.5). One-card. Do not freeze until
  card0. Rank us. Next: sibling K=17408 vs s4
  M=64 N=17408.

### 2026-09-02br - K2 s4 decode K=17408 sibling card0

CONTEXT -> card1 s4 RC=4 K=17408 was 53.4 us at
  M=1 2800, numeric closed. Sibling swap to
  close the down-proj floor.

CONFIG -> sycl+l0, standalone dpas_s4_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=16 pack=2 spin=4000 as bq.
  Fill s4 [-8,7] scales 0.02 out f16. M=1 and
  M=4, N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_sc_k17408.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 53.073 us, pipe_host 53.468
  vs card1 53.367 vs K=5120 16.5 vs N=17408 29.5.
  M=4 pipe 53.381 vs card1 53.303. Spread ~0.2%.

VERDICT -> Sibling matches. New s4 down-proj
  floor 53.4 us at 2800 both cards. ~3.24x
  K=5120, near linear. Rank us.

### 2026-09-02bs - K2 s4 M=64 N=17408 4x8 A-db card1

CONTEXT -> s4 4x8 A-db is 33.6 us at M=64
  N=5120. M=1 N=17408 was 1.80x not 3.4x.
  Napkin N-linear 33.6*17408/5120 ~114 us.
  Steal wide N on the M=64 floor tile. One-card.

CONFIG -> sycl+l0, standalone dpas_s4_db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=16 pack=2 spin=512. Fill s4 [-8,7]
  scales 0.02 out f16. M=64 N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_db48_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 94.297 us, pipe_host 94.560
  vs N=5120 33.6 vs napkin 114. Ratio 94.6/33.6
  ~2.81x, closer to N-linear than M=1's 1.80x.

VERDICT -> Prefill wide-N is a real 94.6 us at
  2800, ~2.81x N=5120. One-card. Do not freeze
  until card0. Rank us. Next: sibling M=64
  N=17408 vs s4 M=64 K=17408.

### 2026-09-02bt - K2 s4 M=64 N=17408 sibling card0

CONTEXT -> card1 s4 4x8 A-db N=17408 was 94.6 us
  at M=64 2800, numeric closed. Sibling swap to
  close the prefill wide-N floor.

CONFIG -> sycl+l0, standalone dpas_s4_db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=16 pack=2 spin=512 as bs.
  Fill s4 [-8,7] scales 0.02 out f16. M=64
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_db48_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 94.109 us, pipe_host 94.805
  vs card1 94.560 vs N=5120 33.6 vs napkin 114.
  Spread ~0.3%.

VERDICT -> Sibling matches. New s4 M=64 wide-N
  floor 94.7 us at 2800 both cards. ~2.81x
  N=5120. Rank us.

### 2026-09-02bu - K2 s4 M=64 K=17408 card1

CONTEXT -> s4 4x8 A-db is 33.6 us at M=64
  K=5120. M=1 K=17408 was 3.24x near linear.
  Napkin K-linear 33.6*17408/5120 ~114 us.
  Steal wide K on the M=64 floor tile. One-card.

CONFIG -> sycl+l0, standalone dpas_s4_db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=16 pack=2 spin=512. Fill s4 [-8,7]
  scales 0.02 out f16. M=64 N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_db48_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 104.958 us, pipe_host 105.843
  vs K=5120 33.6 vs N=17408 94.7 vs napkin 114.
  Ratio 105.8/33.6 ~3.15x, near K-linear 3.40x.

VERDICT -> Prefill wide-K is a real 105.8 us at
  2800, ~3.15x K=5120, slower than wide-N 94.7
  at the same B bytes. Gap shrinks vs M=1
  (53.4 vs 29.5). One-card. Do not freeze until
  card0. Rank us. Next: sibling M=64 K=17408
  vs s4 M=256 N=17408.

### 2026-09-02bv - K2 s4 M=64 K=17408 sibling card0

CONTEXT -> card1 s4 4x8 A-db K=17408 was 105.8 us
  at M=64 2800, numeric closed. Sibling swap to
  close the prefill wide-K floor.

CONFIG -> sycl+l0, standalone dpas_s4_db48, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=16 pack=2 spin=512 as bu.
  Fill s4 [-8,7] scales 0.02 out f16. M=64
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_db48_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 105.354 us, pipe_host 106.099
  vs card1 105.843 vs K=5120 33.6 vs N=17408 94.7.
  Spread ~0.2%.

VERDICT -> Sibling matches. New s4 M=64 wide-K
  floor 106.0 us at 2800 both cards. ~3.15x
  K=5120. Rank us.

### 2026-09-02bw - K2 s4 M=256 N=17408 4-acc card1

CONTEXT -> s4 4-acc is 48.6 us at M=256 N=5120.
  M=64 N=17408 was 2.81x. Napkin N-linear
  48.6*17408/5120 ~165 us. Steal wide N on the
  M=256 floor tile. One-card.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=8 pack=2 spin=512. Fill s4 [-8,7]
  scales 0.02 out f16. M=256 N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_w48m4_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=256 card1: event 140.958 us, pipe_host 140.531
  vs N=5120 48.6 vs napkin 165. Ratio 140.5/48.6
  ~2.89x, like M=64's 2.81x, not 3.40x.

VERDICT -> Prefill-256 wide-N is a real 140.5 us
  at 2800, ~2.89x N=5120. One-card. Do not freeze
  until card0. Rank us. Next: sibling M=256
  N=17408 vs s4 M=256 K=17408.

### 2026-09-02bx - K2 s4 M=256 N=17408 sibling card0

CONTEXT -> card1 s4 4-acc N=17408 was 140.5 us
  at M=256 2800, numeric closed. Sibling swap to
  close the prefill-256 wide-N floor.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=8 pack=2 spin=512 as bw.
  Fill s4 [-8,7] scales 0.02 out f16. M=256
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_w48m4_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=256 card0: event 139.313 us, pipe_host 139.436
  vs card1 140.531 vs N=5120 48.6 vs napkin 165.
  Spread ~0.8%.

VERDICT -> Sibling matches. New s4 M=256 wide-N
  floor 140.0 us at 2800 both cards. ~2.88x
  N=5120. Rank us.

### 2026-09-02by - K2 s4 M=256 K=17408 card1

CONTEXT -> s4 4-acc is 48.6 us at M=256 K=5120.
  M=64 K=17408 was 3.15x. Napkin K-linear
  48.6*17408/5120 ~165 us. Steal wide K on the
  M=256 floor tile. One-card.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 1.
  NT=2 U=8 pack=2 spin=512. Fill s4 [-8,7]
  scales 0.02 out f16. M=256 N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s4_w48m4_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=256 card1: event 149.302 us, pipe_host 149.164
  vs K=5120 48.6 vs N=17408 140.0 vs napkin 165.
  Ratio 149.2/48.6 ~3.07x, near K-linear 3.40x.

VERDICT -> Prefill-256 wide-K is a real 149.2 us
  at 2800, ~3.07x K=5120, slower than wide-N
  140.0 at the same B bytes. One-card. Do not
  freeze until card0. Rank us. Next: sibling
  M=256 K=17408 vs s8 M=64 N=17408 (INT8).

### 2026-09-02bz - K2 s4 M=256 K=17408 sibling card0

CONTEXT -> card1 s4 4-acc K=17408 was 149.2 us
  at M=256 2800, numeric closed. Sibling swap to
  close the prefill-256 wide-K floor.

CONFIG -> sycl+l0, standalone dpas_s4_w48m4, icpx
  2026.1.1 AOT intel_gpu_bmg_g31, gpu-run --card 0.
  Same NT=2 U=8 pack=2 spin=512 as by.
  Fill s4 [-8,7] scales 0.02 out f16. M=256
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s4_w48m4_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=256 card0: event 147.490 us, pipe_host 148.768
  vs card1 149.164 vs K=5120 48.6 vs N=17408 140.0.
  Spread ~0.3%.

VERDICT -> Sibling matches. New s4 M=256 wide-K
  floor 149.0 us at 2800 both cards. ~3.07x
  K=5120. Qwen FFN s4 map is closed. Rank us.

### 2026-09-02ca - K2 s8 M=64 N=17408 4x8 A-db card1

CONTEXT -> s8 4x8 A-db is 75 us at M=64 N=5120.
  s4 same tile N=17408 is 94.7 us (~2.81x).
  Napkin N-linear 75*17408/5120 ~255 us. INT8
  vs s4 at FFN-up prefill. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db48,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=512.
  Fill s8 [-64,64] scales 0.02 out f16. M=64
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc8db48_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 337.609 us, pipe_host 338.151
  vs N=5120 75 vs s4 94.7 vs napkin 255.
  Ratio 338/75 ~4.51x, worse than N-linear.
  s4 94.7 is ~3.57x this s8.

VERDICT -> INT8 4x8 A-db loses hard at wide N:
  4.51x N=5120 vs s4's 2.81x. One-card. Do not
  freeze 338 us until card0. Rank us. Next:
  sibling s8 N=17408 vs s8 M=64 K=17408.

### 2026-09-02cb - K2 s8 M=64 N=17408 sibling card0

CONTEXT -> card1 s8 4x8 A-db N=17408 was 338.2 us
  at M=64 2800, numeric closed. Sibling swap to
  close the INT8 prefill wide-N floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db48,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=512 as ca.
  Fill s8 [-64,64] scales 0.02 out f16. M=64
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8db48_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 336.287 us, pipe_host 339.628
  vs card1 338.151 vs N=5120 75 vs s4 94.7 vs
  napkin 255. Spread ~0.4%.

VERDICT -> Sibling matches. New s8 M=64 wide-N
  floor 338.9 us at 2800 both cards. ~4.52x
  N=5120, worse than linear. s4 94.7 is ~3.58x
  this tile. Rank us.

### 2026-09-02cc - K2 s8 M=64 K=17408 4x8 A-db card1

CONTEXT -> s8 4x8 A-db is 75 us at M=64 K=5120.
  s4 same tile K=17408 is 106.0 us (~3.15x).
  Napkin K-linear 75*17408/5120 ~255 us. INT8
  vs s4 at FFN-down prefill. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db48,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=512.
  Fill s8 [-64,64] scales 0.02 out f16. M=64
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc8db48_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 371.922 us, pipe_host 373.565
  vs K=5120 75 vs N=17408 338.9 vs s4 106.0 vs
  napkin 255. Ratio 373.6/75 ~4.98x, worse than
  K-linear. s4 106.0 is ~3.52x this s8.

VERDICT -> INT8 4x8 A-db loses harder at wide K:
  4.98x K=5120 vs s4's 3.15x, slower than wide-N
  338.9 at the same B bytes. One-card. Do not
  freeze 373.6 us until card0. Rank us. Next:
  sibling s8 K=17408 vs s8 M=256 N=17408.

### 2026-09-02cd - K2 s8 M=64 K=17408 sibling card0

CONTEXT -> card1 s8 4x8 A-db K=17408 was 373.6 us
  at M=64 2800, numeric closed. Sibling swap to
  close the INT8 prefill wide-K floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc8db48,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=512 as cc.
  Fill s8 [-64,64] scales 0.02 out f16. M=64
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8db48_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 375.953 us, pipe_host 375.899
  vs card1 373.565 vs K=5120 75 vs N=17408 338.9
  vs s4 106.0 vs napkin 255. Spread ~0.6%.

VERDICT -> Sibling matches. New s8 M=64 wide-K
  floor 374.7 us at 2800 both cards. ~5.00x
  K=5120, worse than linear. s4 106.0 is ~3.53x
  this tile. Rank us.

### 2026-09-02ce - K2 s8 M=256 N=17408 4-acc card1

CONTEXT -> s8 4-acc is 128 us at M=256 N=5120.
  s4 same tile N=17408 is 140.0 us (~2.88x).
  Napkin N-linear 128*17408/5120 ~435 us. INT8
  vs s4 at FFN-up prefill-256. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=8 spin=512.
  Fill s8 [-64,64] scales 0.02 out f16. M=256
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48m4_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2733
  cur=2800 throttle=1 (same flag as the N=5120
  128 us floor on this tile).
  M=256 card1: event 468.786 us, pipe_host 467.880
  vs N=5120 128 vs s4 140.0 vs napkin 435.
  Ratio 467.9/128 ~3.66x, near N-linear, better
  than M=64's 4.52x. s4 140.0 is ~3.34x this s8.

VERDICT -> Prefill-256 wide-N is a real 467.9 us
  at cur=2800, ~3.66x N=5120. Throttle=1, one-card.
  Do not freeze until card0. Rank us. Next:
  sibling s8 M=256 N=17408 vs s8 M=256 K=17408.

### 2026-09-02cf - K2 s8 M=256 N=17408 sibling card0

CONTEXT -> card1 s8 4-acc N=17408 was 467.9 us at
  M=256 cur=2800 throttle=1, numeric closed.
  Sibling swap to close the INT8 prefill-256
  wide-N floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=8 spin=512 as ce.
  Fill s8 [-64,64] scales 0.02 out f16. M=256
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48m4_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2700
  cur=2800 throttle=1 (same flag as card1 and
  the N=5120 128 us floor).
  M=256 card0: event 472.594 us, pipe_host 471.658
  vs card1 467.880 vs N=5120 128 vs s4 140.0 vs
  napkin 435. Spread ~0.8%.

VERDICT -> Sibling matches. New s8 M=256 wide-N
  floor 469.8 us at cur=2800 both cards,
  throttle=1. ~3.67x N=5120, near linear. s4
  140.0 is ~3.36x this tile. Rank us.

### 2026-09-02cg - K2 s8 M=256 K=17408 4-acc card1

CONTEXT -> s8 4-acc is 128 us at M=256 K=5120.
  s4 same tile K=17408 is 149.0 us (~3.07x).
  Napkin K-linear 128*17408/5120 ~435 us. INT8
  vs s4 at FFN-down prefill-256. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=8 spin=512.
  Fill s8 [-64,64] scales 0.02 out f16. M=256
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc8w48m4_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2750
  cur=2800 throttle=1 (same flag as this tile's
  128 us floor).
  M=256 card1: event 477.453 us, pipe_host 476.927
  vs K=5120 128 vs N=17408 469.8 vs s4 149.0 vs
  napkin 435. Ratio 476.9/128 ~3.73x, near
  K-linear. s4 149.0 is ~3.20x this s8.

VERDICT -> Prefill-256 wide-K is a real 476.9 us
  at cur=2800, ~3.73x K=5120, slightly slower
  than wide-N 469.8 at the same B bytes. Gap
  shrinks vs M=64 (374.7 vs 338.9). Throttle=1,
  one-card. Do not freeze until card0. Rank us.
  Next: sibling s8 M=256 K=17408 vs s8 decode
  N=17408.

### 2026-09-02ch - K2 s8 M=256 K=17408 sibling card0

CONTEXT -> card1 s8 4-acc K=17408 was 476.9 us at
  M=256 cur=2800 throttle=1, numeric closed.
  Sibling swap to close the INT8 prefill-256
  wide-K floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc8w48m4,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=8 spin=512 as cg.
  Fill s8 [-64,64] scales 0.02 out f16. M=256
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc8w48m4_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2733
  cur=2800 throttle=1 (same flag as card1 and
  the N=5120 128 us floor).
  M=256 card0: event 479.161 us, pipe_host 477.797
  vs card1 476.927 vs K=5120 128 vs N=17408 469.8
  vs s4 149.0 vs napkin 435. Spread ~0.2%.

VERDICT -> Sibling matches. New s8 M=256 wide-K
  floor 477.4 us at cur=2800 both cards,
  throttle=1. ~3.73x K=5120, near linear. s4
  149.0 is ~3.20x this tile. Qwen FFN s8 prefill
  map is closed. Rank us.

### 2026-09-02ci - K2 s8 decode N=17408 card1

CONTEXT -> s8 RC=4 8x2-N is 34 us at M=1 N=5120.
  s4 same tile N=17408 is 29.5 us (~1.80x, not
  3.4x). Napkin N-linear 34*17408/5120 ~116 us.
  INT8 vs s4 at FFN-up decode. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=4000.
  Fill s8 [-64,64] scales 0.02 out f16. M=1 and
  M=4, N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc_wide.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 142.516 us, pipe_host 142.052
  vs N=5120 34 vs s4 29.5 vs napkin 116.
  Ratio 142.1/34 ~4.18x, worse than N-linear.
  M=4 tracks (pipe 142.333). s4 29.5 is ~4.82x
  this s8.

VERDICT -> INT8 decode loses hard at wide N:
  4.18x N=5120 vs s4's 1.80x. One-card. Do not
  freeze 142.1 us until card0. Rank us. Next:
  sibling s8 decode N=17408 vs s8 decode K=17408.

### 2026-09-02cj - K2 s8 decode N=17408 sibling card0

CONTEXT -> card1 s8 RC=4 N=17408 was 142.1 us at
  M=1 2800, numeric closed. Sibling swap to
  close the INT8 FFN-up decode floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=4000 as ci.
  Fill s8 [-64,64] scales 0.02 out f16. M=1 and
  M=4, N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc_wide.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 140.989 us, pipe_host 141.222
  vs card1 142.052 vs N=5120 34 vs s4 29.5 vs
  napkin 116. Spread ~0.6%. M=4 tracks
  (pipe 143.080).

VERDICT -> Sibling matches. New s8 decode wide-N
  floor 141.6 us at 2800 both cards. ~4.16x
  N=5120, worse than linear. s4 29.5 is ~4.80x
  this tile. Rank us.

### 2026-09-02ck - K2 s8 decode K=17408 card1

CONTEXT -> s8 RC=4 8x2-N is 34 us at M=1 K=5120.
  s4 same tile K=17408 is 53.4 us (~3.24x).
  Napkin K-linear 34*17408/5120 ~116 us. INT8
  vs s4 at FFN-down decode. One-card.

CONFIG -> sycl+l0, standalone dpas_s8_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=4000.
  Fill s8 [-64,64] scales 0.02 out f16. M=1 and
  M=4, N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_sc_k17408.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 261.716 us, pipe_host 261.510
  vs K=5120 34 vs N=17408 141.6 vs s4 53.4 vs
  napkin 116. Ratio 261.5/34 ~7.69x, much worse
  than K-linear. M=4 tracks (pipe 261.442).
  s4 53.4 is ~4.90x this s8.

VERDICT -> INT8 decode loses harder at wide K:
  7.69x K=5120 vs s4's 3.24x, slower than
  wide-N 141.6 at the same B bytes. One-card.
  Do not freeze 261.5 us until card0. Rank us.
  Next: sibling s8 decode K=17408 vs oneDNN
  W8A8 M=1 N=17408.

### 2026-09-02cl - K2 s8 decode K=17408 sibling card0

CONTEXT -> card1 s8 RC=4 K=17408 was 261.5 us at
  M=1 2800, numeric closed. Sibling swap to
  close the INT8 FFN-down decode floor.

CONFIG -> sycl+l0, standalone dpas_s8_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=4000 as ck.
  Fill s8 [-64,64] scales 0.02 out f16. M=1 and
  M=4, N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_sc_k17408.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 261.193 us, pipe_host 261.675
  vs card1 261.510 vs K=5120 34 vs N=17408 141.6
  vs s4 53.4 vs napkin 116. Spread ~0.06%.
  M=4 tracks (pipe 261.257).

VERDICT -> Sibling matches. New s8 decode wide-K
  floor 261.6 us at 2800 both cards. ~7.69x
  K=5120, much worse than linear. s4 53.4 is
  ~4.90x this tile. Qwen FFN s8 map is closed.
  Rank us.

### 2026-09-02cm - K4 oneDNN W8A8 M=1 N=17408 card1

CONTEXT -> K4 sweep M=1 N=17408 was 161 us, clocks
  unnamed. Hand s8 decode N=17408 is 141.6 us at
  2800. Held-clock incumbent control. One-card.

CONFIG -> pytorch-xpu on sycl+l0, sglang int8 mtp6
  `int8_gemm_w8a8` GEMM-only, gpu-run --card 1.
  spin=2000 of M=1 then 30 warmup + 40 timed.
  Host oracle after timed. n=17408 k=5120.
  First two holds (M=64 heat; oracle-before-timed)
  dropped to 2050/1733 MHz and are not ranked.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m1hold_wide.sh 1
  ```

RESULT -> cosine=1.000 max_abs=0.055. timed
  act=cur=2800 throttle=0.
  M=1 card1: 158.006 us vs K4 sweep 161 vs hand
  s8 141.6 vs s4 29.5.

VERDICT -> Held-2800 oneDNN wide-N decode is
  158.0 us on card1. Hand s8 141.6 is ~1.12x
  this incumbent at the same clock and contract.
  One-card. Do not freeze 158 us until card0.
  Rank us. Next: sibling W8A8 M=1 N=17408 vs
  W8A8 M=1 K=17408.

### 2026-09-02cn - K4 oneDNN W8A8 M=1 N=17408 sibling card0

CONTEXT -> card1 oneDNN W8A8 M=1 N=17408 was
  158.0 us at 2800. Sibling swap to close the
  incumbent wide-N decode floor.

CONFIG -> pytorch-xpu on sycl+l0, sglang int8 mtp6
  `int8_gemm_w8a8` GEMM-only, gpu-run --card 0.
  Same spin=2000 of M=1, oracle after timed.
  n=17408 k=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m1hold_wide.sh 0 17408 5120
  ```

RESULT -> cosine=1.000 max_abs=0.055. timed
  act=cur=2800 throttle=0.
  M=1 card0: 158.132 us vs card1 158.006 vs hand
  s8 141.6 vs s4 29.5 vs K4 sweep 161.
  Spread ~0.08%.

VERDICT -> Sibling matches. New oneDNN W8A8
  wide-N decode floor 158.1 us at 2800 both
  cards. Hand s8 141.6 is ~1.12x this
  incumbent. Rank us.

### 2026-09-02co - K4 oneDNN W8A8 M=1 K=17408 card1

CONTEXT -> oneDNN W8A8 M=1 N=17408 is 158.1 us.
  Same B bytes as FFN-down K=17408 N=5120.
  Hand s8 decode K is 261.6 us (~7.69x). Napkin
  oneDNN K-linear 44*17408/5120 ~150 us. One-card.

CONFIG -> pytorch-xpu on sycl+l0, sglang int8 mtp6
  `int8_gemm_w8a8` GEMM-only, gpu-run --card 1.
  spin=2000 of M=1, oracle after timed.
  n=5120 k=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m1hold_wide.sh 1 5120 17408
  ```

RESULT -> cosine=1.000 max_abs=0.104. timed
  act=cur=2800 throttle=0.
  M=1 card1: 155.368 us vs N=17408 158.1 vs hand
  s8 261.6 vs s4 53.4 vs W8A8 5120 44.
  Ratio 155.4/44 ~3.53x, near linear.

VERDICT -> oneDNN wide-K decode is 155.4 us on
  card1, N/K-symmetric with 158.1, unlike hand
  s8 261.6 vs 141.6. Hand RC=4 8x2-N loses to
  oneDNN at FFN-down (~1.68x). One-card. Do not
  freeze 155.4 us until card0. Rank us. Next:
  sibling W8A8 K=17408 vs serving-shaped NVFP4
  LUT on the decode tile.

### 2026-09-02cp - K4 oneDNN W8A8 M=1 K=17408 sibling card0

CONTEXT -> card1 oneDNN W8A8 M=1 K=17408 was
  155.4 us at 2800. Sibling swap to close the
  incumbent FFN-down decode floor.

CONFIG -> pytorch-xpu on sycl+l0, sglang int8 mtp6
  `int8_gemm_w8a8` GEMM-only, gpu-run --card 0.
  Same spin=2000 of M=1, oracle after timed.
  n=5120 k=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m1hold_wide.sh 0 5120 17408
  ```

RESULT -> cosine=1.000 max_abs=0.070. timed
  act=cur=2800 throttle=0.
  M=1 card0: 155.310 us vs card1 155.368 vs
  N=17408 158.1 vs hand s8 261.6 vs s4 53.4.
  Spread ~0.04%.

VERDICT -> Sibling matches. New oneDNN W8A8
  wide-K decode floor 155.3 us at 2800 both
  cards. N/K-symmetric with 158.1. Hand s8
  261.6 loses ~1.68x at FFN-down. Qwen FFN
  oneDNN W8A8 decode map is closed. Rank us.
  Next: both-card K6 nibble LUT on the s8
  decode tile (packed E2M1, never bitcast s4).

### 2026-09-02cq - K6 nibble LUT on s8 decode tile both cards

CONTEXT -> K6 two-launch unpack closed at 1024^3.
  Scalar in-register LUT lost (2316 us). simd LUT
  at 1024^3 was clock-bound. Serving-shaped
  question: packed E2M1 in HBM, simd LUT, VNNI4,
  then the K2 RC=4 8x2-N s8 tile. Never bitcast
  s4. Napkin: LUT tax vs s8 34 us; unpack of
  25 MiB B every token would also lose.

CONFIG -> sycl+l0, standalone nibble_lut_sc, icpx
  2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0 || --card 1. NT=2 U=16
  spin=4000. A s8 [-64,64], B random E2M1
  nibbles packed 2/byte along K. Scales 0.02
  out f16. M=1 and M=4, N=K=5120. New numeric.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_sc.sh 0 2 4000
  gpu-run --card 1 kernels/nvfp4/run_k6_sc.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0 both cards.
  timed act=cur=2800 throttle=0.
  M=1 pipe_host 158.172/158.178 vs s8 34 vs
  W8A8 44. M=4 tracks (158.304/158.182).
  Packed-B 83 GB/s. Spread ~0.004%.

VERDICT -> First serving-shaped NVFP4
  in-register spoof is numerically closed.
  Packed E2M1 stays in HBM. Not a us beat of
  s8 34 (~4.65x) or W8A8 44. LUT tax, not
  HBM. "Cannot feed XMX" is false; "as fast
  as s8" is false. Rank us. Next: two-launch
  unpack control on this tile vs LUT tax steal.

### 2026-09-02cr - K6 16-entry iselect table LUT both cards

CONTEXT -> nibble_lut_sc merge LUT is 158 us at
  2800. Napkin: a 16-entry E2M1 table + iselect
  is fewer merges. Same RC=4 8x2-N tile. New
  LUT implementation, both cards.

CONFIG -> sycl+l0, standalone nibble_lut_sct,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0 || --card 1. NT=2 U=16
  spin=4000. Same fill/scales as cq.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_sct.sh 0 2 4000
  gpu-run --card 1 kernels/nvfp4/run_k6_sct.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0 both cards.
  timed act=cur=2800 throttle=0.
  M=1 pipe_host 1021.73/1021.88 vs merge LUT
  158 vs s8 34. M=4 tracks. Packed-B 12.8 GB/s.
  Spread ~0.01%. Zebin 2.5 MiB vs merge 0.86.

VERDICT -> iselect table is a loss (~6.46x
  merge LUT). Numeric closed, us lost. Stop
  GRF gather tables on this tile. Keep the
  merge-chain simd LUT. Rank us. Next:
  two-launch unpack control on the decode tile.

### 2026-09-03a - K6 two-launch unpack on decode tile both cards

CONTEXT -> nibble_lut_sc merge LUT is 158 us at
  2800. Two-launch unpack was the fast spoof at
  1024^3. Serving-shape control: unpack packed
  E2M1 to s8 each iter, then Transformed s8 GEMM
  on the RC=4 tile. Never bitcast s4. Napkin:
  unpack of 25 MiB s8 B plus 34 us GEMM.

CONFIG -> sycl+l0, standalone nibble_unpack_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0 || --card 1. NT=2 U=16
  spin=4000. Scalar 1-byte unpack then s8 GEMM.
  Same fill/scales as cq.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_unpack.sh 0 2 4000
  gpu-run --card 1 kernels/nvfp4/run_k6_unpack.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0 both cards.
  timed cur=2800 throttle=1 (act 2733-2783).
  M=1 two-launch pipe_host 266.10/263.31 vs
  fused LUT 158 vs s8ctrl 34.55/35.24 vs s8 34.
  M=4 tracks. Spread ~1%. Rank pipe, not the
  55 us gemm-only event.

VERDICT -> Naive two-launch unpack loses to
  the 158 us in-register LUT (~1.67x). s8ctrl
  34.5 confirms the GEMM is still the 34 us
  tile; the tax is scalar unpack. throttle=1
  is part of this control. Vectorized unpack
  still open. Rank us.

### 2026-09-03b - K6 one packed load per k64 both cards

CONTEXT -> nibble_lut_sc does two k32 packed
  loads + LUTs per k64. Steal: one height-32
  packed load, merge LUT width 512, two VNNI4
  dpas. No iselect. Same tile.

CONFIG -> sycl+l0, standalone nibble_lut_sck,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0 || --card 1. NT=2 U=16
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_sck.sh 0 2 4000
  gpu-run --card 1 kernels/nvfp4/run_k6_sck.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0 both cards.
  timed act=cur=2800 throttle=0.
  M=1 pipe_host 169.02/169.14 vs two-k32 LUT
  158. M=4 tracks. Spread ~0.07%.

VERDICT -> k64 combined load is a small loss
  (~1.07x). Keep two k32 packed loads. Numeric
  closed. Rank us. Next: vectorized unpack.

### 2026-09-03c - K6 vectorized two-launch unpack card0

CONTEXT -> scalar unpack_sc is 265 us pipe at
  2800 vs fused LUT 158. Steal simd nibble
  decode on the unpack kernel, same two-launch
  Transformed s8 GEMM. One-card.

CONFIG -> sycl+l0, standalone nibble_unpack_scv,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. NT=2 U=16 spin=4000.
  ESIMD 16-wide unpack. Never bitcast s4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_unpackv.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 54.945 us (GEMM only),
  pipe_host 314.721 vs scalar unpack 265 vs LUT
  158 vs s8ctrl 34.291. M=4 tracks (pipe 314.387).

VERDICT -> Vectorized two-launch unpack loses to
  scalar unpack (~1.19x) and to fused LUT
  (~2.0x). Numeric closed. One-card. Do not
  freeze. Keep fused 158 us LUT. Rank pipe.

### 2026-09-03d - K3/K6 E2M1 two-term s4 decode card1

CONTEXT -> K3 two-term w_lo+8*w_hi was closed on
  the square tile. Steal onto the RC=4 8x2-N
  decode tile. A is s4. Never bitcast. Napkin:
  two s4 dpas ~2x 16.5 = 33 us. One-card.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=4000.
  Two packed s4 B planes. Fill E2M1 split.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_sc.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 28.250 us, pipe_host 28.544
  vs s4 16.5 vs s8 34 vs W8A8 44 vs LUT 158.
  Ratio 28.5/16.5 ~1.73x, under 2x napkin.
  M=4 tracks (pipe 28.549).

VERDICT -> Overflow-split E2M1 on two s4 DPAS is
  a real 28.5 us at 2800, under s8 34 and W8A8
  44, with A=s4. Numeric closed. One-card. Do
  not freeze 28.5 us until card0. Rank us.
  Next: sibling compose_e2m1_sc vs sibling
  unpackv.

### 2026-09-03e - K3/K6 E2M1 two-term s4 sibling card0

CONTEXT -> card1 compose_e2m1_sc was 28.54 us
  at 2800, numeric closed, A=s4. Sibling swap
  to close the overflow-split decode floor.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_sc.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 28.203 us, pipe_host 28.520
  vs card1 28.544 vs s4 16.5 vs s8 34 vs W8A8
  44. Spread ~0.08%. M=4 pipe 28.685.

VERDICT -> Sibling matches. New E2M1 two-term
  s4 decode floor 28.5 us at 2800 both cards.
  ~1.73x native s4. A is s4, not the s8-A LUT
  contract. Rank us.

### 2026-09-03f - K6 vectorized unpack sibling card1

CONTEXT -> card0 nibble_unpack_scv was 314.7 us
  at 2800, a loss. Sibling swap.

CONFIG -> sycl+l0, standalone nibble_unpack_scv,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. Same NT=2 U=16 spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_unpackv.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: pipe_host 314.444 vs card0 314.721
  vs scalar 265 vs LUT 158 vs s8ctrl 34.061.
  Spread ~0.09%. M=4 tracks.

VERDICT -> Sibling matches. Vectorized unpack
  is 314.6 us both cards, a loss. Stop this
  unpack path. Keep fused 158 us LUT for s8-A.

### 2026-09-03g - K3/K6 E2M1 two-term N=17408 card0

CONTEXT -> compose_e2m1_sc is 28.5 us at N=5120.
  s4 N=17408 is 29.5 us (1.80x, not 3.4x).
  Napkin N-linear 28.5*17408/5120 ~97 us.
  FFN-up decode. One-card. A=s4.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. NT=2 U=16 spin=4000.
  M=1 and M=4, N=17408 K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_sc_wide.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 101.336 us, pipe_host 102.729
  vs 5120 28.5 vs s4 29.5 vs s8 141.6 vs napkin
  97. Ratio 102.7/28.5 ~3.60x, near linear, not
  s4's 1.80x. M=4 tracks (pipe 101.733).

VERDICT -> Wide-N compose is a real 102.7 us at
  2800, ~3.60x square, slower than native s4
  29.5 at this shape. One-card. Do not freeze
  until card1. Rank us.

### 2026-09-03h - K3/K6 E2M1 two-term K=17408 card1

CONTEXT -> compose N=17408 is 102.7 us. s4
  K=17408 is 53.4 us (~3.24x). Napkin K-linear
  ~97 us. FFN-down decode. One-card. A=s4.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=4000.
  M=1 and M=4, N=5120 K=17408. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_sc_k17408.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 192.753 us, pipe_host 193.096
  vs 5120 28.5 vs N=17408 102.7 vs s4 53.4 vs
  napkin 97. Ratio 193.1/28.5 ~6.78x, worse than
  linear. M=4 tracks (pipe 193.146).

VERDICT -> Wide-K compose is 193.1 us at 2800,
  ~6.78x square, K-hostile like s8 8x2-N.
  Native s4 53.4 still wins. One-card. Do not
  freeze until card0. Rank us. Next: sibling
  N=17408 vs sibling K=17408.

### 2026-09-03i - K3/K6 E2M1 two-term N=17408 sibling card1

CONTEXT -> card0 compose N=17408 was 102.7 us
  at 2800, numeric closed. Sibling swap.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. Same NT=2 U=16 spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_sc_wide.sh 1 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card1: event 103.401 us, pipe_host 104.353
  vs card0 102.729 vs s4 29.5 vs s8 141.6.
  Spread ~1.6%. M=4 pipe 102.142.

VERDICT -> Sibling matches within 2%. New
  E2M1 two-term wide-N floor 103.5 us at 2800
  both cards. ~3.63x square, not s4's 1.80x.
  Native s4 29.5 still wins this shape. Rank us.

### 2026-09-03j - K3/K6 E2M1 two-term K=17408 sibling card0

CONTEXT -> card1 compose K=17408 was 193.1 us
  at 2800, numeric closed. Sibling swap.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_sc_k17408.sh 0 2 4000
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=1 card0: event 192.922 us, pipe_host 194.021
  vs card1 193.096 vs s4 53.4 vs s8 261.6 vs
  W8A8 155.3. Spread ~0.5%. M=4 pipe 195.476.

VERDICT -> Sibling matches. New E2M1 two-term
  wide-K floor 193.6 us at 2800 both cards.
  ~6.79x square, K-hostile. Native s4 53.4 and
  oneDNN W8A8 155 both beat this at FFN-down.
  Qwen FFN compose decode map is closed. Rank us.

### 2026-09-03k - K3/K6 E2M1 two-term M=64 card0

CONTEXT -> compose 8x2-N is 28.5 us at M=1.
  s4 4x8 A-db is 33.6 us at M=64. Prefill on
  the decode tile. One-card. A=s4.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. NT=2 U=16 spin=512.
  M=64 N=K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_sc_m64.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 217.974 us, pipe_host 217.915
  vs M=1 28.5 vs s4 33.6 vs s8 75 vs W8A8 46.
  Ratio 217.9/28.5 ~7.65x.

VERDICT -> 8x2-N compose loses hard at M=64:
  ~6.5x s4 4x8, ~4.7x W8A8. Decode tile is not
  a prefill floor. One-card. Do not freeze.
  Rank us.

### 2026-09-03l - K3/K6 E2M1 two-term M=256 card1

CONTEXT -> compose M=64 8x2-N was 217.9 us.
  s4 4-acc is 48.6 us at M=256. One-card. A=s4.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=512.
  M=256 N=K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_sc_m256.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2683
  cur=2800 throttle=1.
  M=256 card1: event 603.057 us, pipe_host 601.181
  vs M=1 28.5 vs s4 48.6 vs s8 128 vs W8A8 75.
  Ratio 601/28.5 ~21.1x.

VERDICT -> 8x2-N compose loses harder at M=256:
  ~12.4x s4 4-acc, ~8.0x W8A8. throttle=1.
  One-card. Do not freeze 601 us. Stop this
  tile at prefill. Rank us. Next: sibling
  M=256 vs nibble_lut_sc M=64.

### 2026-09-03m - K3/K6 E2M1 two-term M=256 sibling card0

CONTEXT -> card1 compose M=256 8x2-N was
  601 us at cur=2800 throttle=1. Sibling swap.

CONFIG -> sycl+l0, standalone compose_e2m1_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_sc_m256.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=2667
  cur=2800 throttle=1.
  M=256 card0: event 610.969 us, pipe_host
  612.683 vs card1 601.181 vs s4 48.6 vs s8
  128 vs W8A8 75. Spread ~1.9%.

VERDICT -> Sibling matches the loss. 8x2-N
  compose at M=256 is ~607 us both cards,
  throttle=1. Stop this tile at prefill.
  Keep 28.5 us decode-only. Rank us.

### 2026-09-03n - K6 nibble LUT M=64 card1

CONTEXT -> fused LUT is 158 us at M=1 on
  8x2-N. s8 4x8 A-db is 75 us at M=64.
  Prefill on the decode LUT tile. One-card.

CONFIG -> sycl+l0, standalone nibble_lut_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 1. NT=2 U=16 spin=512.
  M=64 N=K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_sc_m64.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 651.839 us, pipe_host
  645.630 vs M=1 158 vs s8 75 vs W8A8 46.
  min/max 575-807.

VERDICT -> 8x2-N LUT loses at M=64 (~8.6x
  s8 4x8, ~4.1x its own M=1). Decode tile is
  not a prefill floor for the s8-A spoof
  either. One-card. Do not freeze 646 us.

### 2026-09-03o - K6 nibble LUT M=64 sibling card0

CONTEXT -> card1 nibble_lut_sc M=64 8x2-N was
  646 us at 2800, numeric closed. Sibling
  swap. One-card.

CONFIG -> sycl+l0, standalone nibble_lut_sc,
  icpx 2026.1.1 AOT intel_gpu_bmg_g31,
  gpu-run --card 0. Same NT=2 U=16 spin=512.
  M=64 N=K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_sc_m64.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card0: event 643.667 us, pipe_host
  665.562 vs card1 645.630 vs M=1 158 vs
  s8 75 vs W8A8 46. Spread ~3.1%. min/max
  577-740.

VERDICT -> Sibling matches the loss. 8x2-N
  LUT at M=64 is ~656 us both cards. Stop
  this tile at prefill for the s8-A spoof.
  Rank us.

### 2026-09-03p - K3/K6 E2M1 two-term 4x8 A-db M=64 card1

CONTEXT -> 8x2-N compose loses at M=64
  (217.9 us vs s4 4x8 33.6). Steal two-term
  E2M1 onto the s4 4x8 A-db prefill tile.
  New geometry. One-card. A=s4. Never
  bitcast. Napkin: 2x s4 ~67 us if the two
  DPAS terms are compute-bound.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 (64 s4 dpas.8x8 lo+hi),
  wg 4x8 NxM, k64 A ping-pong, pack=2,
  grf_size<256>, no SLM. spin=512
  warmup=10 iters=20. Fill A s4 [-8,7],
  B E2M1 split w_lo+8*w_hi. M=64 N=K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_m64.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: NT=2 64x dpas.8x8 rW:s4
  rA:s4, store_block2d d16, grf_count 128,
  no slm_size, no spill. cosine=1.0
  max_abs=0. timed act=cur=2800 throttle=0.
  M=64 card1: event 67.943 us, pipe_host
  68.681 vs 8x2-N 217.9 vs s4 33.6 vs s8
  75 vs W8A8 46 vs napkin 67. Ratio
  68.7/33.6 ~2.04x. min/max 66.25-68.96.

VERDICT -> 4x8 A-db compose is a real
  ~3.17x beat of 8x2-N 217.9 and ~2.04x
  native s4, under s8 75, over W8A8 46.
  Napkin 2x held. One-card. Do not freeze
  68.7 us until card0. Rank us. Next:
  sibling 4x8 compose vs nibble LUT on
  4x8 A-db.

### 2026-09-03q - K3/K6 E2M1 two-term 4x8 A-db sibling card0

CONTEXT -> card1 compose_e2m1_db48 M=64 was
  68.7 us at 2800, numeric closed, IGA s4.
  Sibling swap to close the prefill floor.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_m64.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 67.859 us, pipe_host
  68.732 vs card1 68.681 vs 8x2-N 217.9 vs
  s4 33.6 vs s8 75 vs W8A8 46. Spread
  ~0.07%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 A-db floor 68.7 us at 2800
  both cards. ~2.04x native s4, ~3.17x
  8x2-N. Beats s8 75, loses to W8A8 46.
  A is s4. Rank us.

### 2026-09-03r - K3/K6 E2M1 two-term 4x8 A-db M=256 card0

CONTEXT -> 4x8 A-db compose is 68.7 us at
  M=64. 8x2-N compose at M=256 is 607 us
  throttle=1. s4 4-acc is 48.6. Prefill on
  the M=64 tile. One-card. A=s4.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  RC=8 NT=2 U=16 spin=512. M=256 N=K=5120.
  Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_m256.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 193.984 us, pipe_host
  194.823 vs M=64 68.7 vs 8x2-N 607 vs s4
  4-acc 48.6 vs s8 128 vs W8A8 75. Ratio
  194.8/68.7 ~2.84x (M*4). min/max
  191.9-197.0.

VERDICT -> 4x8 A-db compose at M=256 is a
  real ~3.12x beat of 8x2-N 607, throttle=0.
  ~4.0x native s4 4-acc, over W8A8 75. Not
  the 4-acc tile. One-card. Do not freeze
  194.8 us until card1. Rank us.

### 2026-09-03s - K6 nibble LUT 4x8 A-db M=64 card1

CONTEXT -> 8x2-N LUT loses at M=64 (656 us
  vs s8 4x8 75). Steal merge LUT onto the
  s8 4x8 A-db prefill tile. New geometry.
  One-card. Never bitcast. Napkin: decode
  LUT tax 4.65x * 75 ~349 us.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 (64 s8 dpas.8x8), wg 4x8
  NxM, k64 A ping-pong, packed E2M1 B,
  simd LUT + VNNI4. spin=512 warmup=10
  iters=20. M=64 N=K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_db48_m64.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: NT=2 64x dpas.8x8 rW:b
  rA:b, packed B load_block2d d8 (not d8v),
  store_block2d d16, grf_count 128, no
  slm_size. cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 392.375 us, pipe_host
  392.443 vs 8x2-N 656 vs s8 75 vs W8A8 46
  vs compose 68.7 vs napkin 349. Ratio
  392.4/75 ~5.23x. min/max 392.08-392.81.

VERDICT -> 4x8 A-db LUT is a real ~1.67x
  beat of 8x2-N 656. Still ~5.23x s8 75.
  Packed E2M1 stays in HBM. One-card. Do
  not freeze 392 us until card0. Rank us.
  Next: sibling compose M=256 vs compose
  on the M=256 4-acc s4 tile.

### 2026-09-03t - K3/K6 E2M1 two-term 4x8 A-db M=256 sibling card1

CONTEXT -> card0 compose_e2m1_db48 M=256 was
  194.8 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_m256.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card1: event 194.333 us, pipe_host
  195.034 vs card0 194.823 vs s4 48.6 vs
  s8 128 vs W8A8 75. Spread ~0.1%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 A-db M=256 floor 194.9 us
  at 2800 both cards. ~2.84x M=64, ~4.0x
  s4 4-acc. Beats 8x2-N 607 and s8 128,
  loses to W8A8 75. A is s4. Rank us.

### 2026-09-03u - K6 nibble LUT 4x8 A-db sibling card0

CONTEXT -> card1 nibble_lut_db48 M=64 was
  392.4 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_db48_m64.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 391.786 us, pipe_host
  392.427 vs card1 392.443 vs 8x2-N 656
  vs s8 75 vs W8A8 46. Spread ~0.004%.

VERDICT -> Sibling matches. New 4x8 A-db
  LUT floor 392.4 us at 2800 both cards.
  ~1.67x 8x2-N, still ~5.23x s8 75. Packed
  E2M1 stays in HBM. Rank us.

### 2026-09-03v - K3/K6 E2M1 two-term 4x8 A-db M=64 N=17408 card0

CONTEXT -> compose 4x8 A-db is 68.7 us at
  M=64 N=5120. s4 same tile N=17408 is
  94.7 us (~2.81x). Napkin N-linear
  68.7*17408/5120 ~233 us. FFN-up prefill.
  One-card. A=s4.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  RC=8 NT=2 U=16 spin=512. M=64 N=17408
  K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 326.505 us, pipe_host
  328.026 vs 5120 68.7 vs s4 94.7 vs s8
  338.9 vs napkin 233. Ratio 328.0/68.7
  ~4.77x, worse than s4's 2.81x. min/max
  308.8-350.9.

VERDICT -> Wide-N compose is a real 328 us
  at 2800, ~4.77x square, ~3.46x native s4
  94.7. N-hostile vs s4. One-card. Do not
  freeze until card1. Rank us.

### 2026-09-03w - K3/K6 E2M1 two-term 4x8 A-db M=64 K=17408 card1

CONTEXT -> compose N=17408 is 328 us.
  s4 K=17408 is 106.0 us (~3.15x). Napkin
  K-linear ~233 us. FFN-down prefill.
  One-card. A=s4.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  Same RC=8 NT=2 U=16 spin=512. M=64
  N=5120 K=17408. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 402.849 us, pipe_host
  403.192 vs 5120 68.7 vs N=17408 328 vs
  s4 106.0 vs s8 374.7 vs napkin 233.
  Ratio 403.2/68.7 ~5.87x, worse than s4's
  3.15x. min/max 393.0-410.7.

VERDICT -> Wide-K compose is 403 us at
  2800, ~5.87x square, ~3.80x native s4
  106, and loses to s8 374.7. K-hostile.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling N=17408 vs
  sibling K=17408.

### 2026-09-03x - K3/K6 E2M1 two-term 4x8 A-db N=17408 sibling card1

CONTEXT -> card0 compose M=64 N=17408 was
  328 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 324.958 us, pipe_host
  325.801 vs card0 328.026 vs s4 94.7 vs
  s8 338.9. Spread ~0.7%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 wide-N floor 326.9 us at
  2800 both cards. ~4.76x square, ~3.45x
  s4 94.7. Barely under s8 338.9. Rank us.

### 2026-09-03y - K3/K6 E2M1 two-term 4x8 A-db K=17408 sibling card0

CONTEXT -> card1 compose M=64 K=17408 was
  403 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 402.219 us, pipe_host
  403.596 vs card1 403.192 vs s4 106.0 vs
  s8 374.7. Spread ~0.1%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 wide-K floor 403.4 us at
  2800 both cards. ~5.87x square, ~3.81x
  s4 106, loses to s8 374.7. Qwen FFN
  compose M=64 map is closed. Rank us.

### 2026-09-03z - K3/K6 E2M1 two-term 4x8 A-db M=256 N=17408 card0

CONTEXT -> compose 4x8 A-db is 194.9 us at
  M=256 N=5120. s4 4-acc N=17408 is 140.0
  us (~2.88x). s8 469.8 throttle=1. M=64
  N=17408 is 326.9 (~4.76x). FFN-up
  prefill. One-card. A=s4.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  RC=8 NT=2 U=16 spin=512. M=256 N=17408
  K=5120. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_m256_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 983.870 us, pipe_host
  985.644 vs 5120 194.9 vs s4 140.0 vs s8
  469.8 vs M=64 326.9. Ratio 985.6/194.9
  ~5.06x, worse than s4's 2.88x. min/max
  971.8-996.9.

VERDICT -> Wide-N compose at M=256 is a
  real 986 us at 2800, ~5.06x square,
  ~7.0x native s4 140, ~2.10x s8 469.8.
  throttle=0. One-card. Do not freeze.
  Rank us.

### 2026-09-03aa - K3/K6 E2M1 two-term 4x8 A-db M=256 K=17408 card1

CONTEXT -> compose N=17408 M=256 is 986 us.
  s4 4-acc K=17408 is 149.0 us (~3.07x).
  s8 477.4 throttle=1. M=64 K=17408 is
  403.4. FFN-down prefill. One-card. A=s4.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  Same RC=8 NT=2 U=16 spin=512. M=256
  N=5120 K=17408. Never bitcast.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_m256_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card1: event 970.375 us, pipe_host
  973.110 vs 5120 194.9 vs N=17408 986 vs
  s4 149.0 vs s8 477.4 vs M=64 403.4.
  Ratio 973.1/194.9 ~4.99x. min/max
  920.2-1053.2.

VERDICT -> Wide-K compose at M=256 is 973
  us at 2800, ~4.99x square, ~6.53x native
  s4 149, ~2.04x s8 477.4. throttle=0.
  One-card. Do not freeze. Rank us. Next:
  sibling N=17408 vs sibling K=17408.

### 2026-09-03ab - K3/K6 E2M1 two-term 4x8 A-db M=256 K=17408 sibling card0

CONTEXT -> card1 compose M=256 K=17408 was
  973 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_db48_m256_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 966.937 us, pipe_host
  964.294 vs card1 973.110 vs s4 149.0 vs
  s8 477.4. Spread ~0.9%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 M=256 wide-K floor 968.7 us
  at 2800 both cards. ~4.97x square,
  ~6.50x s4 149, ~2.03x s8 477.4. Rank us.

### 2026-09-03ac - K3/K6 E2M1 two-term 4x8 A-db M=256 N=17408 sibling card1

CONTEXT -> card0 compose M=256 N=17408 was
  986 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  compose_e2m1_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k3_e2m1_db48_m256_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card1: event 978.516 us, pipe_host
  982.879 vs card0 985.644 vs s4 140.0 vs
  s8 469.8. Spread ~0.3%.

VERDICT -> Sibling matches. New E2M1
  two-term 4x8 M=256 wide-N floor 984.3 us
  at 2800 both cards. ~5.05x square,
  ~7.03x s4 140, ~2.10x s8 469.8. Qwen FFN
  compose M=256 map is closed. Rank us.
  Next: compose on 4-acc M=256 vs nibble
  LUT M=64 N=17408.

### 2026-09-03ad - K6 closed-form nibble->s8 is 134.8 us

CONTEXT -> Family-A merge LUT is 158 us
  at held 2800. CONFIG prior: IEEE E2M1
  mag = (e==0)? m : (2+m)<<(e-1) needs
  no 16-entry table. Same RC=4 8x2-N
  tile, packed E2M1 B. Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0
  || --card 1. NT=2 spin=4000 M=1 5120.
  New numeric (closed-form decode).

COMMAND ->
  ```
  gpu-run --card N kernels/nvfp4/bin/nibble_lut_scf --nt 2 --m 1 --n 5120 --k 5120 --spin 4000 --card N
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  card0 pipe_host 134.756 us packed-B
  97.557 GB/s. card1 134.783 us /
  97.542 GB/s. vs merge 158 vs s8 34
  vs W8A8 44. Spread ~0.02%.

VERDICT -> New Family-A s8-A floor
  134.8 us at 2800 both cards. ~1.17x
  merge LUT. Still ~4.0x s8 34 and
  ~3.06x W8A8 44. Keep packed E2M1
  in HBM. Rank us.

### 2026-09-03ae - K6 12-idea NVFP4 sprint write-downs

CONTEXT -> User: try all 12 crazy
  NVFP4 spoofs and write down what we
  come across, including refusals.
  Napkin is CONFIG. Measure on cards.
  Backend named per arm.

CONFIG -> sycl+l0 standalone icpx
  2026.1.1 AOT intel_gpu_bmg_g31 unless
  noted pytorch-xpu. gpu-run --card N.
  Never bitcast E2M1 onto s4 as a
  supposed integer path.

COMMAND ->
  ```
  kernels/nvfp4/compile_extra.sh ...
  gpu-run --card 0 run_sprint_card0.sh 0
  gpu-run --card 1 run_sprint_card1.sh 1
  python3 kernels/nvfp4/hist_nvfp4.py
  python3 kernels/nvfp4/persist_vram.py
  python3 kernels/nvfp4/g16_scale_landmine.py
  gpu-run --card N run_bench_nvfp4_m1.sh N
  ```

RESULT -> twelve write-downs. Logs
  under results/k6/.

1 Sparse hi-plane / skip-hi. Real
  Qwen3.8 nvfp4-radixark FFN nibbles
  (8 tensors, 89e6 nibbles each):
  ov_frac 0.2464-0.2505, zeros 3.42-
  3.47% per sign. Uniform-ish, not
  sparse. lo-only compose (drop hi
  DPAS) both-card held 2800: pipe
  16.342/16.352 us cosine=0.760548
  max_abs=8.1953 ok=0. us matches
  s4 16.5. Cheap and wrong. P(all
  hi zero in k64) ~0. Stop skip-hi.

2 Dyadic {1,2,4}/s2. dyadic_s2
  COMPILE_OK. card1 256^3 max_abs=0
  ok=1, 5.906 us (clocks not held).
  s2 range is [-2,1], not E2M1.
  4-plane E2M1 not fused. s2xs4
  COMPILE_REFUSED (dpas.hpp size
  assert 2048==1024). Native s2xs2
  already lit in K2.

3 Mixed dpas. sprint_dpas_mix:
  MIX_OK s8A_s4B and s4A_s8B both
  cards (runtime lights, no host
  s32 oracle). s2xs4/s4xs2 skipped
  after compile static_assert.

4 256-entry product LUT GEMV.
  prod_lut_gemv W4A4 16x16 table
  M=1 5120. max_abs=0 ok=1 both
  cards. card1 697.042 us end
  cur=1050. card0 1105.573 us
  start 1983 end 2800. Clock-noisy,
  ~5-8x closed-form 134.8. Stop
  naive scalar GEMV.

5 Closed-form. 134.8 us both-card
  held 2800. See 03ad.

6 Incumbent nvfp4_gemm_w4a16.
  Stock mtp6 image: OP_ABSENT.
  v028 _xpu_C.abi3.so load_library
  OK if the image _xpu_C is NOT
  imported first (TORCH_LIBRARY
  collision). Packed NT: B.stride(0)
  must be 1. After M=64 heat,
  us_bench M=1 5120: folded-bf16
  scale 36.809/37.169 us; f8scale
  38.448/39.611 us. out bf16
  [1,5120]. Clocks NOT held 2800
  (card0 freq mostly 2800 then
  1583; card1 1750/1383, never
  2800). Same us class as W8A8
  44, not LUT 135. A is bf16, not
  s8. No E2M1 cosine this dump.
  Do not freeze vs held-2800 s8 34.

7 Explicit bitcast E2M1 nibble as
  s4 two's complement vs E2M1 q
  oracle. Both cards. check 8x16x64
  max_abs=352 ok=0. timed 256^3
  max_abs=1408 ok=0. Explicit
  negative CONFIRMED.

8 Group-16 e4m3. Hand s8
  dpas<4,4> K=16 COMPILE_REFUSED:
  "Systolic depth must be equal
  to 8". CPU: one scale after
  k32 vs two g16 scales differs
  (-288 vs -720). oneDNN
  nvfp4_gemm_w4a16_f8scale LIGHTS
  (~39 us, idea 6). JIT isolates
  g16. Hand integer dpas cannot.

9 Histogram. qwen3.8-27b
  nvfp4-radixark model-00001.
  1043 tensors, 73 U8. FFN
  down/gate/up layers 0,1,10:
  ov_frac 0.2464-0.2505. hist
  nearly uniform. Sparse-hi
  prior DIES on this checkpoint.

10 MXFP4. hf_quant_config
  MIXED_PRECISION: NVFP4 193
  all group_size 16, MXFP4 0,
  FP8 208 (attn). Third format,
  not this checkpoint.

11 Persist s8 vs resident 4-bit.
  CPU envelope 3 shards: U8
  packed 8.561 GiB, s8 unpack
  17.122, F8 7.789, bf16 4.066.
  resident 20.416 GiB fits 30.3.
  persist-s8 weights-only 28.977
  GiB leaves ~1.3 GiB, not a
  serve. Matches "8B served, 27B
  s8 did not fit". Load-time s8
  ctrl was 34.5 us vs LUT 158
  (03a); VRAM is the 27B gate.

12 ISA toys. skip-hi = (1).
  s2xs8 already lit K2. s8xs4
  lights (3). s2xs4 refuse (2).
  SLM LUT and u4+sign not built
  this sprint.

VERDICT -> DoA split stands, with
  new numbers. Cannot feed XMX:
  false (LUT, compose, mixed
  dpas, oneDNN W4A16). As fast
  as s8/W8A8 with s8-A: false
  (134.8 vs 34/44). Can beat
  s8/W8A8 if A is s4: true at
  decode 5120 (compose 28.5).
  oneDNN W4A16 bf16-A is ~37 us
  (clocks not held) vs LUT 135.
  Bitcast is wrong. Sparse-hi is
  dead on this ckpt. Stop skip-hi,
  iselect, naive product GEMV,
  s2xs4, hand K=16 dpas.
  Family-A floor is now 134.8.
  Rank us. Never bitcast s4.

### 2026-09-03af - K3/K6 E2M1 two-term 4-acc M=256 card0

CONTEXT -> 4x8 A-db compose is 194.9 us at
  M=256 vs s4 4-acc 48.6. Steal two-term
  E2M1 onto the 4-acc tile. New geometry.
  One-card. A=s4. Never bitcast. Napkin:
  2x s4 ~97 us if the two DPAS terms are
  compute-bound.

CONFIG -> sycl+l0, standalone
  compose_e2m1_w48m4, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  RC=8 4-acc NT=2 U=8 (256 s4 dpas.8x8
  lo+hi), wg 4x8 k128, pack=2,
  grf_size<256>, no SLM. spin=512
  warmup=10 iters=20. M=256 N=K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k3_e2m1_w48m4_m256.sh 0 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: NT=2 256x dpas.8x8 rW:s4
  rA:s4, store_block2d d16, grf_count 128,
  no slm_size. cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 412.646 us, pipe_host
  411.303 vs 4x8 compose 194.9 vs s4 48.6
  vs s8 128 vs W8A8 75 vs napkin 97. Ratio
  411.3/48.6 ~8.46x. min/max 392.9-439.7.

VERDICT -> 4-acc compose loses to 4x8 A-db
  194.9 (~2.11x) and ~8.46x native s4.
  Napkin 2x died. GRF 128. One-card. Do
  not freeze 411 us. Rank us.

### 2026-09-03ag - K6 nibble LUT 4x8 A-db M=64 N=17408 card1

CONTEXT -> LUT 4x8 A-db is 392.4 us at
  M=64 N=5120. s8 N=17408 is 338.9. s4
  94.7 (~2.81x). FFN-up prefill. One-card.
  Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=64 N=17408
  K=5120. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_db48_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 1028.037 us, pipe_host
  1037.007 vs 5120 392.4 vs s8 338.9 vs
  s4 94.7 vs compose 326.9. Ratio
  1037/392.4 ~2.64x, near s4's 2.81x.
  min/max 1025.5-1040.5.

VERDICT -> Wide-N LUT is a real 1037 us
  at 2800, ~2.64x square, ~3.06x s8 338.9.
  Better N-scale than compose 4.76x. Still
  a loss vs s8. One-card. Do not freeze.
  Rank us. Next: sibling LUT N=17408 vs
  LUT M=64 K=17408.

### 2026-09-03ah - K6 nibble LUT 4x8 A-db N=17408 sibling card0

CONTEXT -> card1 LUT M=64 N=17408 was
  1037 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_db48_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 1034.781 us, pipe_host
  1027.424 vs card1 1037.007 vs s8 338.9
  vs s4 94.7. Spread ~0.9%.

VERDICT -> Sibling matches. New 4x8 A-db
  LUT wide-N floor 1032 us at 2800 both
  cards. ~2.63x square, ~3.05x s8 338.9.
  Rank us.

### 2026-09-03ai - K6 nibble LUT 4x8 A-db M=64 K=17408 card1

CONTEXT -> LUT N=17408 is 1032 us (~2.63x).
  s8 K=17408 is 374.7. s4 106.0 (~3.15x).
  Napkin K-linear 392.4*17408/5120 ~1334
  us. FFN-down prefill. One-card. Never
  bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=64 N=5120
  K=17408. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_db48_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 1331.870 us, pipe_host
  1332.672 vs 5120 392.4 vs N=17408 1032
  vs s8 374.7 vs s4 106.0 vs compose 403.4
  vs napkin 1334. Ratio 1332.7/392.4
  ~3.40x, K-linear. min/max 1329.1-1346.0.

VERDICT -> Wide-K LUT is 1333 us at 2800,
  K-linear, ~3.56x s8 374.7. Napkin held.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling LUT K=17408 vs
  LUT M=256.

### 2026-09-03aj - K6 nibble LUT 4x8 A-db K=17408 sibling card0

CONTEXT -> card1 LUT M=64 K=17408 was
  1333 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_db48_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 1330.865 us, pipe_host
  1332.410 vs card1 1332.672 vs s8 374.7
  vs s4 106.0. Spread ~0.02%.

VERDICT -> Sibling matches. New 4x8 A-db
  LUT wide-K floor 1333 us at 2800 both
  cards. K-linear ~3.40x, ~3.56x s8 374.7.
  Qwen FFN LUT M=64 map is closed. Rank us.

### 2026-09-03ak - K6 nibble LUT 4x8 A-db M=256 card1

CONTEXT -> LUT 4x8 is 392.4 us at M=64.
  s8 4-acc is 128 us at M=256. compose
  194.9. Napkin M-linear 1570 us. Prefill
  on the M=64 LUT tile. One-card. Never
  bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=256 N=K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_db48_m256.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card1: event 1193.964 us, pipe_host
  1207.283 vs M=64 392.4 vs s8 128 vs
  compose 194.9 vs W8A8 75 vs napkin 1570.
  Ratio 1207/392.4 ~3.08x, under M-linear.
  min/max 1174.5-1267.8.

VERDICT -> LUT M=256 is 1207 us at 2800,
  ~3.08x M=64, ~9.4x s8 128, ~6.2x compose
  194.9. Decode/prefill LUT tile is not a
  M=256 floor. One-card. Do not freeze.
  Rank us. Next: sibling LUT M=256 vs
  closed-form LUT on 4x8 A-db.

### 2026-09-03al - K6 nibble LUT 4x8 A-db M=256 sibling card0

CONTEXT -> card1 LUT M=256 was 1207 us at
  2800, numeric closed. Sibling swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_db48_m256.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 1194.062 us, pipe_host
  1198.437 vs card1 1207.283 vs s8 128 vs
  compose 194.9. Spread ~0.7%.

VERDICT -> Sibling matches. New 4x8 A-db
  LUT M=256 floor 1203 us at 2800 both
  cards. ~3.07x M=64, ~9.4x s8 128. Rank
  us.

### 2026-09-03am - K6 closed-form LUT 4x8 A-db M=64 card1

CONTEXT -> merge LUT 4x8 is 392.4 us.
  closed-form scf is 134.8 us at decode
  vs merge 158 (~1.17x). Steal scf onto
  the 4x8 A-db tile. New geometry.
  One-card. Napkin 392.4*134.8/158 ~335
  us. Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=64 N=K=5120.
  Packed E2M1 B. exp/mant shift decode.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_m64.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc: NT=2 64x dpas.8x8 rW:b
  rA:b, grf_count 128, no slm_size.
  cosine=1.0 max_abs=0. timed act=cur=2800
  throttle=0.
  M=64 card1: event 331.672 us, pipe_host
  331.554 vs merge 392.4 vs scf decode
  134.8 vs s8 75 vs W8A8 46 vs napkin 335.
  Ratio 392.4/331.6 ~1.18x. min/max
  331.46-333.65.

VERDICT -> Closed-form 4x8 LUT is a real
  331.6 us at 2800, ~1.18x merge, napkin
  held. Still ~4.42x s8 75. One-card. Do
  not freeze 332 us until card0. Rank us.
  Next: sibling scf 4x8 vs scf 4x8 M=256.

### 2026-09-03an - K6 closed-form LUT 4x8 A-db M=64 sibling card0

CONTEXT -> card1 closed-form 4x8 M=64 was
  331.6 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_m64.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 330.823 us, pipe_host
  331.665 vs card1 331.554 vs merge 392.4
  vs s8 75 vs W8A8 46. Spread ~0.03%.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT floor 331.6 us at 2800
  both cards. ~1.18x merge 392.4, still
  ~4.42x s8 75. Rank us.

### 2026-09-03ao - K6 closed-form LUT 4x8 A-db M=256 card1

CONTEXT -> scf 4x8 is 331.6 us at M=64.
  merge LUT M=256 is 1203. s8 4-acc 128.
  compose 194.9. Napkin 331.6*3.08 ~1021.
  Prefill on the M=64 scf tile. One-card.
  Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=256 N=K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_m256.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card1: event 1069.130 us, pipe_host
  1089.132 vs M=64 331.6 vs merge 1203 vs
  s8 128 vs compose 194.9 vs W8A8 75 vs
  napkin 1021. Ratio 1089/331.6 ~3.28x.
  min/max 1008.1-1127.7.

VERDICT -> Closed-form LUT M=256 is 1089
  us at 2800, ~3.28x M=64, ~1.10x merge
  1203, ~8.51x s8 128. Napkin 1021 missed
  ~6.7%. One-card. Do not freeze 1089 us
  until card0. Rank us. Next: sibling scf
  M=256 vs scf M=64 N=17408.

### 2026-09-03ap - K6 closed-form LUT 4x8 A-db M=256 sibling card0

CONTEXT -> card1 closed-form 4x8 M=256 was
  1089 us at 2800, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_m256.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=256 card0: event 1088.057 us, pipe_host
  1077.148 vs card1 1089.132 vs M=64 331.6
  vs merge 1203 vs s8 128 vs compose 194.9.
  Spread ~1.1%. min/max 1021.5-1246.0.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT M=256 floor 1083 us at
  2800 both cards. ~3.27x M=64, ~1.11x
  merge 1203, ~8.46x s8 128. Rank us.

### 2026-09-03aq - K6 closed-form LUT 4x8 A-db M=64 N=17408 card1

CONTEXT -> scf 4x8 is 331.6 us at square.
  merge LUT N=17408 is 1032 (~2.63x). s8
  338.9. s4 94.7. compose 326.9. Napkin
  331.6*1032/392.4 ~872. FFN-up prefill.
  One-card. Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=64 N=17408
  K=5120. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2783 cur=2800 throttle=1.
  M=64 card1: event 879.787 us, pipe_host
  877.318 vs 5120 331.6 vs merge 1032 vs
  s8 338.9 vs s4 94.7 vs compose 326.9 vs
  napkin 872. Ratio 877.3/331.6 ~2.64x.
  min/max 872.4-923.5.

VERDICT -> Wide-N closed-form LUT is 877
  us at 2783/2800, napkin held, ~2.64x
  square, ~1.18x merge 1032, ~2.59x s8
  338.9. Throttle=1. One-card. Do not
  freeze until card0. Rank us. Next:
  sibling scf N=17408 vs scf M=64 K=17408.

### 2026-09-03ar - K6 closed-form LUT 4x8 A-db N=17408 sibling card0

CONTEXT -> card1 closed-form 4x8 N=17408
  was 877 us at 2783/2800, throttle=1,
  numeric closed. Sibling swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2767 cur=2800 throttle=1.
  M=64 card0: event 883.708 us, pipe_host
  882.536 vs card1 877.318 vs 5120 331.6
  vs merge 1032 vs s8 338.9 vs s4 94.7.
  Spread ~0.6%. min/max 879.7-900.1.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT wide-N floor 880 us at
  ~2770/2800 both cards, throttle=1.
  ~2.65x square, ~1.17x merge 1032,
  ~2.60x s8 338.9. Rank us.

### 2026-09-03as - K6 closed-form LUT 4x8 A-db M=64 K=17408 card1

CONTEXT -> scf N=17408 is 880 us (~2.65x).
  merge LUT K=17408 is 1333 (~3.40x). s8
  374.7. s4 106.0. compose 403.4. Napkin
  331.6*1333/392.4 ~1127. FFN-down
  prefill. One-card. Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=64 N=5120
  K=17408. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card1: event 1123.245 us, pipe_host
  1123.392 vs 5120 331.6 vs N=17408 880
  vs merge 1333 vs s8 374.7 vs s4 106.0
  vs compose 403.4 vs napkin 1127.
  Ratio 1123.4/331.6 ~3.39x, K-linear.
  min/max 1121.7-1129.7.

VERDICT -> Wide-K closed-form LUT is 1123
  us at 2800, K-linear, napkin held,
  ~1.19x merge 1333, ~3.00x s8 374.7.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling scf K=17408 vs
  scf M=256 N=17408.

### 2026-09-03at - K6 closed-form LUT 4x8 A-db K=17408 sibling card0

CONTEXT -> card1 closed-form 4x8 K=17408
  was 1123 us at 2800, numeric closed.
  Sibling swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=cur=2800 throttle=0.
  M=64 card0: event 1124.917 us, pipe_host
  1125.896 vs card1 1123.392 vs 5120 331.6
  vs merge 1333 vs s8 374.7 vs s4 106.0.
  Spread ~0.2%. min/max 1121.6-1140.2.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT wide-K floor 1125 us at
  2800 both cards. K-linear ~3.39x,
  ~1.18x merge 1333, ~3.00x s8 374.7.
  Qwen FFN closed-form LUT M=64 map is
  closed. Rank us.

### 2026-09-03au - K6 closed-form LUT 4x8 A-db M=256 N=17408 card1

CONTEXT -> scf M=256 is 1083 us. scf
  M=64 N=17408 is 880 (~2.65x). s8 469.8.
  s4 140. compose 984.3. Napkin
  1083*880/331.6 ~2874. FFN-up prefill.
  One-card. Never bitcast.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=256 N=17408
  K=5120. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_m256_wide.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2683 cur=2800 throttle=1.
  M=256 card1: event 3106.740 us, pipe_host
  3113.855 vs 5120 1083 vs M=64 N=17408
  880 vs s8 469.8 vs s4 140 vs compose
  984.3 vs napkin 2874. Ratio 3114/1083
  ~2.88x. min/max 3094.8-3138.2.

VERDICT -> Wide-N closed-form LUT at
  M=256 is 3114 us at 2683/2800, ~2.88x
  square (M=64 was 2.65x), ~6.63x s8
  469.8. Napkin 2874 missed ~8.3%.
  Throttle=1. One-card. Do not freeze
  until card0. Rank us. Next: sibling
  scf M=256 N=17408 vs scf M=256 K=17408.

### 2026-09-03av - K6 closed-form LUT 4x8 A-db M=256 N=17408 sibling card0

CONTEXT -> card1 closed-form 4x8 M=256
  N=17408 was 3114 us at 2683/2800,
  throttle=1, numeric closed. Sibling
  swap.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_m256_wide.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2650 cur=2800 throttle=1.
  M=256 card0: event 3158.255 us, pipe_host
  3163.038 vs card1 3113.855 vs 5120 1083
  vs s8 469.8 vs s4 140 vs compose 984.3.
  Spread ~1.6%. min/max 3143.6-3186.5.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT M=256 wide-N floor 3138
  us at ~2660/2800 both cards, throttle=1.
  ~2.90x square, ~6.68x s8 469.8. Rank us.

### 2026-09-03aw - K6 closed-form LUT 4x8 A-db M=256 K=17408 card1

CONTEXT -> scf M=256 is 1083 us. scf
  M=64 K=17408 is 1125 (~3.39x). s8 477.4.
  s4 149. compose 968.7. Napkin
  1083*1125/331.6 ~3675. FFN-down prefill.
  One-card. Never bitcast. Host oracle is
  slow at this shape (~4 min wall).

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 1.
  RC=8 NT=2 U=16 spin=512. M=256 N=5120
  K=17408. Packed E2M1 B.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_k6_lut_scf_db48_m256_k17408.sh 1 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2783 cur=2800 throttle=1.
  M=256 card1: event 3407.661 us, pipe_host
  3412.241 vs 5120 1083 vs M=64 K=17408
  1125 vs s8 477.4 vs s4 149 vs compose
  968.7 vs napkin 3675. Ratio 3412/1083
  ~3.15x. min/max 3362.6-3462.0.

VERDICT -> Wide-K closed-form LUT at
  M=256 is 3412 us at 2783/2800, ~3.15x
  square (M=64 was 3.39x), ~7.15x s8
  477.4. Napkin 3675 was high ~7.2%.
  Throttle=1. One-card. Do not freeze
  until card0. Rank us. Next: sibling
  scf M=256 K=17408 vs held-clock
  nvfp4_gemm_w4a16 M=1.

### 2026-09-03ax - K6 closed-form LUT 4x8 A-db M=256 K=17408 sibling card0

CONTEXT -> card1 closed-form 4x8 M=256
  K=17408 was 3412 us at 2783/2800,
  throttle=1, numeric closed. Sibling
  swap. Host oracle ~4 min wall.

CONFIG -> sycl+l0, standalone
  nibble_lut_scf_db48, icpx 2026.1.1 AOT
  intel_gpu_bmg_g31, gpu-run --card 0.
  Same RC=8 NT=2 U=16 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_k6_lut_scf_db48_m256_k17408.sh 0 2 512
  ```

RESULT -> cosine=1.0 max_abs=0. timed
  act=2750/2767 cur=2800 throttle=1.
  M=256 card0: event 3433.594 us, pipe_host
  3444.610 vs card1 3412.241 vs 5120 1083
  vs s8 477.4 vs s4 149 vs compose 968.7.
  Spread ~0.9%. min/max 3400.2-3472.6.

VERDICT -> Sibling matches. New 4x8 A-db
  closed-form LUT M=256 wide-K floor 3428
  us at ~2760/2800 both cards, throttle=1.
  ~3.17x square, ~7.18x s8 477.4. Qwen
  FFN closed-form LUT M=256 map is closed.
  4x8 LUT loses to s8/s4/compose at FFN
  prefill. Rank us.

### 2026-09-03ay - K6 held-clock nvfp4_gemm_w4a16 M=1 card1

CONTEXT -> sprint unheld was 37.2 us
  (card1 cur 1750 then 1383, never 2800).
  s8 34. W8A8 44. LUT 134.8. Hold 2800
  via M=64 heat + M=1 spin=2000. A=bf16.
  No E2M1 cosine. One-card.

CONFIG -> pytorch-xpu on sycl+l0, image
  b70-sglang-xpu-int8-runtime:20260826-mtp6,
  v028 _xpu_C.abi3.so load_library, gpu-run
  --card 1. Packed NT stride(0)=1, g16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m1_hold.sh 1
  ```

RESULT -> HAS folded and f8scale. out
  bf16 [1,5120]. spin/timed act=cur=2800
  throttle=0. us_bench folded 34.395 vs
  unheld 37.169 vs s8 34 vs W8A8 44 vs
  LUT 134.8. f8scale 37.944 vs unheld
  39.611, also 2800 throttle=0.

VERDICT -> Held-clock folded w4a16 is
  34.4 us at 2800 card1, ~1.08x unheld
  37.2, under W8A8 44, ~3.9x LUT 135.
  Same us class as hand s8 34. A is
  bf16, not s8. One-card. Do not freeze
  34.4 us until card0. Do not call this
  a beat of s8. Rank us. Next: sibling
  w4a16 M=1 vs held-clock w4a16 M=64.

### 2026-09-03az - K6 held-clock nvfp4_gemm_w4a16 M=1 sibling card0

CONTEXT -> card1 held-clock folded
  w4a16 M=1 was 34.4 us at 2800.
  Sibling swap. A=bf16. No E2M1 cosine.

CONFIG -> pytorch-xpu on sycl+l0, image
  b70-sglang-xpu-int8-runtime:20260826-mtp6,
  v028 _xpu_C.abi3.so load_library, gpu-run
  --card 0. Packed NT, g16, spin=2000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m1_hold.sh 0
  ```

RESULT -> HAS folded and f8scale. out
  bf16 [1,5120]. spin/timed act=cur=2800
  throttle=0. us_bench folded 34.964 vs
  card1 34.395 vs unheld 36.809 vs s8 34
  vs W8A8 44 vs LUT 134.8. Spread ~1.6%.
  f8scale 37.738 vs card1 37.944, also
  2800 throttle=0.

VERDICT -> Sibling matches. New held-
  clock folded w4a16 M=1 floor 34.7 us
  at 2800 both cards. Same us class as
  s8 34, under W8A8 44, ~3.9x LUT 135.
  A is bf16, not s8. Do not call this a
  beat of s8. Rank us.

### 2026-09-03ba - K6 held-clock nvfp4_gemm_w4a16 M=64 card1

CONTEXT -> w4a16 M=1 is 34.7 us. W8A8
  M=64 46. s8 75. compose 68.7. LUT
  331.6. Napkin ~48-75. Prefill. A=bf16.
  One-card. No E2M1 cosine.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=1000 of M=64 then us_bench.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m64_hold.sh 1
  ```

RESULT -> out bf16 [64,5120]. timed
  act=2400 cur=2800 throttle=0. us_bench
  folded 36.761 vs M=1 34.7 vs W8A8 46
  vs s8 75 vs compose 68.7 vs LUT 331.6.
  f8scale 39.047, act=2250 cur=2800
  throttle=0.

VERDICT -> w4a16 M=64 is 36.8 us at
  act=2400/2800 card1, only ~1.06x M=1,
  under W8A8 46 and ~2.04x s8 75. Napkin
  48-75 was high. Act not 2800. One-card.
  Do not freeze until card0. Rank us.
  Next: sibling w4a16 M=64 vs held-clock
  w4a16 M=256.

### 2026-09-03bb - K6 nvfp4_gemm_w4a16 M=64 sibling card0

CONTEXT -> card1 w4a16 M=64 was 36.8 us
  at act=2400/2800. Sibling swap. More
  spin=2000 to try act=2800. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=2000 of M=64 then us_bench.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m64_hold.sh 0 2000
  ```

RESULT -> out bf16 [64,5120]. timed
  act=2150 cur=2800 throttle=0. us_bench
  folded 37.406 vs card1 36.761 vs M=1
  34.7 vs W8A8 46 vs s8 75. Spread ~1.7%.
  f8scale 39.153 vs card1 39.047, act=2200
  cur=2800. More spin did not raise act.

VERDICT -> Sibling matches. New w4a16
  M=64 floor 37.1 us both cards, cur=2800,
  act 2150-2400. ~1.07x M=1, under W8A8
  46, ~2.02x s8 75. Not a 2800-act hold.
  Rank us.

### 2026-09-03bc - K6 nvfp4_gemm_w4a16 M=256 card1

CONTEXT -> w4a16 M=64 is 37.1 us. W8A8
  75. s8 128. compose 194.9. LUT 1083.
  s4 48.6. Napkin ~40 if launch-bound,
  ~147 if 4x M=64. Prefill. A=bf16.
  One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=512 of M=256 then us_bench.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m256_hold.sh 1 512
  ```

RESULT -> out bf16 [256,5120]. timed
  act=2550/2500 cur=2800 throttle=0.
  us_bench folded 115.560 vs M=64 37.1
  vs W8A8 75 vs s8 128 vs compose 194.9
  vs LUT 1083 vs s4 48.6. Ratio
  115.6/37.1 ~3.12x. f8scale 114.151,
  act=2550 cur=2800.

VERDICT -> w4a16 M=256 is 116 us at
  act~2500/2800 card1, ~3.12x M=64, loses
  to W8A8 75 (~1.54x) and s4 48.6, beats
  s8 128 and LUT 1083. Launch-bound story
  dies at M=256. One-card. Do not freeze
  until card0. Rank us. Next: sibling
  w4a16 M=256 vs w4a16 M=1 N=17408.

### 2026-09-03bd - K6 nvfp4_gemm_w4a16 M=256 sibling card0

CONTEXT -> card1 w4a16 M=256 was 116 us
  at act~2500/2800. Sibling swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=512 of M=256 then us_bench.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m256_hold.sh 0 512
  ```

RESULT -> try1 folded 140.093 (act=2350,
  19% vs card1, discarded as cold). try2
  out bf16 [256,5120]. timed act=2350
  cur=2800 throttle=0. us_bench folded
  120.540 vs card1 115.560 vs M=64 37.1
  vs W8A8 75 vs s8 128. Spread ~4.2%.
  f8scale 116.389 vs card1 114.151.

VERDICT -> Sibling matches on try2. New
  w4a16 M=256 floor 118 us both cards,
  cur=2800, act 2350-2550. ~3.18x M=64,
  loses to W8A8 75 (~1.57x) and s4 48.6,
  beats s8 128. Rank us.

### 2026-09-03be - K6 nvfp4_gemm_w4a16 M=1 N=17408 card1

CONTEXT -> w4a16 M=1 square is 34.7 us.
  s8 N=17408 141.6. W8A8 158.1. s4 29.5.
  compose 103.5. Napkin 34.7*17408/5120
  ~118. Decode FFN-up. A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. M=64 heat on same B, spin=2000 of
  M=1 then us_bench. N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m1_wide_hold.sh 1 2000
  ```

RESULT -> out bf16 [1,17408]. timed
  act=2717 cur=2800 throttle=1. us_bench
  folded 97.050 vs square 34.7 vs s8 141.6
  vs W8A8 158.1 vs s4 29.5 vs compose
  103.5 vs napkin 118. Ratio 97.05/34.7
  ~2.80x. f8scale 89.023, act=2667
  cur=2800 throttle=1.

VERDICT -> Wide-N w4a16 M=1 is 97 us at
  2717/2800 card1, ~2.80x square, under
  napkin 118, beats s8 141.6 and W8A8
  158.1, loses to s4 29.5. Throttle=1.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling w4a16 N=17408
  vs w4a16 M=1 K=17408.

### 2026-09-03bf - K6 nvfp4_gemm_w4a16 M=1 N=17408 sibling card0

CONTEXT -> card1 w4a16 M=1 N=17408 was
  97 us at 2717/2800, throttle=1.
  Sibling swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. M=64 heat, spin=2000 of M=1.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m1_wide_hold.sh 0 2000
  ```

RESULT -> out bf16 [1,17408]. timed
  act=2700 cur=2800 throttle=1. us_bench
  folded 97.878 vs card1 97.050 vs square
  34.7 vs s8 141.6 vs W8A8 158.1 vs s4
  29.5. Spread ~0.8%. f8scale 89.449 vs
  card1 89.023, act=2650 throttle=1.

VERDICT -> Sibling matches. New w4a16
  M=1 wide-N floor 97 us both cards at
  ~2700/2800, throttle=1. ~2.80x square,
  beats s8 141.6 and W8A8 158.1, loses
  to s4 29.5. Rank us.

### 2026-09-03bg - K6 nvfp4_gemm_w4a16 M=1 K=17408 card1

CONTEXT -> w4a16 N=17408 is 97 us
  (~2.80x). s8 K=17408 261.6. W8A8 155.3.
  s4 53.4. compose 193.6. Napkin
  34.7*17408/5120 ~118. Decode FFN-down.
  A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. M=64 heat on same B, spin=2000 of
  M=1. N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m1_k17408_hold.sh 1 2000
  ```

RESULT -> out bf16 [1,5120]. timed
  act=2750 cur=2800 throttle=1. us_bench
  folded 101.270 vs square 34.7 vs N=17408
  97 vs s8 261.6 vs W8A8 155.3 vs s4 53.4
  vs compose 193.6 vs napkin 118. Ratio
  101.3/34.7 ~2.92x. f8scale 97.931,
  act=2733 cur=2800 throttle=1.

VERDICT -> Wide-K w4a16 M=1 is 101 us at
  2750/2800 card1, ~2.92x square, under
  napkin 118, beats s8 261.6 (~2.58x) and
  W8A8 155.3, loses to s4 53.4. Throttle=1.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling w4a16 K=17408
  vs w4a16 M=64 N=17408.

### 2026-09-03bh - K6 nvfp4_gemm_w4a16 M=1 K=17408 sibling card0

CONTEXT -> card1 w4a16 M=1 K=17408 was
  101 us at 2750/2800, throttle=1.
  Sibling swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. M=64 heat, spin=2000 of M=1.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m1_k17408_hold.sh 0 2000
  ```

RESULT -> out bf16 [1,5120]. timed
  act=2733 cur=2800 throttle=1. us_bench
  folded 101.909 vs card1 101.270 vs
  square 34.7 vs s8 261.6 vs W8A8 155.3
  vs s4 53.4. Spread ~0.6%. f8scale
  97.361 vs card1 97.931, act=2717
  throttle=1.

VERDICT -> Sibling matches. New w4a16
  M=1 wide-K floor 101 us both cards at
  ~2740/2800, throttle=1. ~2.92x square,
  beats s8 261.6 and W8A8 155.3, loses
  to s4 53.4. Qwen FFN w4a16 decode map
  is closed. Rank us.

### 2026-09-03bi - K6 nvfp4_gemm_w4a16 M=64 N=17408 card1

CONTEXT -> w4a16 M=64 square is 37.1 us.
  M=1 N=17408 is 97 (~2.80x). s8 338.9.
  s4 94.7. compose 326.9. LUT 880. Napkin
  37.1*97/34.7 ~104. FFN-up prefill.
  A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=512 of M=64 then us_bench.
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m64_wide_hold.sh 1 512
  ```

RESULT -> out bf16 [64,17408]. timed
  act=2300 cur=2800 throttle=0. us_bench
  folded 138.903 vs square 37.1 vs M=1
  N=17408 97 vs s8 338.9 vs s4 94.7 vs
  compose 326.9 vs LUT 880 vs napkin 104.
  Ratio 138.9/37.1 ~3.74x. f8scale
  138.595, act=2200 cur=2800.

VERDICT -> Wide-N w4a16 M=64 is 139 us at
  act=2300/2800 card1, ~3.74x square
  (napkin 104 missed), beats s8 338.9 and
  compose 326.9, loses to s4 94.7. Act
  not 2800. One-card. Do not freeze until
  card0. Rank us. Next: sibling w4a16
  M=64 N=17408 vs w4a16 M=64 K=17408.

### 2026-09-03bj - K6 nvfp4_gemm_w4a16 M=64 N=17408 sibling card0

CONTEXT -> card1 w4a16 M=64 N=17408 was
  139 us at act=2300/2800. Sibling swap.
  A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=512 of M=64 then us_bench.
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m64_wide_hold.sh 0 512
  ```

RESULT -> out bf16 [64,17408]. timed
  act=2050 cur=2800 throttle=0. us_bench
  folded 144.451 vs card1 138.903 vs
  square 37.1 vs M=1 N=17408 97 vs s8
  338.9 vs s4 94.7 vs compose 326.9 vs
  LUT 880 vs napkin 104. Spread ~4.0%.
  f8scale 141.560 vs card1 138.595,
  act=2100 cur=2800.

VERDICT -> Sibling matches. New w4a16
  M=64 wide-N floor 142 us both cards,
  cur=2800, act 2050-2300. ~3.82x square
  (napkin 104 missed), beats s8 338.9
  and compose 326.9, loses to s4 94.7.
  Act not 2800. Rank us.

### 2026-09-03bk - K6 nvfp4_gemm_w4a16 M=64 K=17408 card1

CONTEXT -> w4a16 M=64 square is 37.1 us.
  M=1 K=17408 is 101 (~2.92x). N-wide
  142. s8 374.7. s4 106.0. compose 403.4.
  LUT 1125. Napkin 37.1*101/34.7 ~108.
  FFN-down prefill. A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=512 of M=64 then us_bench.
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m64_k17408_hold.sh 1 512
  ```

RESULT -> out bf16 [64,5120]. timed
  act=2350-2400 cur=2800 throttle=0.
  us_bench folded 127.793 vs square 37.1
  vs M=1 K=17408 101 vs N-wide 142 vs
  s8 374.7 vs s4 106.0 vs compose 403.4
  vs LUT 1125 vs napkin 108. Ratio
  127.8/37.1 ~3.44x. ~K-linear (37.1*
  17408/5120 ~126). f8scale 129.001,
  act=2450-2400 cur=2800.

VERDICT -> Wide-K w4a16 M=64 is 128 us
  at act=2350-2400/2800 card1, ~3.44x
  square (napkin 108 missed, K-linear
  held), beats s8 374.7 and compose
  403.4, loses to s4 106.0. Act not
  2800. One-card. Do not freeze until
  card0. Rank us. Next: sibling w4a16
  M=64 K=17408 vs w4a16 M=256 N=17408.

### 2026-09-03bl - K6 nvfp4_gemm_w4a16 M=64 K=17408 sibling card0

CONTEXT -> card1 w4a16 M=64 K=17408 was
  128 us at act=2350-2400/2800. Sibling
  swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=512 of M=64 then us_bench.
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m64_k17408_hold.sh 0 512
  ```

RESULT -> out bf16 [64,5120]. timed
  act=2100-2200 cur=2800 throttle=0.
  us_bench folded 132.966 vs card1
  127.793 vs square 37.1 vs M=1 K=17408
  101 vs N-wide 142 vs s8 374.7 vs s4
  106.0 vs compose 403.4 vs LUT 1125 vs
  napkin 108. Spread ~4.0%. f8scale
  139.400 vs card1 129.001, act=2200-2300
  cur=2800.

VERDICT -> Sibling matches on folded.
  New w4a16 M=64 wide-K floor 130 us
  both cards, cur=2800, act 2100-2400.
  ~3.51x square, ~K-linear (126), beats
  s8 374.7 and compose 403.4, loses to
  s4 106.0. Qwen FFN w4a16 M=64 map is
  closed. Act not 2800. Rank us.

### 2026-09-03bm - K6 nvfp4_gemm_w4a16 M=256 N=17408 card1

CONTEXT -> w4a16 M=256 square is 118 us.
  M=64 N=17408 is 142 (~3.82x). s8 469.8.
  s4 140.0. compose 984.3. LUT 3138.
  Napkin 118*97/34.7 ~330. FFN-up
  prefill. A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=512 of M=256 then us_bench.
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m256_wide_hold.sh 1 512
  ```

RESULT -> out bf16 [256,17408]. timed
  act=2283-2267 cur=2800 throttle=1.
  us_bench folded 397.434 vs square 118
  vs M=64 N=17408 142 vs s8 469.8 vs s4
  140.0 vs compose 984.3 vs LUT 3138 vs
  napkin 330. Ratio 397.4/118 ~3.37x.
  ~N-linear (118*17408/5120 ~401).
  f8scale 390.800, act=2300-2283
  throttle=1.

VERDICT -> Wide-N w4a16 M=256 is 397 us
  at act~2280/2800 card1, ~3.37x square
  (napkin 330 missed, N-linear held),
  beats s8 469.8 and compose 984.3,
  loses to s4 140.0 (~2.84x). Throttle=1.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling w4a16 M=256
  N=17408 vs w4a16 M=256 K=17408.

### 2026-09-03bn - K6 nvfp4_gemm_w4a16 M=256 N=17408 sibling card0

CONTEXT -> card1 w4a16 M=256 N=17408 was
  397 us at act~2280/2800, throttle=1.
  Sibling swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=512 of M=256 then us_bench.
  N=17408 K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m256_wide_hold.sh 0 512
  ```

RESULT -> out bf16 [256,17408]. timed
  act=2250-2300 cur=2800 throttle=0 then
  1. us_bench folded 390.997 vs card1
  397.434 vs square 118 vs M=64 N=17408
  142 vs s8 469.8 vs s4 140.0 vs compose
  984.3 vs LUT 3138 vs napkin 330.
  Spread ~1.6%. f8scale 384.683 vs
  card1 390.800, act=2350-2317
  throttle=1.

VERDICT -> Sibling matches. New w4a16
  M=256 wide-N floor 394 us both cards,
  cur=2800, act 2250-2300, throttle=1.
  ~3.34x square, ~N-linear (401), beats
  s8 469.8 and compose 984.3, loses to
  s4 140.0. Rank us.

### 2026-09-03bo - K6 nvfp4_gemm_w4a16 M=256 K=17408 card1

CONTEXT -> w4a16 M=256 square is 118 us.
  M=64 K=17408 is 130 (~3.51x). N-wide
  394. s8 477.4. s4 149.0. compose 968.7.
  LUT 3428. Napkin 118*101/34.7 ~343.
  FFN-down prefill. A=bf16. One-card.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 1. Packed NT
  g16. spin=512 of M=256 then us_bench.
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 1 kernels/nvfp4/run_bench_nvfp4_m256_k17408_hold.sh 1 512
  ```

RESULT -> out bf16 [256,5120]. timed
  act=2317-2283 cur=2800 throttle=1.
  us_bench folded 375.885 vs square 118
  vs M=64 K=17408 130 vs N-wide 394 vs
  s8 477.4 vs s4 149.0 vs compose 968.7
  vs LUT 3428 vs napkin 343. Ratio
  375.9/118 ~3.19x. Under K-linear
  (118*17408/5120 ~401). f8scale
  361.498, act=2333-2300 throttle=1.

VERDICT -> Wide-K w4a16 M=256 is 376 us
  at act~2300/2800 card1, ~3.19x square
  (napkin 343 close, under K-linear),
  beats s8 477.4 and compose 968.7,
  loses to s4 149.0 (~2.52x). Throttle=1.
  One-card. Do not freeze until card0.
  Rank us. Next: sibling w4a16 M=256
  K=17408 vs oneDNN W8A8 M=256 N=17408.

### 2026-09-03bp - K6 nvfp4_gemm_w4a16 M=256 K=17408 sibling card0

CONTEXT -> card1 w4a16 M=256 K=17408 was
  376 us at act~2300/2800, throttle=1.
  Sibling swap. A=bf16.

CONFIG -> pytorch-xpu on sycl+l0, same
  v028 so, gpu-run --card 0. Packed NT
  g16. spin=512 of M=256 then us_bench.
  N=5120 K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/nvfp4/run_bench_nvfp4_m256_k17408_hold.sh 0 512
  ```

RESULT -> out bf16 [256,5120]. timed
  act=2283-2250 cur=2800 throttle=1.
  us_bench folded 378.110 vs card1
  375.885 vs square 118 vs M=64 K=17408
  130 vs N-wide 394 vs s8 477.4 vs s4
  149.0 vs compose 968.7 vs LUT 3428 vs
  napkin 343. Spread ~0.6%. f8scale
  367.919 vs card1 361.498, act=2267-2233
  throttle=1.

VERDICT -> Sibling matches. New w4a16
  M=256 wide-K floor 377 us both cards,
  cur=2800, act 2250-2317, throttle=1.
  ~3.19x square, under K-linear (401),
  beats s8 477.4 and compose 968.7,
  loses to s4 149.0. Qwen FFN w4a16
  M=256 map is closed. Rank us.

### 2026-09-03bq - K1/K4 oneDNN W8A8 M=256 N=17408 card1

CONTEXT -> W8A8 M=256 square is 75 us.
  W8A8 M=1 N=17408 is 158.1. w4a16 394.
  s8 469.8. s4 140.0. Napkin
  75*17408/5120 ~255. FFN-up prefill.
  GEMM-only. One-card.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 1.
  spin=512 of M=256 then us_bench.
  N=17408 K=5120. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m256_wide_hold.sh 1 512
  ```

RESULT -> out f16 [256,17408]. timed
  act=2517-2500 cur=2800 throttle=1.
  us_bench 248.116 vs square 75 vs
  w4a16 394 vs s8 469.8 vs s4 140.0 vs
  napkin 255. Ratio 248.1/75 ~3.31x.
  ~N-linear (255). cosine=1.000
  max_abs=0.062. 359 GB/s.

VERDICT -> Wide-N W8A8 M=256 is 248 us
  at act~2510/2800 card1, ~3.31x square
  (napkin 255 held), beats w4a16 394
  (~1.59x) and s8 469.8, loses to s4
  140.0 (~1.77x). Throttle=1. Numeric
  closed. One-card. Do not freeze until
  card0. Rank us. Next: sibling W8A8
  M=256 N=17408 vs W8A8 M=256 K=17408.

### 2026-09-03br - K1/K4 oneDNN W8A8 M=256 N=17408 sibling card0

CONTEXT -> card1 W8A8 M=256 N=17408 was
  248 us at act~2510/2800, throttle=1.
  Sibling swap. GEMM-only.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 0.
  spin=512 of M=256 then us_bench.
  N=17408 K=5120. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m256_wide_hold.sh 0 512
  ```

RESULT -> out f16 [256,17408]. timed
  act=2500-2467 cur=2800 throttle=1.
  us_bench 248.232 vs card1 248.116 vs
  square 75 vs w4a16 394 vs s8 469.8 vs
  s4 140.0 vs napkin 255. Spread ~0.05%.
  cosine=1.000 max_abs=0.062. 359 GB/s.

VERDICT -> Sibling matches. New W8A8
  M=256 wide-N floor 248 us both cards,
  cur=2800, act 2467-2517, throttle=1.
  ~3.31x square, N-linear held, beats
  w4a16 394 (~1.59x) and s8 469.8, loses
  to s4 140.0. Numeric closed. Rank us.

### 2026-09-03bs - K1/K4 oneDNN W8A8 M=256 K=17408 card1

CONTEXT -> W8A8 M=256 square is 75 us.
  W8A8 M=1 K=17408 is 155.3. N-wide 248.
  w4a16 377. s8 477.4. s4 149.0. Napkin
  75*155.3/44 ~265. FFN-down prefill.
  GEMM-only. One-card.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 1.
  spin=512 of M=256 then us_bench.
  N=5120 K=17408. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m256_k17408_hold.sh 1 512
  ```

RESULT -> out f16 [256,5120]. timed
  act=2500-2483 cur=2800 throttle=1.
  us_bench 228.094 vs square 75 vs N-wide
  248 vs w4a16 377 vs s8 477.4 vs s4
  149.0 vs napkin 265. Ratio 228.1/75
  ~3.04x. Under K-linear (255). cosine=
  1.000 max_abs=0.125. 391 GB/s.

VERDICT -> Wide-K W8A8 M=256 is 228 us
  at act~2490/2800 card1, ~3.04x square
  (under napkin 265 and K-linear 255),
  beats w4a16 377 (~1.65x) and s8 477.4,
  loses to s4 149.0 (~1.53x). Throttle=1.
  Numeric closed. One-card. Do not freeze
  until card0. Rank us. Next: sibling
  W8A8 M=256 K=17408 vs W8A8 M=64 N=17408.

### 2026-09-03bt - K1/K4 oneDNN W8A8 M=256 K=17408 sibling card0

CONTEXT -> card1 W8A8 M=256 K=17408 was
  228 us at act~2490/2800, throttle=1.
  Sibling swap. GEMM-only.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 0.
  spin=512 of M=256 then us_bench.
  N=5120 K=17408. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m256_k17408_hold.sh 0 512
  ```

RESULT -> out f16 [256,5120]. timed
  act=2517-2567 cur=2800 throttle=0 then
  1. us_bench 223.594 vs card1 228.094 vs
  square 75 vs N-wide 248 vs w4a16 377 vs
  s8 477.4 vs s4 149.0 vs napkin 265.
  Spread ~2.0%. cosine=1.000 max_abs=0.125.
  399 GB/s.

VERDICT -> Sibling matches. New W8A8
  M=256 wide-K floor 226 us both cards,
  cur=2800, act 2483-2567, throttle=1.
  ~3.01x square, under K-linear (255),
  beats w4a16 377 and s8 477.4, loses to
  s4 149.0. Qwen FFN W8A8 M=256 map is
  closed. Numeric closed. Rank us.

### 2026-09-03bu - K1/K4 oneDNN W8A8 M=64 N=17408 card1

CONTEXT -> W8A8 M=64 square is 46 us.
  W8A8 M=1 N=17408 is 158.1. w4a16 142.
  s8 338.9. s4 94.7. Napkin
  46*17408/5120 ~156. FFN-up prefill.
  GEMM-only. One-card.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 1.
  spin=512 of M=64 then us_bench.
  N=17408 K=5120. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m64_wide_hold.sh 1 512
  ```

RESULT -> out f16 [64,17408]. timed
  act=2783 cur=2800 throttle=1. us_bench
  201.221 vs square 46 vs M=1 N=17408
  158.1 vs w4a16 142 vs s8 338.9 vs s4
  94.7 vs napkin 156. Ratio 201.2/46
  ~4.37x. Superlinear vs 156. cosine=
  1.000 max_abs=0.062. 443 GB/s.

VERDICT -> Wide-N W8A8 M=64 is 201 us at
  2783/2800 card1, ~4.37x square (napkin
  156 missed). Loses to w4a16 142 (~1.42x)
  and s4 94.7, beats s8 338.9. Crossover:
  w4a16 wins M=1 and M=64 FFN-up; W8A8
  wins M=256 FFN-up. Throttle=1. Numeric
  closed. One-card. Do not freeze until
  card0. Rank us. Next: sibling W8A8
  M=64 N=17408 vs W8A8 M=64 K=17408.

### 2026-09-03bv - K1/K4 oneDNN W8A8 M=64 N=17408 sibling card0

CONTEXT -> card1 W8A8 M=64 N=17408 was
  201 us at 2783/2800, throttle=1, lost
  to w4a16 142. Sibling swap. GEMM-only.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 0.
  spin=512 of M=64 then us_bench.
  N=17408 K=5120. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m64_wide_hold.sh 0 512
  ```

RESULT -> out f16 [64,17408]. timed
  act=2783 cur=2800 throttle=1. us_bench
  202.772 vs card1 201.221 vs square 46
  vs w4a16 142 vs s8 338.9 vs s4 94.7 vs
  napkin 156. Spread ~0.8%. cosine=1.000
  max_abs=0.062. 440 GB/s.

VERDICT -> Sibling matches. New W8A8
  M=64 wide-N floor 202 us both cards at
  2783/2800, throttle=1. ~4.39x square,
  superlinear vs 156. Loses to w4a16 142
  (~1.42x) both cards. Crossover holds.
  Numeric closed. Rank us.

### 2026-09-03bw - K1/K4 oneDNN W8A8 M=64 K=17408 card1

CONTEXT -> W8A8 M=64 square is 46 us.
  N-wide 202. W8A8 M=1 K=17408 is 155.3.
  w4a16 130. s8 374.7. s4 106.0. Napkin
  46*155.3/44 ~162. FFN-down prefill.
  GEMM-only. One-card.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 1.
  spin=512 of M=64 then us_bench.
  N=5120 K=17408. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 1 kernels/w8_compare/run_w8_m64_k17408_hold.sh 1 512
  ```

RESULT -> out f16 [64,5120]. timed
  act=2733 cur=2800 throttle=1. us_bench
  184.009 vs square 46 vs N-wide 202 vs
  w4a16 130 vs s8 374.7 vs s4 106.0 vs
  napkin 162. Ratio 184.0/46 ~4.00x.
  Superlinear vs K-linear 156. cosine=
  1.000 max_abs=0.124. 484 GB/s.

VERDICT -> Wide-K W8A8 M=64 is 184 us at
  2733/2800 card1, ~4.00x square. Loses
  to w4a16 130 (~1.42x) and s4 106.0,
  beats s8 374.7. Same crossover as N-wide.
  Throttle=1. Numeric closed. One-card.
  Do not freeze until card0. Rank us.
  Next: sibling W8A8 M=64 K=17408 vs
  K2 s2 decode tile at 5120.

### 2026-09-03bx - K1/K4 oneDNN W8A8 M=64 K=17408 sibling card0

CONTEXT -> card1 W8A8 M=64 K=17408 was
  184 us at 2733/2800, throttle=1, lost
  to w4a16 130. Sibling swap. GEMM-only.

CONFIG -> pytorch-xpu on sycl+l0, mtp6
  int8_gemm_w8a8, gpu-run --card 0.
  spin=512 of M=64 then us_bench.
  N=5120 K=17408. Oracle after timed.

COMMAND ->
  ```
  gpu-run --card 0 kernels/w8_compare/run_w8_m64_k17408_hold.sh 0 512
  ```

RESULT -> out f16 [64,5120]. timed
  act=2733-2717 cur=2800 throttle=1.
  us_bench 177.372 vs card1 184.009 vs
  square 46 vs N-wide 202 vs w4a16 130 vs
  s8 374.7 vs s4 106.0 vs napkin 162.
  Spread ~3.7%. cosine=1.000 max_abs=0.123.
  502 GB/s.

VERDICT -> Sibling matches. New W8A8
  M=64 wide-K floor 181 us both cards at
  ~2725/2800, throttle=1. ~3.93x square,
  superlinear vs 156. Loses to w4a16 130
  (~1.39x) both cards. Qwen FFN W8A8 M=64
  map is closed. Numeric closed. Rank us.

### 2026-09-03by - K2 s2 RC=4 8x2-N decode tile card1

CONTEXT -> s8 RC=4 decode is 34 us. s4
  same tile is 16.5 us. W8A8 44. INT2
  silicon lit on 1024^3. First serving-
  shaped s2. IGC s2 [-2,1]. Never E2M1
  bitcast. Napkin ~8 us if 2x s4. One-card.

CONFIG -> backend sycl+l0, standalone
  icpx 2026.1.1 AOT intel_gpu_bmg_g31.
  dpas_s2_sc RC=4 NT=2 unroll=16 pack=4
  along K, scale-to-f16, pad M to RC.
  gpu-run --card 1. NT=2 spin=4000.

COMMAND ->
  ```
  compile_extra.sh dpas_s2_sc.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_s2_sc.sh 1 2 4000
  ```

RESULT -> COMPILE_OK. check 4x32x1024
  cosine=1.000 max_abs=0. timed M=1
  5120 act=cur=2800 throttle=0. event
  11.096 pipe_host 11.474 vs s4 16.5 vs
  s8 34 vs W8A8 44. M=4 pipe 11.458
  tracks. ~1.43x s4, ~2.96x s8. Napkin
  8 missed.

VERDICT -> First serving-shaped s2 decode
  is 11.5 us pipe_host at 2800 card1,
  numeric closed. Beats s4 16.5, not 2x
  s4. New dtype, one-card. Do not freeze
  11.5 us until card0. Rank pipe_host.
  Next: sibling s2 decode vs s2xs8
  serving-shaped.

### 2026-09-03bz - K2 s2 RC=4 decode sibling card0

CONTEXT -> card1 s2 decode was 11.5 us
  at 2800, cosine=1 max_abs=0. New dtype
  sibling. IGC s2 [-2,1]. Never E2M1
  bitcast.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s2_sc. gpu-run --card 0.
  NT=2 spin=4000. M=1 and M=4 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_sc.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 11.102 pipe_host 11.468 vs card1
  11.474 vs s4 16.5 vs s8 34. M=4 pipe
  11.476. Spread ~0.05%.

VERDICT -> Sibling matches. New s2 decode
  floor 11.5 us pipe_host both cards at
  2800. ~1.43x s4, ~2.96x s8. Numeric
  closed. Rank pipe_host.

### 2026-09-03ca - K2 s2xs8 RC=4 decode tile card1

CONTEXT -> Literature mix A=s8 B=s2,
  K=32 dpas (same OPC as s8). s2xs2 is
  11.5. s8 34. Napkin ~34 if paper rate.
  IGC s2 [-2,1]. Never E2M1 bitcast.
  One-card.

CONFIG -> backend sycl+l0, standalone
  AOT dpas_s2xs8_sc RC=4 NT=2 unroll=16
  packB=4 A=s8. gpu-run --card 1. NT=2
  spin=4000.

COMMAND ->
  ```
  compile_extra.sh dpas_s2xs8_sc.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_s2xs8_sc.sh 1 2 4000
  ```

RESULT -> COMPILE_OK. check 4x32x512
  cosine=1.000 max_abs=0. timed M=1
  5120 act=cur=2800 throttle=0. event
  13.557 pipe_host 14.140 vs s2 11.5 vs
  s4 16.5 vs s8 34 vs napkin 34. M=4
  pipe 13.962 tracks. ~2.41x s8, ~1.23x
  s2xs2.

VERDICT -> Mix lights and is numeric-
  closed. 14.1 us pipe_host at 2800
  card1. Beats s8 34 (paper same-rate
  napkin missed). Loses to s2xs2 11.5.
  New mix. One-card. Do not freeze 14.1
  us until card0. Rank pipe_host. Next:
  sibling s2xs8 vs K5 producer N=17408.

### 2026-09-03cb - K2 s2xs8 RC=4 decode sibling card0

CONTEXT -> card1 s2xs8 decode was 14.1
  us at 2800, cosine=1 max_abs=0. New
  mix sibling. A=s8 B=s2. Never E2M1
  bitcast.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s2xs8_sc. gpu-run --card 0.
  NT=2 spin=4000. M=1 and M=4 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2xs8_sc.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 13.583 pipe_host 13.971 vs card1
  14.140 vs s2 11.5 vs s4 16.5 vs s8 34.
  M=4 pipe 13.965. Spread ~1.2%.

VERDICT -> Sibling matches. New s2xs8
  decode floor 14.1 us pipe_host both
  cards at 2800. ~2.41x s8. Numeric
  closed. Rank pipe_host.

### 2026-09-03cc - K5 producer+GEMM M=1 N=17408 card1

CONTEXT -> Square pair is 44 us (prod
  ~10 + gemm ~33). s8 GEMM N=17408 is
  141.6. W8A8 158.1. Napkin 44*17408/5120
  ~151. Producer is over K=5120. One-card.

CONFIG -> backend sycl+l0, dpas_s8_prod
  RC=4 NT=2. gpu-run --card 1. M=1 and
  M=4 N=17408 K=5120. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_prod_wide.sh 1 2 4000
  ```

RESULT -> timed act=cur=2800 throttle=0.
  M=1 prod 10.818 gemm 142.625 pair_event
  153.581 pipe_host 154.033 vs square 44
  vs s8 GEMM 141.6 vs W8A8 158.1 vs
  napkin 151. cosine=1.000 max_abs=0.
  M=4 pipe 155.938 tracks. Extra ~11 us
  over GEMM, same as square.

VERDICT -> Wide-N producer+GEMM is 154
  us pipe_host at 2800 card1. ~3.48x
  square, N-linear held. Extra is still
  the ~11 us producer, not N. Beats
  W8A8 158.1. One-card. Do not freeze
  154 us until card0. Rank pipe_host.
  Next: sibling producer N=17408 vs
  producer K=17408.

### 2026-09-03cd - K5 producer+GEMM M=1 N=17408 sibling card0

CONTEXT -> card1 producer+GEMM N=17408 was
  154 us at 2800, cosine=1 max_abs=0.
  Extra ~11 us over GEMM. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8_prod RC=4 NT=2.
  gpu-run --card 0. M=1 and M=4
  N=17408 K=5120. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_prod_wide.sh 0 2 4000
  ```

RESULT -> timed act=cur=2800 throttle=0.
  M=1 prod 10.846 gemm 143.529 pair_event
  154.505 pipe_host 156.354 vs card1
  154.033 vs square 44 vs s8 GEMM 141.6
  vs W8A8 158.1. cosine=1.000 max_abs=0.
  M=4 pipe 155.764. Spread ~1.5%. Extra
  ~11 us over GEMM.

VERDICT -> Sibling matches. New
  producer+GEMM N=17408 floor 155 us
  pipe_host both cards at 2800. ~3.52x
  square, N-linear. Extra is still the
  ~11 us producer (K=5120), not N.
  Beats W8A8 158.1. Numeric closed.
  Rank pipe_host.

### 2026-09-03ce - K5 producer+GEMM M=1 K=17408 card1

CONTEXT -> Square pair is 44 us (prod
  ~10 + gemm ~33). N-wide pair 155.
  s8 GEMM K=17408 is 261.6. W8A8 155.3.
  Napkin prod 10*17408/5120~35 + 262
  ~297. Producer is over K. One-card.

CONFIG -> backend sycl+l0, dpas_s8_prod
  RC=4 NT=2. gpu-run --card 1. M=1 and
  M=4 N=5120 K=17408. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_prod_k17408.sh 1 2 4000
  ```

RESULT -> timed act=cur=2800 throttle=0.
  M=1 prod 33.099 gemm 261.068 pair_event
  294.305 pipe_host 294.453 vs square 44
  vs N-wide 155 vs s8 GEMM 261.6 vs
  W8A8 155.3 vs napkin 297. cosine=
  0.999995 max_abs=0.064. M=4 pipe
  295.423 tracks. Extra ~33 us over
  GEMM, K-linear. ok=1.

VERDICT -> Wide-K producer+GEMM is 294
  us pipe_host at 2800 card1. ~6.68x
  square. Extra is the ~33 us producer
  (K-linear), GEMM matches s8 261.6.
  Loses to W8A8 155.3 (~1.90x). Napkin
  297 hit. One-card. Do not freeze 294
  us until card0. Rank pipe_host. Next:
  sibling producer K=17408 vs mixed
  s8xs4 numeric oracle.

### 2026-09-03cf - K5 producer+GEMM M=1 K=17408 sibling card0

CONTEXT -> card1 producer+GEMM K=17408 was
  294 us at 2800, cosine=1 max_abs=0.064.
  Extra ~33 us K-linear. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8_prod RC=4 NT=2.
  gpu-run --card 0. M=1 and M=4
  N=5120 K=17408. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_prod_k17408.sh 0 2 4000
  ```

RESULT -> timed act=cur=2800 throttle=0.
  M=1 prod 33.102 gemm 260.284 pair_event
  293.518 pipe_host 294.411 vs card1
  294.453 vs square 44 vs N-wide 155 vs
  s8 GEMM 261.6 vs W8A8 155.3. cosine=
  0.999995 max_abs=0.064 ok=1. M=4 pipe
  295.006. Spread ~0.01%. Extra ~33 us.

VERDICT -> Sibling matches. New
  producer+GEMM K=17408 floor 294 us
  pipe_host both cards at 2800. ~6.68x
  square. Extra is K-linear producer.
  Loses to W8A8 155.3 (~1.90x). Qwen
  FFN producer decode map is closed.
  Rank pipe_host.

### 2026-09-03cg - K2 mixed s8xs4 host s32 oracle card1

CONTEXT -> Sprint MIX_OK s8A_s4B and
  s4A_s8B had no s32 oracle. s4 [-8,7].
  K=32 (A or B is s8, OPC=4). Never
  E2M1 bitcast. New numeric. One-card.

CONFIG -> backend sycl+l0, standalone
  AOT dpas_s8xs4. gpu-run --card 1.
  Check 8x16x32 and 32x32x128, timed
  256^3. Host unpacked s8*s4 s32.

COMMAND ->
  ```
  compile_extra.sh dpas_s8xs4.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4.sh 1
  ```

RESULT -> COMPILE_OK. All six rows
  max_abs=0 ok=1. s8A_s4B check 8x16x32
  27.1 us. s4A_s8B check 18.7 us.
  check2 32x32x128 both 20.5/20.8 us.
  timed 256^3 24.2 / 5.0 us (clocks
  not held; do not rank). bin_rc=0.

VERDICT -> Mixed s8xs4 is host-s32
  closed on card1. Both directions.
  K=32 mix, not s4xs4 K=64. Do not
  freeze numeric until card0. Do not
  quote 256^3 us. Rank the close.
  Next: sibling s8xs4 oracle vs
  GPTQ/AWQ s4 checkpoint.

### 2026-09-03ch - K2 mixed s8xs4 host s32 sibling card0

CONTEXT -> card1 s8xs4 oracle was
  max_abs=0 both mixes. New numeric
  sibling. s4 [-8,7]. K=32. Never
  E2M1 bitcast.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4. gpu-run --card 0.
  Check 8x16x32 and 32x32x128, timed
  256^3.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4.sh 0
  ```

RESULT -> All six rows max_abs=0 ok=1.
  s8A_s4B check 19.745 vs card1 27.130.
  s4A_s8B check 18.677 vs 18.677.
  check2 20.542/20.781. timed 256^3
  10.031/8.896 vs card1 24.2/5.0
  (clocks not held). bin_rc=0.

VERDICT -> Sibling matches. Mixed
  s8xs4 is host-s32 closed both cards.
  Do not quote 256^3 us. Rank the
  close.

### 2026-09-03ci - K6 GPTQ INT4 through ESIMD s4 card1

CONTEXT -> Qwen3.8-27B gptq-int4
  g128 sym pack i32. True INT4 XMX
  control (K6 arm 9). s4 [-8,7].
  Never E2M1 bitcast. One-card.

CONFIG -> backend sycl+l0, CPU unpack
  then AOT dpas_s4_ckpt. gpu-run
  --card 1. Layer0/1 FFN hist. Dump
  down_proj 256x256 s4=q-8. Synthetic
  s4 A, host s32.

COMMAND ->
  ```
  compile_extra.sh dpas_s4_ckpt.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4.sh 1
  ```

RESULT -> COMPILE_OK. 6/6 FFN tensors
  s4_ov=0 s4 in [-8,7] g_idx_linear=1.
  z_uniq=7 (all; zp stored as 7, code
  is q-8). dump 256x256. check 8x16x64
  max_abs=0 ok=1. tile 8x256x256
  max_abs=0 ok=1. bin_rc=0. Clocks not
  held; do not rank us.

VERDICT -> Real GPTQ INT4 codes are
  s4 and feed dpas<s4,s4> bit-exact
  vs host s32 on card1. Stored qzeros
  are 7, not 8. Integer path closed
  on this tile. One-card. Do not
  freeze until card0. Group-scale
  f16 epilogue is a later question.
  Next: sibling GPTQ s4 vs serving
  s8xs4 decode tile.

### 2026-09-03cj - K6 GPTQ INT4 ESIMD s4 sibling card0

CONTEXT -> card1 GPTQ s4 was max_abs=0
  on 8x16x64 and 8x256x256, s4_ov=0,
  qzeros=7. New numeric sibling.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s4_ckpt. gpu-run --card 0.
  Same dump path, layer0/1 hist.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4.sh 0
  ```

RESULT -> 6/6 FFN s4_ov=0 s4 in [-8,7]
  g_idx_linear=1 z_uniq=7. check 8x16x64
  max_abs=0 ok=1. tile 8x256x256
  max_abs=0 ok=1 vs card1 0/0. bin_rc=0.

VERDICT -> Sibling matches. GPTQ INT4
  codes feed ESIMD s4 both cards.
  Stored qzeros are 7. Integer path
  closed. Do not rank us. Group-scale
  f16 epilogue is next on this arm.

### 2026-09-03ck - K2 s8xs4 RC=4 decode tile card1

CONTEXT -> Mix A=s8 B=s4 pack=2 K=32
  dpas (OPC=4). s2xs8 is 14.1. s4 16.5.
  s8 34. Napkin ~34 if s8 rate, ~14 if
  s2xs8 B-bytes. s4 [-8,7]. Never E2M1
  bitcast. First scale-to-f16 of mix.
  One-card.

CONFIG -> backend sycl+l0, standalone
  AOT dpas_s8xs4_sc RC=4 NT=2 unroll=16
  packB=2 A=s8. gpu-run --card 1.
  NT=2 spin=4000.

COMMAND ->
  ```
  compile_extra.sh dpas_s8xs4_sc.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_sc.sh 1 2 4000
  ```

RESULT -> COMPILE_OK. check 4x32x512
  cosine=1.000 max_abs=0. timed M=1
  5120 act=cur=2800 throttle=0. event
  21.565 pipe_host 22.149 vs s2xs8 14.1
  vs s4 16.5 vs s8 34 vs W8A8 44 vs
  napkin 34. M=4 pipe 21.966 tracks.
  ~1.53x s8, loses to s4.

VERDICT -> Mix decode lights and is
  numeric-closed. 22.1 us pipe_host at
  2800 card1. Beats s8 34 (same-rate
  napkin missed). Loses to s4 16.5 and
  s2xs8 14.1. New mix. One-card. Do
  not freeze 22.1 us until card0. Rank
  pipe_host. Next: sibling s8xs4 decode
  vs GPTQ group-scale f16.

### 2026-09-03cl - K2 s8xs4 RC=4 decode sibling card0

CONTEXT -> card1 s8xs4 decode was 22.1
  us at 2800, cosine=1 max_abs=0. New
  mix sibling. A=s8 B=s4. Never E2M1
  bitcast.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_sc. gpu-run --card 0.
  NT=2 spin=4000. M=1 and M=4 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_sc.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 21.565 pipe_host 21.961 vs card1
  22.149 vs s2xs8 14.1 vs s4 16.5 vs
  s8 34. M=4 pipe 21.957. Spread ~0.9%.

VERDICT -> Sibling matches. New s8xs4
  decode floor 22.1 us pipe_host both
  cards at 2800. ~1.53x s8. Numeric
  closed. Rank pipe_host.

### 2026-09-03cm - K6 GPTQ s4 group-scale f16 card1

CONTEXT -> GPTQ INT4 codes already
  feed dpas<s4,s4> both cards. Next:
  apply g128 f16 scales in epilogue.
  Partial s32 per group, * scale, f16.
  Never E2M1. One-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s4_gptq. gpu-run --card 1.
  down_proj 256x256 dump + scales.
  Synthetic s4 A * 0.02. Host group
  s32 oracle.

COMMAND ->
  ```
  compile_extra.sh dpas_s4_gptq.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_sc.sh 1
  ```

RESULT -> COMPILE_OK. DUMP_SC gs=128
  ng=2 sc 0.0028-0.0102. check 8x16x128
  cosine=1.000 max_abs=7.6e-6 ok=1.
  tile 8x256x256 cosine=1.000 max_abs=0
  ok=1. bin_rc=0. Clocks not held.

VERDICT -> Group-scale f16 epilogue
  is numeric-closed on card1 vs host
  s32*scale. Real GPTQ scales applied.
  One-card. Do not freeze until card0.
  Do not rank us. Next: sibling GPTQ
  scale vs s8xs4 wide-N.

### 2026-09-03cn - K6 GPTQ s4 group-scale sibling card0

CONTEXT -> card1 GPTQ group-scale was
  cosine=1 max_abs=0 on 8x256x256.
  New numeric sibling.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s4_gptq. gpu-run --card 0.
  Same dump + g128 scales.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_sc.sh 0
  ```

RESULT -> check 8x16x128 cosine=1.000
  max_abs=7.6e-6 ok=1. tile 8x256x256
  cosine=1.000 max_abs=0 ok=1 vs card1
  0. bin_rc=0.

VERDICT -> Sibling matches. GPTQ s4
  group-scale f16 is numeric-closed
  both cards. Do not rank us.

### 2026-09-03co - K2 s8xs4 RC=4 N=17408 card1

CONTEXT -> s8xs4 square is 22.1 us.
  s8 N=17408 is 141.6. s4 29.5. W8A8
  158.1. Napkin N-linear 22.1*17408/5120
  ~75. Mix A=s8 B=s4. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_sc RC=4 NT=2. gpu-run
  --card 1. M=1 and M=4 N=17408 K=5120.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_sc_wide.sh 1 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 38.086 pipe_host 38.554 vs
  square 22.1 vs s4 29.5 vs s8 141.6
  vs W8A8 158.1 vs napkin 75. M=4 pipe
  38.575 tracks. ~1.74x square, not
  3.4x.

VERDICT -> Wide-N s8xs4 is 38.6 us
  pipe_host at 2800 card1. Under
  linear. Beats s8 141.6 and W8A8
  158.1, loses to s4 29.5 (~1.31x).
  Napkin 75 missed. One-card. Do not
  freeze 38.6 us until card0. Rank
  pipe_host. Next: sibling wide-N vs
  s8xs4 K=17408.

### 2026-09-03cp - K2 s8xs4 RC=4 N=17408 sibling card0

CONTEXT -> card1 s8xs4 N=17408 was
  38.6 us at 2800, cosine=1 max_abs=0.
  Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_sc. gpu-run --card 0.
  NT=2 spin=4000. M=1 and M=4 N=17408
  K=5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_sc_wide.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 38.026 pipe_host 38.637 vs card1
  38.554 vs square 22.1 vs s4 29.5 vs
  s8 141.6. M=4 pipe 38.531. Spread
  ~0.2%.

VERDICT -> Sibling matches. New s8xs4
  N=17408 floor 38.6 us pipe_host both
  cards at 2800. ~1.74x square. Numeric
  closed. Rank pipe_host.

### 2026-09-03cq - K2 s8xs4 RC=4 K=17408 card1

CONTEXT -> s8xs4 square 22.1. N-wide
  38.6. s8 K=17408 is 261.6. s4 53.4.
  W8A8 155.3. Napkin K-linear ~75.
  Mix A=s8 B=s4. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_sc RC=4 NT=2. gpu-run
  --card 1. M=1 and M=4 N=5120 K=17408.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_sc_k17408.sh 1 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 72.857 pipe_host 73.172 vs
  square 22.1 vs N-wide 38.6 vs s4 53.4
  vs s8 261.6 vs W8A8 155.3 vs napkin
  75. M=4 pipe 73.237 tracks. ~3.31x
  square, near K-linear.

VERDICT -> Wide-K s8xs4 is 73.2 us
  pipe_host at 2800 card1. Napkin 75
  hit. Beats s8 261.6 and W8A8 155.3,
  loses to s4 53.4 (~1.37x). More
  K-hostile than N-wide. One-card. Do
  not freeze 73.2 us until card0. Rank
  pipe_host. Next: sibling K=17408 vs
  GPTQ s4 serving decode.

### 2026-09-03cr - K2 s8xs4 RC=4 K=17408 sibling card0

CONTEXT -> card1 s8xs4 K=17408 was
  73.2 us at 2800, cosine=1 max_abs=0.
  Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_sc. gpu-run --card 0.
  NT=2 spin=4000. M=1 and M=4 N=5120
  K=17408.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_sc_k17408.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 72.815 pipe_host 73.188 vs card1
  73.172 vs square 22.1 vs N-wide 38.6
  vs s4 53.4 vs s8 261.6. M=4 pipe
  73.374. Spread ~0.02%.

VERDICT -> Sibling matches. New s8xs4
  K=17408 floor 73.2 us pipe_host both
  cards at 2800. ~3.31x square. Qwen
  FFN s8xs4 decode map is closed. Rank
  pipe_host.

### 2026-09-03cs - K6 GPTQ s4 RC=4 decode card1

CONTEXT -> GPTQ group-scale micro is
  closed. First serving-shaped GPTQ
  s4 on RC=4 8x2-N. Real down_proj
  5120x5120 s4 + g128 f16. s4 16.5.
  W8A8 44. Napkin 16.5+scale tax.
  Never E2M1. One-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s4_gptq_sc RC=4 NT=2 gs=128.
  gpu-run --card 1. M=1 and M=4
  5120. spin=4000.

COMMAND ->
  ```
  compile_extra.sh dpas_s4_gptq_sc.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_sc_dec.sh 1 2 4000
  ```

RESULT -> COMPILE_OK. dump 5120 gs=128
  sc 0.0024-0.108. check cosine=1.000
  max_abs=0. timed M=1 act=cur=2800
  throttle=0. event 29.448 pipe_host
  29.850 vs s4 16.5 vs s8xs4 22.1 vs
  s8 34 vs W8A8 44. cosine=1.000
  max_abs=6e-8 ok=1. M=4 pipe 29.890.
  ~1.81x s4.

VERDICT -> GPTQ serving decode is 29.9
  us pipe_host at 2800 card1. Numeric
  closed. Beats s8 34 and W8A8 44,
  loses to native s4 16.5 and s8xs4
  22.1. Scale tax ~13 us. One-card.
  Do not freeze 29.9 us until card0.
  Rank pipe_host. Next: sibling GPTQ
  decode vs s8xs4 M=64.

### 2026-09-03ct - K6 GPTQ s4 RC=4 decode sibling card0

CONTEXT -> card1 GPTQ s4 decode was
  29.9 us at 2800, cosine=1 max_abs=0.
  New serving sibling.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s4_gptq_sc. gpu-run
  --card 0. NT=2 spin=4000. M=1 and
  M=4 5120.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_sc_dec.sh 0 2 4000
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 29.424 pipe_host 29.955 vs card1
  29.850 vs s4 16.5 vs s8xs4 22.1 vs
  s8 34 vs W8A8 44. cosine=1.000
  max_abs=6e-8. M=4 pipe 29.899. Spread
  ~0.4%.

VERDICT -> Sibling matches. New GPTQ
  s4 decode floor 29.9 us pipe_host
  both cards at 2800. ~1.81x s4.
  Numeric closed. Rank pipe_host.

### 2026-09-03cu - K2 s8xs4 RC=4 M=64 card1

CONTEXT -> s8xs4 M=1 is 22.1. s4 4x8
  M=64 is 33.6. s8 75. W8A8 46.
  compose 8x2-N M=64 was 217.9 a loss.
  Decode tile at prefill. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_sc RC=4 NT=2. gpu-run
  --card 1. M=64 N=K=5120. spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_sc_m64.sh 1 2 512
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 112.865 pipe_host 114.146 vs
  M=1 22.1 vs s4 4x8 33.6 vs s8 75 vs
  W8A8 46 vs compose 217.9. ~5.16x
  M=1.

VERDICT -> Decode-tile s8xs4 at M=64
  is 114 us pipe_host at 2800 card1.
  Numeric closed. Loses to s4 33.6,
  s8 75, W8A8 46. Beats compose 217.9.
  Stop 8x2-N s8xs4 at prefill. One-card.
  Rank pipe_host. Next: GPTQ wide-N vs
  s8xs4 4x8 A-db M=64.

### 2026-09-03cv - K6 GPTQ s4 RC=4 N=17408 card0

CONTEXT -> GPTQ square is 29.9 us.
  s4 N=17408 29.5. s8 141.6. W8A8
  158.1. Napkin 29.9*17408/5120 ~102.
  gate_proj dump. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s4_gptq_sc RC=4 NT=2 gs=128.
  gpu-run --card 0. M=1 and M=4
  N=17408 K=5120. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_sc_wide.sh 0 2 4000
  ```

RESULT -> dump gate 5120x17408. check
  cosine=1.000 max_abs=0. timed M=1
  act=cur=2800 throttle=0. event 99.458
  pipe_host 100.028 vs square 29.9 vs
  s4 29.5 vs s8xs4 38.6 vs s8 141.6 vs
  W8A8 158.1 vs napkin 102. cosine=1
  max_abs=6e-8. M=4 pipe 100.488.
  ~3.35x square, near linear.

VERDICT -> Wide-N GPTQ is 100 us
  pipe_host at 2800 card0. Napkin 102
  hit. Beats s8 141.6 and W8A8 158.1,
  loses to s4 29.5 and s8xs4 38.6.
  One-card. Do not freeze 100 us until
  card1. Rank pipe_host.

### 2026-09-03cw - K2 s8xs4 4x8 A-db M=64 card1

CONTEXT -> 8x2-N mix M=64 was 114 us.
  s4 4x8 33.6. s8 4x8 75. W8A8 46.
  First s8xs4 on 4x8 A-db. A=s8 B=s4.
  Never E2M1. One-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s8xs4_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=64 N=K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  compile_extra.sh dpas_s8xs4_db48.cpp
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_db48.sh 1 2 512
  ```

RESULT -> COMPILE_OK. check cosine=1
  max_abs=0. timed M=64 act=cur=2800
  throttle=0. event 43.260 pipe_host
  43.286 vs 8x2-N 114 vs s4 33.6 vs
  s8 75 vs W8A8 46. ~2.64x 8x2-N.

VERDICT -> Mix 4x8 A-db is 43.3 us
  pipe_host at 2800 card1. Numeric
  closed. Beats s8 75 and W8A8 46,
  loses to s4 33.6 (~1.29x). New mix
  prefill tile. One-card. Do not freeze
  43.3 us until card0. Rank pipe_host.
  Next: sibling 4x8 vs sibling GPTQ
  N=17408.

### 2026-09-03cx - K2 s8xs4 4x8 A-db M=64 sibling card0

CONTEXT -> card1 s8xs4 4x8 A-db M=64
  was 43.3 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_db48. RC=8 wg 4x8
  A-db. gpu-run --card 0. M=64
  N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_db48.sh 0 2 512
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 42.667 pipe_host 43.431 vs
  card1 43.286 vs 8x2-N 114 vs s4
  33.6 vs s8 75 vs W8A8 46. Spread
  ~0.33%.

VERDICT -> Sibling matches. New s8xs4
  4x8 A-db M=64 floor 43.3 us
  pipe_host both cards at 2800.
  ~2.64x 8x2-N. Beats s8 75 and
  W8A8 46, loses to s4 33.6 (~1.29x).
  Rank pipe_host.

### 2026-09-03cy - K6 GPTQ s4 RC=4 N=17408 sibling card1

CONTEXT -> card0 GPTQ N=17408 was
  100 us at 2800, cosine=1
  max_abs=6e-8. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s4_gptq_sc. RC=4 NT=2
  gs=128. gpu-run --card 1. M=1 and
  M=4 N=17408 K=5120. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_sc_wide.sh 1 2 4000
  ```

RESULT -> dump gate 5120x17408.
  check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 99.463 pipe_host 100.068 vs
  card0 100.028 vs square 29.9 vs s4
  29.5 vs s8xs4 38.6 vs s8 141.6 vs
  W8A8 158.1 vs napkin 102. cosine=1
  max_abs=6e-8. M=4 pipe 100.646.
  Spread ~0.04%. ~3.35x square.

VERDICT -> Sibling matches. New GPTQ
  s4 N=17408 floor 100 us pipe_host
  both cards at 2800. Napkin 102 hit.
  Beats s8 141.6 and W8A8 158.1,
  loses to s4 29.5 and s8xs4 38.6.
  Rank pipe_host. Next: GPTQ K=17408
  vs s8xs4 4x8 M=256.

### 2026-09-03cz - K6 GPTQ s4 RC=4 K=17408 card0

CONTEXT -> GPTQ square 29.9. N-wide
  100. s4 K=17408 53.4. s8xs4 73.2.
  s8 261.6. W8A8 155.3. Napkin
  29.9*17408/5120 ~102. down_proj
  dump. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s4_gptq_sc RC=4 NT=2 gs=128.
  gpu-run --card 0. M=1 and M=4
  N=5120 K=17408. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_sc_k17408.sh 0 2 4000
  ```

RESULT -> dump down 17408x5120.
  check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 173.477 pipe_host 174.629 vs
  square 29.9 vs N-wide 100 vs s4
  53.4 vs s8xs4 73.2 vs s8 261.6 vs
  W8A8 155.3 vs napkin 102. cosine=1
  max_abs=0. M=4 pipe 174.561.
  ~5.84x square, not 3.4x.

VERDICT -> Wide-K GPTQ is 174.6 us
  pipe_host at 2800 card0. Napkin 102
  miss. Beats s8 261.6, loses to s4
  53.4, s8xs4 73.2, and W8A8 155.3
  (~1.12x). More K-hostile than
  N-wide. One-card. Do not freeze
  174.6 us until card1. Rank
  pipe_host.

### 2026-09-03da - K2 s8xs4 4x8 A-db M=256 card1

CONTEXT -> mix 4x8 M=64 is 43.3.
  s4 4-acc 48.6. s8 128. W8A8 75.
  compose 4x8 194.9. Napkin 43.3*4
  ~173. Same 4x8 A-db tile. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=256 N=K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_db48_m256.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=2767 cur=2800
  throttle=1. event 124.563
  pipe_host 123.272 vs M=64 43.3 vs
  s4 48.6 vs s8 128 vs W8A8 75 vs
  compose 194.9 vs napkin 173.
  ~2.85x M=64, under linear.

VERDICT -> Mix 4x8 M=256 is 123 us
  pipe_host at 2800 card1. Numeric
  closed. Beats s8 128 and compose
  194.9, loses to s4 48.6 (~2.54x)
  and W8A8 75 (~1.64x). Not a
  prefill floor. throttle=1. One-card.
  Do not freeze. Rank pipe_host.
  Next: sibling mix M=256 vs sibling
  GPTQ K=17408.

### 2026-09-03db - K2 s8xs4 4x8 A-db M=256 sibling card0

CONTEXT -> card1 mix 4x8 M=256 was
  123 us at 2800, throttle=1,
  cosine=1 max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_db48. RC=8 wg 4x8
  A-db. gpu-run --card 0. M=256
  N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_db48_m256.sh 0 2 512
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=256 act=2767/2750 cur=2800
  throttle=1. event 122.859 pipe_host
  122.830 vs card1 123.272 vs M=64
  43.3 vs s4 48.6 vs s8 128 vs W8A8
  75 vs compose 194.9. Spread ~0.36%.

VERDICT -> Sibling matches. Mix 4x8
  M=256 is 123 us pipe_host both
  cards, throttle=1. Still a loss vs
  s4 48.6 and W8A8 75. Stop 4x8 mix
  at M=256 prefill. Rank pipe_host.

### 2026-09-03dc - K6 GPTQ s4 RC=4 K=17408 sibling card1

CONTEXT -> card0 GPTQ K=17408 was
  174.6 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s4_gptq_sc. RC=4 NT=2
  gs=128. gpu-run --card 1. M=1 and
  M=4 N=5120 K=17408. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_sc_k17408.sh 1 2 4000
  ```

RESULT -> dump down 17408x5120.
  check cosine=1.000 max_abs=0.
  timed M=1 act=cur=2800 throttle=0.
  event 173.448 pipe_host 174.509 vs
  card0 174.629 vs square 29.9 vs
  N-wide 100 vs s4 53.4 vs s8xs4
  73.2 vs s8 261.6 vs W8A8 155.3.
  cosine=1 max_abs=0. M=4 pipe
  174.251. Spread ~0.07%. ~5.84x
  square.

VERDICT -> Sibling matches. New GPTQ
  s4 K=17408 floor 174.6 us
  pipe_host both cards at 2800.
  Qwen FFN GPTQ decode map is closed
  (29.9 / 100 / 174.6). Beats s8
  261.6, loses to s4 53.4, s8xs4
  73.2, and W8A8 155.3. Rank
  pipe_host. Next: mix 4x8 M=64
  N=17408 vs K=17408.

### 2026-09-03dd - K2 s8xs4 4x8 A-db M=64 N=17408 card0

CONTEXT -> mix 4x8 M=64 is 43.3.
  s4 N=17408 94.7. s8 338.9. W8A8
  202. Napkin 43.3*17408/5120 ~147.
  One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 0. M=64 N=17408
  K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_db48_wide.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=2783 cur=2800
  throttle=1. event 127.969
  pipe_host 126.931 vs square 43.3
  vs s4 94.7 vs s8 338.9 vs W8A8
  202 vs napkin 147. ~2.93x square,
  under linear.

VERDICT -> Wide-N mix 4x8 is 126.9
  us pipe_host at 2800 card0.
  Numeric closed. Beats s8 338.9
  and W8A8 202 (~1.59x), loses to
  s4 94.7 (~1.34x). throttle=1.
  One-card. Do not freeze 126.9 us
  until card1. Rank pipe_host.

### 2026-09-03de - K2 s8xs4 4x8 A-db M=64 K=17408 card1

CONTEXT -> mix 4x8 M=64 is 43.3.
  s4 K=17408 106.0. s8 374.7. W8A8
  181. Napkin ~147. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s8xs4_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=64 N=5120
  K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_db48_k17408.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 144.771 pipe_host 144.684 vs
  square 43.3 vs s4 106.0 vs s8
  374.7 vs W8A8 181 vs napkin 147.
  ~3.34x square, near linear.

VERDICT -> Wide-K mix 4x8 is 144.7
  us pipe_host at 2800 card1.
  Numeric closed. Beats s8 374.7
  and W8A8 181 (~1.25x), loses to
  s4 106.0 (~1.37x). One-card. Do
  not freeze 144.7 us until card0.
  Rank pipe_host. Next: sibling
  N-wide vs sibling K-wide.

### 2026-09-03df - K2 s8xs4 4x8 A-db M=64 K=17408 sibling card0

CONTEXT -> card1 mix 4x8 K=17408 was
  144.7 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_db48. RC=8 wg 4x8
  A-db. gpu-run --card 0. M=64 N=5120
  K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s8xs4_db48_k17408.sh 0 2 512
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 143.875 pipe_host 144.261 vs
  card1 144.684 vs square 43.3 vs s4
  106.0 vs s8 374.7 vs W8A8 181.
  Spread ~0.29%. ~3.34x square.

VERDICT -> Sibling matches. New mix
  4x8 M=64 K=17408 floor 144.7 us
  pipe_host both cards at 2800.
  Beats s8 374.7 and W8A8 181
  (~1.25x), loses to s4 106.0
  (~1.37x). Rank pipe_host.

### 2026-09-03dg - K2 s8xs4 4x8 A-db M=64 N=17408 sibling card1

CONTEXT -> card0 mix 4x8 N=17408 was
  126.9 us, throttle=1, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  binary dpas_s8xs4_db48. gpu-run
  --card 1. M=64 N=17408 K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s8xs4_db48_wide.sh 1 2 512
  ```

RESULT -> check cosine=1.000 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 129.344 pipe_host 129.215 vs
  card0 126.931 vs square 43.3 vs s4
  94.7 vs s8 338.9 vs W8A8 202.
  Spread ~1.80%. ~2.98x square.

VERDICT -> Sibling matches. New mix
  4x8 M=64 N=17408 floor 129 us
  pipe_host both cards. card0 126.9
  throttle=1, card1 129.2 throttle=0.
  Beats s8 338.9 and W8A8 202
  (~1.56x), loses to s4 94.7.
  Qwen FFN mix M=64 map is closed
  (43.3 / 129 / 144.7). Rank
  pipe_host. Next: GPTQ 8x2-N M=64
  vs M=256.

### 2026-09-03dh - K6 GPTQ s4 RC=4 8x2-N M=64 card0

CONTEXT -> GPTQ decode 29.9. s4 4x8
  33.6. mix 4x8 43.3. s8 75. W8A8
  46. mix 8x2-N lost 114. Decode
  tile at prefill. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s4_gptq_sc RC=4 NT=2 gs=128.
  gpu-run --card 0. M=64 N=K=5120.
  spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_sc_m64.sh 0 2 512
  ```

RESULT -> dump reuse. check cosine=1
  max_abs=0. timed M=64 act=2767
  cur=2800 throttle=1. event 124.219
  pipe_host 123.528 vs decode 29.9
  vs s4 33.6 vs mix 43.3 vs s8 75 vs
  W8A8 46 vs mix 8x2-N 114. cosine=1
  max_abs=3e-5. ~4.13x decode.

VERDICT -> GPTQ 8x2-N M=64 is 123.5
  us pipe_host at 2800 card0.
  Numeric closed. Loses to s4 33.6,
  mix 43.3, W8A8 46 (~2.68x), and
  s8 75. Stop 8x2-N GPTQ at M=64
  prefill. throttle=1. One-card. Do
  not freeze. Rank pipe_host.

### 2026-09-03di - K6 GPTQ s4 RC=4 8x2-N M=256 card1

CONTEXT -> GPTQ decode 29.9. s4
  4-acc 48.6. s8 128. W8A8 75.
  compose 8x2-N 607 a loss. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s4_gptq_sc RC=4 NT=2 gs=128.
  gpu-run --card 1. M=256 N=K=5120.
  spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_sc_m256.sh 1 2 512
  ```

RESULT -> dump reuse. check cosine=1
  max_abs=0. timed M=256 act=2550
  cur=2800 throttle=1. event 352.682
  pipe_host 354.611 vs s4 48.6 vs s8
  128 vs W8A8 75 vs compose 607.
  cosine=1 max_abs=3e-5. ~2.87x
  M=64, ~11.9x decode.

VERDICT -> GPTQ 8x2-N M=256 is 355
  us pipe_host at 2800 card1.
  Numeric closed. Beats compose 607,
  loses to s4 48.6, s8 128, and
  W8A8 75 (~4.73x). Stop 8x2-N GPTQ
  at M=256 prefill. throttle=1.
  One-card. Do not freeze. Rank
  pipe_host. Next: GPTQ 4x8 A-db
  M=64.

### 2026-09-03dj - K6 GPTQ s4 4x8 A-db M=64 card0

CONTEXT -> GPTQ 8x2-N M=64 was 123.5
  a loss. s4 4x8 33.6. mix 43.3.
  W8A8 46. Napkin 33.6*1.81 ~61.
  First GPTQ on 4x8 A-db. One-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s4_gptq_db48 RC=8 wg 4x8 A-db
  gs=128. gpu-run --card 0. M=64
  N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_gptq_s4_db48.sh 0 2 512
  ```

RESULT -> COMPILE_OK prior fire.
  dump reuse. check cosine=1
  max_abs=0. timed M=64 act=cur=2800
  throttle=0. event 102.375
  pipe_host 102.936 vs 8x2-N 123.5
  vs s4 33.6 vs mix 43.3 vs W8A8 46
  vs napkin 61. cosine=1
  max_abs=3e-5. ~3.06x s4, ~1.20x
  8x2-N.

VERDICT -> GPTQ 4x8 M=64 is 102.9
  us pipe_host at 2800 card0.
  Numeric closed. Beats 8x2-N 123.5,
  loses to s4 33.6, mix 43.3, and
  W8A8 46 (~2.24x). Napkin 61 miss.
  Not a prefill floor. One-card. Do
  not freeze. Rank pipe_host.

### 2026-09-03dk - K6 GPTQ s4 4x8 A-db M=256 card1

CONTEXT -> GPTQ 8x2-N M=256 was 355.
  s4 4-acc 48.6. mix 4x8 123 a loss.
  W8A8 75. Same 4x8 GPTQ tile.
  One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s4_gptq_db48. gpu-run --card 1.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_gptq_s4_db48_m256.sh 1 2 512
  ```

RESULT -> dump reuse. check cosine=1
  max_abs=0. timed M=256 act=cur=2800
  throttle=0. event 302.037
  pipe_host 302.722 vs 8x2-N 355 vs
  s4 48.6 vs mix 123 vs W8A8 75.
  cosine=1 max_abs=3e-5. ~2.94x
  M=64.

VERDICT -> GPTQ 4x8 M=256 is 303 us
  pipe_host at 2800 card1. Numeric
  closed. Beats 8x2-N 355, loses to
  s4 48.6, mix 123, and W8A8 75
  (~4.04x). Stop GPTQ 4x8 at prefill
  vs W8A8. One-card. Do not freeze.
  Rank pipe_host. Next: s2 4x8 A-db
  M=64 both-card.

### 2026-09-03dl - K2 s2 4x8 A-db M=64 card0

CONTEXT -> s2 decode 11.5. s4 4x8
  33.6. W8A8 46. First s2 on 4x8
  A-db. IGC s2 [-2,1]. Never E2M1.
  New dtype: both-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_db48 RC=8 wg 4x8 A-db
  pack=4. gpu-run --card 0. M=64
  N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 19.318 pipe_host 19.794 vs
  s2 decode 11.5 vs s4 33.6 vs W8A8
  46. ~1.70x s4, ~2.32x W8A8.

VERDICT -> s2 4x8 M=64 is 19.8 us
  pipe_host at 2800 card0. Numeric
  closed. New M=64 floor vs s4 33.6
  and W8A8 46. Rank pipe_host.

### 2026-09-03dm - K2 s2 4x8 A-db M=64 sibling card1

CONTEXT -> card0 s2 4x8 M=64 was
  19.8 us at 2800, cosine=1
  max_abs=0. New dtype sibling.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=64 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 19.667 pipe_host 20.814 vs
  card0 19.794 vs s4 33.6 vs W8A8
  46. Event spread ~1.8%. pipe
  spread ~5.2%.

VERDICT -> Sibling matches on event.
  New s2 4x8 A-db M=64 floor 20 us
  pipe_host both cards at 2800.
  Beats s4 33.6 (~1.68x) and W8A8
  46 (~2.21x). Rank pipe_host.
  Next: s2 4x8 M=64 N=17408 vs
  K=17408.

### 2026-09-03dn - K2 s2 4x8 A-db M=64 N=17408 card0

CONTEXT -> s2 4x8 M=64 is 20. s4
  N=17408 94.7. mix 129. W8A8 202.
  Napkin 20*17408/5120 ~68. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 0. M=64 N=17408
  K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48_wide.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 52.359 pipe_host 53.079 vs
  square 20 vs s4 94.7 vs mix 129 vs
  W8A8 202 vs napkin 68. ~2.65x
  square, under linear.

VERDICT -> Wide-N s2 4x8 is 53.1 us
  pipe_host at 2800 card0. Numeric
  closed. Beats s4 94.7, mix 129,
  and W8A8 202 (~3.81x). Napkin 68
  beat. One-card. Do not freeze 53.1
  us until card1. Rank pipe_host.

### 2026-09-03do - K2 s2 4x8 A-db M=64 K=17408 card1

CONTEXT -> s2 4x8 M=64 is 20. s4
  K=17408 106.0. mix 144.7. W8A8
  181. Napkin ~68. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=64 N=5120 K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48_k17408.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 62.099 pipe_host 62.540 vs
  square 20 vs s4 106.0 vs mix 144.7
  vs W8A8 181 vs napkin 68. ~3.13x
  square, near linear.

VERDICT -> Wide-K s2 4x8 is 62.5 us
  pipe_host at 2800 card1. Numeric
  closed. Beats s4 106.0, mix 144.7,
  and W8A8 181 (~2.89x). One-card.
  Do not freeze 62.5 us until card0.
  Rank pipe_host. Next: sibling
  N-wide vs sibling K-wide.

### 2026-09-03dp - K2 s2 4x8 A-db M=64 K=17408 sibling card0

CONTEXT -> card1 s2 4x8 K=17408 was
  62.5 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 0.
  M=64 N=5120 K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48_k17408.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 63.703 pipe_host 64.044 vs
  card1 62.540 vs square 20 vs s4
  106.0 vs mix 144.7 vs W8A8 181.
  Spread ~2.4%. ~3.20x square.

VERDICT -> Sibling matches. New s2
  4x8 M=64 K=17408 floor 64 us
  pipe_host both cards at 2800.
  Beats s4 106.0, mix 144.7, and
  W8A8 181 (~2.83x). Rank
  pipe_host.

### 2026-09-03dq - K2 s2 4x8 A-db M=64 N=17408 sibling card1

CONTEXT -> card0 s2 4x8 N=17408 was
  53.1 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=64 N=17408 K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48_wide.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800 throttle=0.
  event 52.646 pipe_host 52.859 vs
  card0 53.079 vs square 20 vs s4
  94.7 vs mix 129 vs W8A8 202.
  Spread ~0.41%. ~2.65x square.

VERDICT -> Sibling matches. New s2
  4x8 M=64 N=17408 floor 53.1 us
  pipe_host both cards at 2800.
  Beats s4 94.7, mix 129, and W8A8
  202 (~3.81x). Qwen FFN s2 M=64
  map is closed (20 / 53.1 / 64).
  Rank pipe_host. Next: s2 4x8 M=256
  vs s2xs8 4x8 M=64.

### 2026-09-03dr - K2 s2 4x8 A-db M=256 card0

CONTEXT -> s2 4x8 M=64 is 20. s4
  4-acc 48.6. mix 4x8 123 a loss.
  W8A8 75. Napkin 20*4 ~80. Same
  4x8 s2 tile. IGC s2 [-2,1].
  Never E2M1.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_db48 RC=8 wg 4x8 A-db
  pack=4. gpu-run --card 0. M=256
  N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48_m256.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 54.943
  pipe_host 55.453 vs M=64 20 vs
  s4 48.6 vs mix 123 vs W8A8 75
  vs napkin 80. ~2.77x M=64.

VERDICT -> s2 4x8 M=256 is 55.5 us
  pipe_host at 2800 card0. Numeric
  closed. Beats W8A8 75 (~1.35x)
  and mix 123. Loses to s4 4-acc
  48.6 (~1.14x). Napkin 80 beat.
  Rank pipe_host.

### 2026-09-03ds - K2 s2xs8 4x8 A-db M=64 card1

CONTEXT -> s2 4x8 20. s8xs4 43.3.
  s8 75. W8A8 46. decode s2xs8
  14.1. First s2xs8 on 4x8 A-db.
  A=s8 B=s2 pack=4 dpas K=32.
  IGC s2 [-2,1]. Never E2M1.
  New mix: both-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2xs8_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=64 N=K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2xs8_db48.sh 1 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> ocloc NT=2 64x dpas.8x8
  rW:s2 rA:b, grf 128, B d8v rd:2,
  no SLM. check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 34.094
  pipe_host 33.200 vs s2 20 vs
  s8xs4 43.3 vs s8 75 vs W8A8 46
  vs decode 14.1.

VERDICT -> s2xs8 4x8 M=64 is 33.2
  us pipe_host at 2800 card1.
  Numeric closed. Beats W8A8 46
  (~1.39x) and mix 43.3. Loses to
  s2 20 (~1.66x). Rank pipe_host.

### 2026-09-03dt - K2 s2 4x8 A-db M=256 sibling card1

CONTEXT -> card0 s2 4x8 M=256 was
  55.5 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48_m256.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 55.438
  pipe_host 55.497 vs card0 55.453
  vs s4 48.6 vs mix 123 vs W8A8 75.
  Spread ~0.08%. ~2.77x M=64.

VERDICT -> Sibling matches. New s2
  4x8 A-db M=256 floor 55.5 us
  pipe_host both cards at 2800.
  Beats W8A8 75 (~1.35x) and mix
  123. Loses to s4 4-acc 48.6
  (~1.14x). Not the M=256 hand
  floor. Rank pipe_host.

### 2026-09-03du - K2 s2xs8 4x8 A-db M=64 sibling card0

CONTEXT -> card1 s2xs8 4x8 M=64 was
  33.2 us at 2800, cosine=1
  max_abs=0. New mix sibling.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2xs8_db48. gpu-run --card 0.
  M=64 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2xs8_db48.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 33.182
  pipe_host 33.152 vs card1 33.200
  vs s2 20 vs s8xs4 43.3 vs s8 75
  vs W8A8 46. Spread ~0.14%.

VERDICT -> Sibling matches. New
  s2xs8 4x8 A-db M=64 floor 33.2
  us pipe_host both cards at 2800.
  Beats W8A8 46 (~1.39x) and mix
  43.3. Loses to s2 20 (~1.66x).
  Rank pipe_host. Next: s2 4x8
  M=256 N=17408 vs s2xs8 4x8 M=64
  N=17408.

### 2026-09-03dv - K2 s2 4x8 A-db M=256 N=17408 card0

CONTEXT -> s2 4x8 M=256 is 55.5.
  s4 N=17408 140. W8A8 248. Napkin
  55.5*17408/5120 ~189. Same 4x8
  s2 tile. IGC s2 [-2,1]. Never
  E2M1. throttle=1 risk.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_db48 RC=8 wg 4x8 A-db
  pack=4. gpu-run --card 0. M=256
  N=17408 K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48_m256_wide.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=2750 cur=2800
  throttle=1. event 169.401
  pipe_host 170.943 vs square 55.5
  vs s4 140 vs W8A8 248 vs napkin
  189. ~3.08x square.

VERDICT -> Wide-N s2 4x8 M=256 is
  171 us pipe_host at 2800 card0.
  Numeric closed. Beats W8A8 248
  (~1.45x). Loses to s4 140
  (~1.22x). throttle=1. Rank
  pipe_host.

### 2026-09-03dw - K2 s2xs8 4x8 A-db M=64 N=17408 card1

CONTEXT -> s2xs8 4x8 is 33.2. s2
  N=17408 53.1. mix 129. W8A8 202.
  Napkin 33.2*17408/5120 ~113.
  A=s8 B=s2. Never E2M1.

CONFIG -> backend sycl+l0, AOT
  dpas_s2xs8_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=64 N=17408
  K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2xs8_db48_wide.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 101.057
  pipe_host 100.528 vs square 33.2
  vs s2 53.1 vs mix 129 vs W8A8
  202 vs napkin 113. ~3.03x
  square.

VERDICT -> Wide-N s2xs8 4x8 is
  100.5 us pipe_host at 2800
  card1. Numeric closed. Beats
  W8A8 202 (~2.01x) and mix 129.
  Loses to s2 53.1 (~1.89x). Rank
  pipe_host.

### 2026-09-03dx - K2 s2 4x8 A-db M=256 N=17408 sibling card1

CONTEXT -> card0 s2 4x8 M=256
  N=17408 was 171 us, cosine=1
  max_abs=0, throttle=1. Sibling
  swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=256 N=17408 K=5120. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48_m256_wide.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=2767 cur=2800
  throttle=1. event 170.083
  pipe_host 170.446 vs card0
  170.943 vs square 55.5 vs s4
  140 vs W8A8 248. Spread ~0.29%.
  ~3.08x square.

VERDICT -> Sibling matches. New s2
  4x8 M=256 N=17408 floor 171 us
  pipe_host both cards at 2800.
  throttle=1 both. Beats W8A8 248
  (~1.45x). Loses to s4 140
  (~1.22x). Rank pipe_host.

### 2026-09-03dy - K2 s2xs8 4x8 A-db M=64 N=17408 sibling card0

CONTEXT -> card1 s2xs8 4x8 N=17408
  was 100.5 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2xs8_db48. gpu-run --card 0.
  M=64 N=17408 K=5120. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2xs8_db48_wide.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 100.016
  pipe_host 100.137 vs card1
  100.528 vs square 33.2 vs s2
  53.1 vs mix 129 vs W8A8 202.
  Spread ~0.39%. ~3.03x square.

VERDICT -> Sibling matches. New
  s2xs8 4x8 M=64 N=17408 floor
  100.5 us pipe_host both cards
  at 2800. Beats W8A8 202 (~2.01x)
  and mix 129. Loses to s2 53.1
  (~1.89x). Rank pipe_host. Next:
  s2 4x8 M=256 K=17408 vs s2xs8
  4x8 M=64 K=17408.

### 2026-09-03dz - K2 s2 4x8 A-db M=256 K=17408 card0

CONTEXT -> s2 4x8 M=256 is 55.5.
  N-wide 171 throttle=1. s4 149.
  W8A8 226. Napkin ~189. Same 4x8
  s2 tile. IGC s2 [-2,1]. Never
  E2M1.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_db48 RC=8 wg 4x8 A-db
  pack=4. gpu-run --card 0. M=256
  N=5120 K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_db48_m256_k17408.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 199.823
  pipe_host 201.019 vs square 55.5
  vs N-wide 171 vs s4 149 vs W8A8
  226 vs napkin 189. ~3.62x
  square.

VERDICT -> Wide-K s2 4x8 M=256 is
  201 us pipe_host at 2800 card0.
  Numeric closed. Beats W8A8 226
  (~1.12x). Loses to s4 149
  (~1.35x). Napkin 189 miss.
  throttle=0 unlike N-wide. Rank
  pipe_host.

### 2026-09-03ea - K2 s2xs8 4x8 A-db M=64 K=17408 card1

CONTEXT -> s2xs8 4x8 is 33.2.
  N-wide 100.5. s2 64. mix 144.7.
  W8A8 181. Napkin ~113. A=s8
  B=s2. Never E2M1.

CONFIG -> backend sycl+l0, AOT
  dpas_s2xs8_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 1. M=64 N=5120
  K=17408. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2xs8_db48_k17408.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 107.536
  pipe_host 107.385 vs square 33.2
  vs N-wide 100.5 vs s2 64 vs mix
  144.7 vs W8A8 181 vs napkin 113.
  ~3.23x square.

VERDICT -> Wide-K s2xs8 4x8 is 107
  us pipe_host at 2800 card1.
  Numeric closed. Beats W8A8 181
  (~1.69x) and mix 144.7. Loses to
  s2 64 (~1.67x). Rank pipe_host.

### 2026-09-03eb - K2 s2 4x8 A-db M=256 K=17408 sibling card1

CONTEXT -> card0 s2 4x8 M=256
  K=17408 was 201 us at 2800,
  cosine=1 max_abs=0, throttle=0.
  Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_db48. gpu-run --card 1.
  M=256 N=5120 K=17408. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_db48_m256_k17408.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 199.240
  pipe_host 199.070 vs card0
  201.019 vs square 55.5 vs
  N-wide 171 vs s4 149 vs W8A8
  226. Spread ~0.98%. ~3.62x
  square.

VERDICT -> Sibling matches. New s2
  4x8 M=256 K=17408 floor 201 us
  pipe_host both cards at 2800.
  throttle=0. Beats W8A8 226
  (~1.12x). Loses to s4 149
  (~1.35x). Qwen FFN s2 M=256 map
  is closed (55.5 / 171 / 201).
  Rank pipe_host.

### 2026-09-03ec - K2 s2xs8 4x8 A-db M=64 K=17408 sibling card0

CONTEXT -> card1 s2xs8 4x8 K=17408
  was 107 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2xs8_db48. gpu-run --card 0.
  M=64 N=5120 K=17408. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2xs8_db48_k17408.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 106.724
  pipe_host 106.722 vs card1
  107.385 vs square 33.2 vs
  N-wide 100.5 vs s2 64 vs mix
  144.7 vs W8A8 181. Spread
  ~0.62%. ~3.23x square.

VERDICT -> Sibling matches. New
  s2xs8 4x8 M=64 K=17408 floor
  107 us pipe_host both cards at
  2800. Beats W8A8 181 (~1.69x)
  and mix 144.7. Loses to s2 64
  (~1.67x). Qwen FFN s2xs8 M=64
  map is closed (33.2 / 100.5 /
  107). Rank pipe_host. Next:
  s2xs8 4x8 M=256 both-card.

### 2026-09-03ed - K2 s2xs8 4x8 A-db M=256 card0

CONTEXT -> s2xs8 4x8 M=64 is 33.2.
  s2 55.5. mix 4x8 123 a loss.
  W8A8 75. Napkin 33.2*4 ~133.
  First s2xs8 at M=256. A=s8 B=s2.
  Never E2M1. Both-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2xs8_db48 RC=8 wg 4x8 A-db.
  gpu-run --card 0. M=256 N=K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2xs8_db48_m256.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=2767 cur=2800
  throttle=1. event 95.333
  pipe_host 95.536 vs M=64 33.2 vs
  s2 55.5 vs mix 123 vs W8A8 75 vs
  napkin 133. ~2.88x M=64.

VERDICT -> s2xs8 4x8 M=256 is 95.5
  us pipe_host at 2800 card0.
  Numeric closed. Beats mix 123
  and napkin 133. Loses to s2 55.5
  (~1.72x) and W8A8 75 (~1.27x).
  throttle=1. Rank pipe_host.

### 2026-09-03ee - K2 s2xs8 4x8 A-db M=256 sibling card1

CONTEXT -> card0 s2xs8 4x8 M=256
  was 95.5 us, cosine=1 max_abs=0,
  throttle=1. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2xs8_db48. gpu-run --card 1.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2xs8_db48_m256.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=2767 cur=2800
  throttle=1. event 96.448
  pipe_host 95.735 vs card0 95.536
  vs s2 55.5 vs mix 123 vs W8A8 75.
  Spread ~0.21%. ~2.88x M=64.

VERDICT -> Sibling matches. s2xs8
  4x8 M=256 is 96 us pipe_host
  both cards at 2800. throttle=1.
  Beats mix 123. Loses to s2 55.5
  (~1.72x) and W8A8 75 (~1.27x).
  Stop 4x8 mix at M=256 prefill vs
  W8A8. Rank pipe_host. Next: s2
  4-acc M=256 both-card.

### 2026-09-03ef - K2 s2 4-acc M=256 card0

CONTEXT -> s4 4-acc is 48.6. s2
  4x8 55.5. W8A8 75. Napkin
  48.6*20/33.6 ~29. First s2 on
  4-acc. IGC s2 [-2,1]. Never
  E2M1. New geometry: both-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_w48m4 RC=8 4-acc wg 4x8
  k128 pack=4. gpu-run --card 0.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  compile_extra.sh dpas_s2_w48m4.cpp
  gpu-run --card 0 kernels/esimd_dpas/run_s2_w48m4.sh 0 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> COMPILE_OK. ocloc NT=2
  128x dpas.8x8 rW:s2 rA:s2, grf
  128, B d8v rd:4, no SLM. check
  cosine=1 max_abs=0. timed M=256
  act=cur=2800 throttle=0. event
  36.995 pipe_host 37.409 vs s4
  48.6 vs s2 4x8 55.5 vs W8A8 75
  vs napkin 29.

VERDICT -> s2 4-acc M=256 is 37.4
  us pipe_host at 2800 card0.
  Numeric closed. Beats s4 48.6
  (~1.30x), s2 4x8 55.5, and W8A8
  75 (~2.01x). Napkin 29 miss.
  New M=256 hand floor. Rank
  pipe_host.

### 2026-09-03eg - K2 s2 4-acc M=256 sibling card1

CONTEXT -> card0 s2 4-acc M=256
  was 37.4 us at 2800, cosine=1
  max_abs=0. New geometry sibling.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4. gpu-run --card 1.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_w48m4.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 37.755
  pipe_host 37.405 vs card0 37.409
  vs s4 48.6 vs s2 4x8 55.5 vs
  W8A8 75. Spread ~0.01%.

VERDICT -> Sibling matches. New s2
  4-acc M=256 floor 37.4 us
  pipe_host both cards at 2800.
  New M=256 hand floor. Beats s4
  48.6 (~1.30x) and W8A8 75
  (~2.01x). Rank pipe_host. Next:
  s2 4-acc M=256 N=17408 vs
  K=17408.

### 2026-09-03eh - K2 s2 4-acc M=256 N=17408 card0

CONTEXT -> s2 4-acc 37.4. s4 140.
  W8A8 248. Napkin 37.4*17408/5120
  ~127. Same 4-acc tile. IGC s2
  [-2,1]. Never E2M1.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_w48m4 RC=8 4-acc wg 4x8
  k128 pack=4. gpu-run --card 0.
  M=256 N=17408 K=5120. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_w48m4_wide.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 108.104
  pipe_host 108.604 vs square 37.4
  vs s4 140 vs W8A8 248 vs napkin
  127. ~2.90x square.

VERDICT -> Wide-N s2 4-acc is 109
  us pipe_host at 2800 card0.
  Numeric closed. Beats s4 140
  (~1.29x) and W8A8 248 (~2.28x).
  Napkin 127 beat. Rank pipe_host.

### 2026-09-03ei - K2 s2 4-acc M=256 K=17408 card1

CONTEXT -> s2 4-acc 37.4. s4 149.
  W8A8 226. Napkin ~127. Same
  4-acc tile. Never E2M1.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4. gpu-run --card 1.
  M=256 N=5120 K=17408. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_w48m4_k17408.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 108.620
  pipe_host 108.414 vs square 37.4
  vs s4 149 vs W8A8 226 vs napkin
  127. ~2.90x square.

VERDICT -> Wide-K s2 4-acc is 108
  us pipe_host at 2800 card1.
  Numeric closed. Beats s4 149
  (~1.37x) and W8A8 226 (~2.09x).
  Rank pipe_host.

### 2026-09-03ej - K2 s2 4-acc M=256 N=17408 sibling card1

CONTEXT -> card0 s2 4-acc N=17408
  was 109 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4. gpu-run --card 1.
  M=256 N=17408 K=5120. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_w48m4_wide.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 109.807
  pipe_host 109.947 vs card0
  108.604 vs square 37.4 vs s4
  140 vs W8A8 248. Spread ~1.24%.
  ~2.94x square.

VERDICT -> Sibling matches. New s2
  4-acc M=256 N=17408 floor 110 us
  pipe_host both cards at 2800.
  Beats s4 140 (~1.27x) and W8A8
  248 (~2.25x). Rank pipe_host.

### 2026-09-03ek - K2 s2 4-acc M=256 K=17408 sibling card0

CONTEXT -> card1 s2 4-acc K=17408
  was 108 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4. gpu-run --card 0.
  M=256 N=5120 K=17408. NT=2
  spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_w48m4_k17408.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 107.865
  pipe_host 108.216 vs card1
  108.414 vs square 37.4 vs s4
  149 vs W8A8 226. Spread ~0.18%.
  ~2.89x square.

VERDICT -> Sibling matches. New s2
  4-acc M=256 K=17408 floor 108 us
  pipe_host both cards at 2800.
  Beats s4 149 (~1.38x) and W8A8
  226 (~2.09x). Qwen FFN s2 4-acc
  M=256 map is closed (37.4 / 110
  / 108). Rank pipe_host. Next:
  s2 4-acc M=64 vs NT=4 M=256.

### 2026-09-03el - K2 s2 4-acc M=64 card0

CONTEXT -> s2 4x8 M=64 is 20. s2
  4-acc M=256 is 37.4. W8A8 46.
  4-acc tile pads M to 256-shaped
  grid. Occupancy check. One-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_w48m4 RC=8 4-acc wg 4x8
  k128 pack=4. gpu-run --card 0.
  M=64 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/esimd_dpas/run_s2_w48m4_m64.sh 0 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=64 act=cur=2800
  throttle=0. event 36.515
  pipe_host 37.152 vs 4x8 20 vs
  M=256 37.4 vs W8A8 46. ~1.86x
  4x8. Matches M=256 us.

VERDICT -> s2 4-acc M=64 is 37 us
  pipe_host at 2800 card0. Numeric
  closed. Occupancy pad: same us
  as M=256. Loses to 4x8 20
  (~1.86x). Stop 4-acc at M=64.
  One-card. Do not freeze. Rank
  pipe_host.

### 2026-09-03em - K2 s2 4-acc M=256 NT=4 card1

CONTEXT -> NT=2 4-acc is 37.4.
  NT=4 unroll=4 vs 8. Same tile.
  Schedule steal. One-card.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4. gpu-run --card 1.
  M=256 N=K=5120. NT=4 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_w48m4.sh 1 4 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 307.016
  pipe_host 307.201 vs NT=2 37.4
  vs s4 48.6 vs W8A8 75. ~8.21x
  NT=2.

VERDICT -> NT=4 is 307 us pipe_host
  at 2800 card1. Numeric closed.
  ~8.2x NT=2. Stop NT=4 on this
  tile. One-card. Do not freeze.
  Rank pipe_host. Next: s2 4-acc
  A-db M=256 both-card.

### 2026-09-03en - K2 s2 4-acc A-db M=256 card0

CONTEXT -> s2 4-acc no A-db is
  37.4. s4 A-db was 51.9 tax vs
  48.6. First s2 4-acc A-db. IGC
  s2 [-2,1]. Never E2M1. New
  geometry: both-card.

CONFIG -> backend sycl+l0, AOT
  dpas_s2_w48m4db RC=8 4-acc k64
  A-db wg 4x8 k128 pack=4.
  gpu-run --card 0. M=256 N=K=5120.
  NT=2 spin=512.

COMMAND ->
  ```
  compile_extra.sh dpas_s2_w48m4db.cpp
  gpu-run --card 0 kernels/esimd_dpas/run_s2_w48m4db.sh 0 2 512
  clang-offload-bundler --unbundle; ocloc disasm -device bmg-g31
  ```

RESULT -> COMPILE_OK. ocloc NT=2
  128x dpas.8x8 rW:s2 rA:s2, grf
  128, B d8v rd:4, no SLM. check
  cosine=1 max_abs=0. timed M=256
  act=cur=2800 throttle=0. event
  36.943 pipe_host 37.138 vs no-db
  37.4 vs W8A8 75.

VERDICT -> s2 4-acc A-db is 37.1 us
  pipe_host at 2800 card0. Numeric
  closed. Wash vs no-db 37.4 (not
  a tax, not a steal). Rank
  pipe_host.

### 2026-09-03eo - K2 s2 4-acc A-db M=256 sibling card1

CONTEXT -> card0 s2 4-acc A-db was
  37.1 us at 2800, cosine=1
  max_abs=0. Sibling swap.

CONFIG -> backend sycl+l0, same AOT
  dpas_s2_w48m4db. gpu-run --card 1.
  M=256 N=K=5120. NT=2 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/esimd_dpas/run_s2_w48m4db.sh 1 2 512
  ```

RESULT -> check cosine=1 max_abs=0.
  timed M=256 act=cur=2800
  throttle=0. event 37.667
  pipe_host 37.274 vs card0 37.138
  vs no-db 37.4 vs W8A8 75. Spread
  ~0.37%.

VERDICT -> Sibling matches. s2
  4-acc A-db is 37.2 us pipe_host
  both cards at 2800. Wash vs
  no-db 37.4. Stop A-db on s2
  4-acc. Floor stays 37.4 no-db.
  Rank pipe_host. Next: persist-s8
  GEMM us still no TU; s2 4-acc
  schedule steals closed.

### 2026-09-03ep - K7 GDN conv1d K=4 card0

CONTEXT -> Qwen3.8-27B is GDN
  hybrid. conv_k=4 key_dim=2048
  val_dim=6144. First GDN micro.
  No serve. s2 decode 11.5. W8A8
  44.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  depthwise conv1d K=4 bf16.
  T=1/64/256. spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_conv1d.sh 0
  ```

RESULT -> ok=1. T=1 q 125.283 k
  114.668 v 116.826. T=64/256
  stay 112-117. GB/s 0.2 at T=1.
  cur 550-2800 throttle=0.

VERDICT -> Conv1d is ~115 us
  launch-bound at 2800-class
  clocks card0. Not a fat GEMM.
  Rank us.

### 2026-09-03eq - K7 GDN delta recurrent card1

CONTEXT -> 48 v-heads, S 128x128
  bf16, 1.5 MiB/layer. Decode
  fused-recurrent. First delta
  micro. No serve.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  eager bmm delta update. spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_delta_recurrent.sh 1
  ```

RESULT -> ok=1. pipe-style host
  309.042 us. 10.3 GB/s. state
  72 MiB all 48 layers. cur
  550-2800 throttle=0.

VERDICT -> Eager delta is 309 us
  card1, ~7x W8A8 44. State bytes
  are small; this path is the
  leftover. Rank us.

### 2026-09-03er - K7 GDN conv1d sibling card1

CONTEXT -> card0 conv1d was ~115
  us launch-bound. Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_conv1d.sh 1
  ```

RESULT -> ok=1. T=1 q 117.114 k
  115.776 v 119.369. T=256 tracks
  ~115. vs card0 125/115/117.
  q T=1 spread ~7% (clocks).

VERDICT -> Sibling matches the
  115 us class. Launch-bound both
  cards. Rank us.

### 2026-09-03es - K7 GDN delta sibling card0

CONTEXT -> card1 delta was 309 us.
  Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_delta_recurrent.sh 0
  ```

RESULT -> ok=1. 306.575 us vs
  card1 309.042. Spread ~0.8%.
  10.4 GB/s.

VERDICT -> Sibling matches. Eager
  GDN delta is 308 us both cards.
  Conv1d 115 us launch-bound.
  Naive GDN >> s2 decode 11.5 and
  W8A8 44. Rank us. Next: qkvz
  GEMM 5120x2048 vs fused GDN
  kernel.

### 2026-09-03et - K7 GDN q-proj W8A8 card0

CONTEXT -> q-proj 5120x2048. W8A8
  square M=1 is 44. conv1d 115.
  delta 308. s2 decode 11.5.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  int8_gemm_w8a8 M=1 n=2048 k=5120
  heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_q_w8a8.sh 0
  ```

RESULT -> cosine=1 max_abs=0.031
  ok=1. 45.344 us. 231 GB/s. vs
  square 44. Not N-linear.

VERDICT -> q-proj W8A8 is 45 us
  card0, same class as 5120
  square. Under conv 115 and
  delta 308. Rank us.

### 2026-09-03eu - K7 GDN v-proj W8A8 card1

CONTEXT -> v-proj 5120x6144. 3x
  q N. Napkin 3x 44 ~132.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  int8_gemm_w8a8 M=1 n=6144 k=5120
  heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_v_w8a8.sh 1
  ```

RESULT -> cosine=1 max_abs=0.059
  ok=1. 46.306 us. 679 GB/s. vs
  q 45 vs napkin 132.

VERDICT -> v-proj W8A8 is 46 us
  card1, not 3x q. Launch class
  like square 44. Rank us.

### 2026-09-03ev - K7 GDN v-proj sibling card0

CONTEXT -> card1 v-proj was 46 us.
  Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_v_w8a8.sh 0
  ```

RESULT -> cosine=1 max_abs=0.043
  ok=1. 46.080 us vs card1 46.306.
  Spread ~0.49%.

VERDICT -> Sibling matches. GDN
  v-proj W8A8 is 46 us both cards.
  Rank us.

### 2026-09-03ew - K7 GDN q-proj sibling card1

CONTEXT -> card0 q-proj was 45 us.
  Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_q_w8a8.sh 1
  ```

RESULT -> cosine=1 max_abs=0.061
  ok=1. 58.429 us vs card0 45.344.
  Spread ~29%. cur 550-2800.

VERDICT -> q-proj clocks disagreed.
  45-58 us, still the 44-class
  launch, under conv 115 and
  delta 308. Do not freeze 45.
  Rank us. Next: fused ESIMD
  conv1d vs fused delta.

### 2026-09-03ex - K7 ESIMD GDN conv1d card0

CONTEXT -> Eager conv1d K=4 is
  ~115 us launch-bound. First
  fused ESIMD TU. C=2048 q/k and
  C=6144 v. T=1 with K-1 state.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_conv1d
  intel_gpu_bmg_g31. gpu-run
  --card 0. f16, VL=16 wg=16.
  NT n/a. spin=4000. Named clock
  not 2800 (short kernel).

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d.sh 0
  ```

RESULT -> cosine=1 max_abs=0
  cosine_st=1 ok=1. C=2048
  event 1.456 pipe_host 4.350
  act=cur=1700 throttle=0.
  C=6144 event 1.083 pipe_host
  4.799 act=cur=2250. vs eager
  115. GB/s 11 / 31.

VERDICT -> ESIMD conv1d is 4.35
  us pipe_host card0 at 1700,
  ~26x eager 115. Numeric closed.
  Clocks not 2800. Do not freeze
  4.35. Rank pipe_host.

### 2026-09-03ey - K7 ESIMD GDN delta card1

CONTEXT -> Eager delta is 308 us.
  First fused ESIMD recurrent TU.
  48 heads, S 128x128 f16.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta
  intel_gpu_bmg_g31. gpu-run
  --card 1. VL=16 wg=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta.sh 1
  ```

RESULT -> cosine=1 max_abs=0.015625
  (1 ulp f16) cosine_o=1
  max_abs_o=0 ok=1. event 8.432
  pipe_host 7.093. 450 GB/s vs
  copy 550. act=cur=2800
  throttle=0. vs eager 308.

VERDICT -> ESIMD delta is 7.09 us
  pipe_host card1 at 2800, ~43x
  eager 308. Near HBM. Numeric
  closed. One-card. Do not freeze
  7.09 until sibling. Rank
  pipe_host. Next: sibling swap.

### 2026-09-03ez - K7 ESIMD GDN delta sibling card0

CONTEXT -> card1 delta was 7.09
  us pipe_host at 2800, cosine=1
  max_abs=0.015625. Sibling swap.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta. gpu-run --card 0.
  nv=48 dv=128 dk=128. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta.sh 0
  ```

RESULT -> cosine=1 max_abs=0.015625
  cosine_o=1 max_abs_o=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 7.825 pipe_host 7.028
  vs card1 7.093. 455 GB/s.
  Spread ~0.9%.

VERDICT -> Sibling matches. ESIMD
  delta is 7.1 us pipe_host both
  cards at 2800. ~43x eager 308.
  Rank pipe_host.

### 2026-09-03fa - K7 ESIMD GDN conv1d sibling card1

CONTEXT -> card0 conv1d was 4.35
  us pipe_host at 1700. Sibling.
  Clocks not 2800 last time.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d. gpu-run --card 1.
  T=1 k=4 f16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  C=2048 event 1.799 pipe_host
  4.500 act=cur=1400. C=6144
  pipe_host 5.000 at 2167. vs
  card0 4.350 at 1700. Spread
  ~3.4%. vs eager 115.

VERDICT -> Sibling matches the
  4.4 us class both cards. Short
  kernel, clocks 1400-1700 not
  2800. Do not freeze 4.4 as a
  2800 floor. Mixer 4.4+7.1 ~11.5
  us under W8A8 46. Rank
  pipe_host. Next: o-proj vs
  fused qkv conv.

### 2026-09-03fb - K7 GDN o-proj W8A8 card0

CONTEXT -> o-proj value_dim->H
  6144x5120. v-proj 46. q-proj
  45-58. Mixer ESIMD 4.4+7.1.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  int8_gemm_w8a8 M=1 n=5120
  k=6144. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_o_w8a8.sh 0
  ```

RESULT -> cosine=1 max_abs=0.060
  ok=1. 46.293 us. 680 GB/s. vs
  v-proj 46 vs square 44. cur
  2017-2800 end 2717 throttle=0.

VERDICT -> o-proj W8A8 is 46 us
  card0, same class as v-proj
  and square. Not K-linear.
  One-card. Rank us.

### 2026-09-03fc - K7 ESIMD fused qkv conv1d card1

CONTEXT -> q+k+v = C=10240.
  One-arm ~4.4. 3x napkin ~13.2.
  First fuse of conv family.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_conv1d_qkv.
  gpu-run --card 1. f16 VL=16
  wg=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_qkv.sh 1
  ```

RESULT -> fused cosine=1
  max_abs=0 ok=1. C=10240 event
  0.917 pipe_host 4.437. 55 GB/s.
  act=cur=2800 throttle=0. trio
  event 2.755 pipe_host 13.449.

VERDICT -> Fused qkv conv is
  4.44 us pipe_host card1 at
  2800, same class as one-arm
  4.4, ~3.03x trio 13.4. Launch
  bound. One-card. Do not freeze
  4.44 until sibling. Rank
  pipe_host. Next: sibling swap.

### 2026-09-03fd - K7 ESIMD fused qkv conv sibling card0

CONTEXT -> card1 fused was 4.44
  us pipe_host at 2800 vs trio
  13.4. First fuse. Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_qkv. gpu-run
  --card 0. C=10240. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_qkv.sh 0
  ```

RESULT -> fused cosine=1
  max_abs=0 ok=1. pipe_host 4.856
  event 2.138 act=cur=1183. vs
  card1 4.437 at 2800. Spread
  ~9%. trio pipe_host 14.233 at
  2800 vs card1 13.449.

VERDICT -> Sibling matches the
  4.4-4.9 us class. Fused clocks
  1183 vs 2800, us spread >5%.
  Do not freeze 4.44 as 2800.
  Trio ~13.8 at 2800 both. Rank
  pipe_host.

### 2026-09-03fe - K7 GDN o-proj W8A8 sibling card1

CONTEXT -> card0 o-proj was 46
  us. Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  int8_gemm_w8a8 M=1 n=5120
  k=6144. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_o_w8a8.sh 1
  ```

RESULT -> cosine=1 max_abs=0.056
  ok=1. 47.133 us vs card0
  46.293. Spread ~1.8%.

VERDICT -> Sibling matches. GDN
  o-proj W8A8 is 46-47 us both
  cards, same class as v-proj 46.
  Rank us. Next: packed qkv
  W8A8 vs fuse conv+delta.

### 2026-09-03ff - K7 GDN packed qkv W8A8 card0

CONTEXT -> q+k+v N=10240 K=5120
  one GEMM vs 3x 46 ~138. v-proj
  n=6144 is 46.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  int8_gemm_w8a8 M=1 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_qkv_w8a8.sh 0
  ```

RESULT -> cosine=1 max_abs=0.043
  ok=1. 95.783 us. 547 GB/s. vs
  3x 46 ~138 vs v-proj 46. cur
  end 2400.

VERDICT -> Packed qkv W8A8 is 96
  us card0, ~1.44x 3 sequential
  138, ~2.08x v-proj 46. Not
  launch-class 46. One-card. Do
  not freeze 96. Rank us.

### 2026-09-03fg - K7 ESIMD mixer conv+delta card1

CONTEXT -> conv 4.4 + delta 7.1
  = 11.5. First mixer fuse.
  Packed C=10240 then 48-head
  delta, q/k repeat 16->48.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer.
  gpu-run --card 1. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer.sh 1
  ```

RESULT -> cosine=1 max_abs=
  0.000122 cosine_o=1 max_abs_o=0
  ok=1. event 9.758 pipe_host
  8.229. act=cur=2800 throttle=0.
  vs 11.5.

VERDICT -> Mixer is 8.23 us
  pipe_host card1 at 2800, ~1.40x
  the 11.5 sum. Conv hides under
  delta. One-card. Do not freeze
  8.23 until sibling. Rank
  pipe_host. Next: sibling swap.

### 2026-09-03fh - K7 ESIMD mixer sibling card0

CONTEXT -> card1 mixer was 8.23
  us pipe_host at 2800. First
  mixer fuse. Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer. gpu-run --card 0.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer.sh 0
  ```

RESULT -> cosine=1 max_abs=
  0.000122 cosine_o=1 ok=1.
  timed act=cur=2800 throttle=0.
  event 8.898 pipe_host 8.746 vs
  card1 8.229. Spread ~6.3%.

VERDICT -> Sibling matches the
  8.2-8.7 us class both cards at
  2800. Spread >5%. Do not freeze
  8.23. Conv still hides under
  delta vs 11.5. Rank pipe_host.

### 2026-09-03fi - K7 packed qkv W8A8 sibling card1

CONTEXT -> card0 packed qkv was
  96 us. Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  int8_gemm_w8a8 M=1 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_qkv_w8a8.sh 1
  ```

RESULT -> cosine=1 max_abs=0.055
  ok=1. 95.481 us vs card0
  95.783. Spread ~0.3%.

VERDICT -> Sibling matches. Packed
  qkv W8A8 is 96 us both cards,
  ~1.44x 3 sequential 138. Rank
  us. Next: packed qkv M=64 vs
  conv T=64.

### 2026-09-03fj - K7 packed qkv W8A8 M=64 card0

CONTEXT -> M=1 packed qkv is 96
  us. W8A8 M=64 square 46. 3x
  napkin ~138.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  int8_gemm_w8a8 M=64 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_qkv_w8a8_m64.sh 0
  ```

RESULT -> cosine=1 max_abs=0.062
  ok=1. 142.053 us. 369 GB/s. vs
  M=1 96 vs square M=64 46 vs 3x
  46 ~138.

VERDICT -> Packed qkv M=64 is 142
  us card0, ~1.48x M=1 96, wash
  vs 3x 46. One-card. Do not
  freeze 142. Rank us.

### 2026-09-03fk - K7 ESIMD conv1d T=64 card1

CONTEXT -> eager T=64 ~115.
  decode T=1 4.4. First T>1 conv.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_conv1d_t.
  gpu-run --card 1. C=2048 T=64
  k=4 f16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t64.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  event 9.771 pipe_host 10.161.
  53 GB/s. act=cur=2800
  throttle=0. vs eager 115 vs
  decode 4.4.

VERDICT -> ESIMD conv T=64 is
  10.2 us pipe_host card1 at
  2800, ~11x eager 115, ~2.3x
  decode T=1. Not T-linear.
  One-card. Do not freeze 10.2
  until sibling. Rank pipe_host.
  Next: sibling swap.

### 2026-09-03fl - K7 ESIMD conv1d T=64 sibling card0

CONTEXT -> card1 conv T=64 was
  10.2 us pipe_host at 2800.
  First T>1. Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=2048 T=64. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t64.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 9.768 pipe_host 10.130 vs
  card1 10.161. Spread ~0.3%.

VERDICT -> Sibling matches. ESIMD
  conv T=64 is 10.1 us pipe_host
  both cards at 2800. ~11x eager
  115. Rank pipe_host.

### 2026-09-03fm - K7 packed qkv W8A8 M=64 sibling card1

CONTEXT -> card0 packed qkv M=64
  was 142 us. Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  int8_gemm_w8a8 M=64 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_qkv_w8a8_m64.sh 1
  ```

RESULT -> cosine=1 max_abs=0.062
  ok=1. 138.079 us vs card0
  142.053. Spread ~2.9%.

VERDICT -> Sibling matches. Packed
  qkv M=64 is 138-142 us both
  cards, wash vs 3x 46. Rank us.
  Next: conv T=256 vs packed qkv
  M=256.

### 2026-09-03fn - K7 ESIMD conv1d T=256 card0

CONTEXT -> eager T=256 ~115.
  T=64 10.1. decode T=1 4.4.
  Napkin 4x 10.1 ~40.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=2048 T=256 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t256.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 37.253 pipe_host 37.607.
  56 GB/s. vs T=64 10.1 vs eager
  115.

VERDICT -> ESIMD conv T=256 is
  37.6 us pipe_host card0 at
  2800, ~3.72x T=64, ~3.1x eager
  115. Near T-linear. One-card.
  Do not freeze 37.6 until
  sibling. Rank pipe_host.

### 2026-09-03fo - K7 packed qkv W8A8 M=256 card1

CONTEXT -> M=1 96. M=64 140.
  W8A8 M=256 square 75. 3x 75
  ~225.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 1.
  int8_gemm_w8a8 M=256 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_proj_qkv_w8a8_m256.sh 1
  ```

RESULT -> cosine=1 max_abs=0.062
  ok=1. 163.539 us. 321 GB/s. vs
  M=64 140 vs square 75 vs 3x 75
  ~225.

VERDICT -> Packed qkv M=256 is
  164 us card1, ~1.17x M=64, ~1.37x
  3 sequential 225. One-card. Do
  not freeze 164. Rank us. Next:
  sibling swap.

### 2026-09-03fp - K7 packed qkv W8A8 M=256 sibling card0

CONTEXT -> card1 packed qkv M=256
  was 164 us. Sibling.

CONFIG -> backend pytorch-xpu on
  sycl+l0. gpu-run --card 0.
  int8_gemm_w8a8 M=256 n=10240
  k=5120. heat M=64 spin=512.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_proj_qkv_w8a8_m256.sh 0
  ```

RESULT -> cosine=1 max_abs=0.063
  ok=1. 163.739 us vs card1
  163.539. Spread ~0.12%.

VERDICT -> Sibling matches. Packed
  qkv M=256 is 164 us both cards,
  ~1.17x M=64, ~1.37x 3x 75. Rank
  us.

### 2026-09-03fq - K7 ESIMD conv1d T=256 sibling card1

CONTEXT -> card0 conv T=256 was
  37.6 us pipe_host at 2800.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. C=2048 T=256.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t256.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 37.229 pipe_host 37.811 vs
  card0 37.607. Spread ~0.5%.

VERDICT -> Sibling matches. ESIMD
  conv T=256 is 37.7 us pipe_host
  both cards at 2800. ~3.72x T=64.
  Rank pipe_host. Next: conv
  C=6144 T=256 vs delta T=64.

### 2026-09-03fr - K7 ESIMD conv1d T=256 C=6144 card0

CONTEXT -> C=2048 T=256 is 37.7.
  v-channels C=6144. Napkin 3x
  ~113. Eager v T=256 ~115.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=6144 T=256 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t256_c6144.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 37.630 pipe_host 38.010.
  167 GB/s. vs C=2048 37.7 vs
  eager 115.

VERDICT -> ESIMD conv T=256 C=6144
  is 38.0 us pipe_host card0 at
  2800, wash vs C=2048 37.7 not
  3x. Occupancy. One-card. Do not
  freeze 38.0 until sibling. Rank
  pipe_host.

### 2026-09-03fs - K7 ESIMD delta T=64 card1

CONTEXT -> decode T=1 7.1. Napkin
  64x ~454. First T>1 delta. New
  TU gdn_delta_t. Unnormalized
  /50 fill NaN'd; L2-norm q/k.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_t.
  gpu-run --card 1. T=64 nv=48
  dv=128 dk=128 f16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_t64.sh 1
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 266.430 pipe_host
  264.906. 24 GB/s. act=2633-2650
  cur=2800 throttle=1. vs decode
  7.1 vs napkin 454.

VERDICT -> ESIMD delta T=64 is
  265 us pipe_host card1, ~37x
  T=1 not 64x. throttle=1. Do not
  freeze 265 as 2800. Rank
  pipe_host. Next: sibling
  (throttle=1).

### 2026-09-03ft - K7 ESIMD delta T=64 sibling card0

CONTEXT -> card1 delta T=64 was
  265 us pipe_host throttle=1.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_t. gpu-run
  --card 0. T=64. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_t64.sh 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 271.950 pipe_host
  271.249 vs card1 264.906.
  Spread ~2.4%. act=2583 cur=2800
  throttle=1.

VERDICT -> Sibling matches. ESIMD
  delta T=64 is 265-271 us
  pipe_host both cards.
  throttle=1 both. Do not freeze
  265 as 2800. ~37x decode 7.1
  not 64x. Rank pipe_host. Next:
  conv C=6144 sibling vs delta
  T=256.

### 2026-09-03fu - K7 ESIMD conv1d T=256 C=6144 sibling card1

CONTEXT -> card0 conv T=256 C=6144
  was 38.0 us pipe_host at 2800.
  Occupancy wash vs C=2048.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. C=6144 T=256.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t256_c6144.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 37.747 pipe_host 38.032 vs
  card0 38.010. Spread ~0.06%.

VERDICT -> Sibling matches. ESIMD
  conv T=256 C=6144 is 38.0 us
  pipe_host both cards at 2800.
  Wash vs C=2048 37.7 not 3x.
  Rank pipe_host.

### 2026-09-03fv - K7 ESIMD delta T=256 card0

CONTEXT -> T=64 265-271 throttle=1.
  Napkin 4x ~1060. decode T=1 7.1.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_t. gpu-run
  --card 0. T=256 nv=48 dv=128
  dk=128 f16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_t256.sh 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 1112.104 pipe_host
  1109.372. 14 GB/s. act=2617
  cur=2800 throttle=1. vs T=64
  271 vs napkin 1060.

VERDICT -> ESIMD delta T=256 is
  1109 us pipe_host card0, ~4.1x
  T=64 near T-linear. throttle=1.
  Prefill leftover. Do not freeze
  1109 as 2800. Rank pipe_host.
  Next: sibling (throttle=1).

### 2026-09-03fw - K7 ESIMD delta T=256 sibling card1

CONTEXT -> card0 delta T=256 was
  1109 us pipe_host throttle=1.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_t. gpu-run
  --card 1. T=256. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_t256.sh 1
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 1101.310 pipe_host
  1099.419 vs card0 1109.372.
  Spread ~0.9%. act=2650 cur=2800
  throttle=1.

VERDICT -> Sibling matches. ESIMD
  delta T=256 is 1100-1109 us
  pipe_host both cards.
  throttle=1 both. Do not freeze
  1100 as 2800. ~4.1x T=64. Rank
  pipe_host. Next: mixer T=64 vs
  conv T=64 C=6144.

### 2026-09-03fx - K7 ESIMD conv1d T=64 C=6144 card1

CONTEXT -> C=2048 T=64 is 10.1.
  C=6144 T=256 is 38.0 occupancy
  wash. Napkin 3x ~30 if C-linear.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. C=6144 T=64 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t64_c6144.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 9.872 pipe_host 10.231.
  159 GB/s. vs C=2048 10.1.

VERDICT -> ESIMD conv T=64 C=6144
  is 10.2 us pipe_host card1 at
  2800, wash vs C=2048 10.1 not
  3x. Occupancy. One-card. Do not
  freeze 10.2 until sibling. Rank
  pipe_host.

### 2026-09-03fy - K7 ESIMD mixer T=64 card0

CONTEXT -> decode mixer 8.2.
  conv T=64 10.1. delta T=64 265.
  Napkin sequential ~275. New TU
  gdn_mixer_t. L2-norm q/k.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_t.
  gpu-run --card 0. T=64 C=10240
  nv=48. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_t64.sh 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 400.487 pipe_host
  399.311. act=2633-2650 cur=2800
  throttle=1. vs napkin 275 vs
  delta 265.

VERDICT -> ESIMD mixer T=64 is
  399 us pipe_host card0, ~1.45x
  sequential 275. throttle=1. Do
  not freeze 399 as 2800. Rank
  pipe_host. Next: sibling
  (throttle=1).

### 2026-09-03fz - K7 ESIMD mixer T=64 sibling card1

CONTEXT -> card0 mixer T=64 was
  399 us pipe_host throttle=1.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_t. gpu-run
  --card 1. T=64. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_t64.sh 1
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 395.102 pipe_host
  394.542 vs card0 399.311.
  Spread ~1.2%. act=2667 cur=2800
  throttle=1.

VERDICT -> Sibling matches. ESIMD
  mixer T=64 is 395-399 us
  pipe_host both cards.
  throttle=1 both. Loses to
  sequential ~275. Stop two-kernel
  packed mixer at prefill. Do not
  freeze 395 as 2800. Rank
  pipe_host. Next: conv T=64
  C=6144 sibling vs conv T=64
  C=10240.

### 2026-09-03ga - K7 ESIMD conv1d T=64 C=6144 sibling card0

CONTEXT -> card1 conv T=64 C=6144
  was 10.2 us pipe_host at 2800.
  Wash vs C=2048 10.1. Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=6144 T=64 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t64_c6144.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 9.880 pipe_host 10.449 vs
  card1 10.231. Spread ~2.1%.
  155 GB/s.

VERDICT -> Sibling matches. ESIMD
  conv T=64 C=6144 is 10.2-10.4 us
  pipe_host both cards at 2800.
  Wash vs C=2048 10.1 not 3x.
  Occupancy. Rank pipe_host.

### 2026-09-03gb - K7 ESIMD conv1d T=64 C=10240 card1

CONTEXT -> C=2048 T=64 is 10.1.
  C=6144 T=64 is 10.2-10.4.
  Packed qkv width. Napkin 5x
  ~50 if C-linear.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. C=10240 T=64 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t64_c10240.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 10.107 pipe_host 10.535.
  257 GB/s. vs C=2048 10.1 vs
  C=6144 10.2-10.4.

VERDICT -> ESIMD conv T=64 C=10240
  is 10.5 us pipe_host card1 at
  2800, wash vs C=2048 10.1 not
  5x. Occupancy. One-card. Do not
  freeze 10.5 until sibling. Rank
  pipe_host. Next: sibling vs
  conv T=256 C=10240.

### 2026-09-03gc - K7 ESIMD conv1d T=64 C=10240 sibling card0

CONTEXT -> card1 conv T=64 C=10240
  was 10.5 us pipe_host at 2800.
  Wash vs C=2048 10.1. Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=10240 T=64 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t64_c10240.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 10.128 pipe_host 10.667 vs
  card1 10.535. Spread ~1.3%.
  253 GB/s.

VERDICT -> Sibling matches. ESIMD
  conv T=64 C=10240 is 10.5-10.7 us
  pipe_host both cards at 2800.
  Wash vs C=2048 10.1 not 5x.
  Occupancy. Rank pipe_host.

### 2026-09-03gd - K7 ESIMD conv1d T=256 C=10240 card1

CONTEXT -> C=2048 T=256 is 37.7.
  C=6144 T=256 is 38.0. Packed
  qkv width. Napkin 5x ~189 if
  C-linear. T=64 C=10240 10.5.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. C=10240 T=256 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t256_c10240.sh 1
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 40.393 pipe_host 40.742.
  259 GB/s. vs C=6144 38.0
  (~1.07x) vs napkin 189.

VERDICT -> ESIMD conv T=256
  C=10240 is 40.7 us pipe_host
  card1 at 2800, ~1.07x C=6144
  38.0 not 5x. Small packed-width
  tax, occupancy still. One-card.
  Do not freeze 40.7 until
  sibling. Rank pipe_host. Next:
  sibling vs chunk/WY delta.

### 2026-09-03ge - K7 ESIMD conv1d T=256 C=10240 sibling card0

CONTEXT -> card1 conv T=256
  C=10240 was 40.7 us pipe_host
  at 2800. ~1.07x C=6144 38.0.
  Sibling.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. C=10240 T=256 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t256_c10240.sh 0
  ```

RESULT -> cosine=1 max_abs=0 ok=1.
  timed act=cur=2800 throttle=0.
  event 40.344 pipe_host 40.797 vs
  card1 40.742. Spread ~0.13%.
  259 GB/s.

VERDICT -> Sibling matches. ESIMD
  conv T=256 C=10240 is 40.7-40.8
  us pipe_host both cards at 2800.
  ~1.07x C=6144 38.0 not 5x.
  Occupancy. Rank pipe_host.

### 2026-09-03gf - K7 ESIMD chunk/WY delta T=256 C=16 card1

CONTEXT -> fused delta T=256 is
  1100-1109 throttle=1. FLA
  chunk/WY C=16. Napkin beats
  serial 1100. New TU
  gdn_delta_chunk.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_chunk.
  gpu-run --card 1. T=256 C=16
  nv=48 dv=128 dk=128 f16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_chunk_t256.sh 1
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 3199.857 pipe_host
  3210.272. 4.91 GB/s. timed
  act=cur=2800 throttle=0. vs
  fused 1100 (~2.92x).

VERDICT -> ESIMD chunk/WY C=16
  T=256 is 3210 us pipe_host
  card1 at 2800, numeric closed,
  ~2.92x fused 1100. Napkin
  miss. Stop C=16 vs fused.
  One-card. Do not freeze 3210
  as a floor. Rank pipe_host.
  Next: chunk C=64 vs fused.

### 2026-09-03gg - K7 ESIMD fused delta T=256 hold retry card1

CONTEXT -> fv/fw fused T=256 was
  1100-1109 us pipe_host,
  throttle=1 act=2617-2650.
  Held-clock retry. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_t. gpu-run
  --card 1. T=256 nv=48. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_t256_hold.sh 1
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 1088.060 pipe_host
  1085.686. 14.5 GB/s. spin_done
  act=2683 cur=2800 throttle=1.
  timed act=2683 throttle=1. vs
  fw 1099.

VERDICT -> Hold retry still
  throttle=1. ESIMD fused delta
  T=256 is 1086 us pipe_host
  card1, ~1.3% under fw 1099.
  Cannot hold 2800. Do not freeze
  1086 as 2800. Prefill leftover.
  Rank pipe_host.

### 2026-09-03gh - K7 ESIMD chunk/WY delta T=256 C=64 card0

CONTEXT -> C=16 3210 us. fused
  1086-1109 throttle=1. FLA
  default C=64. Napkin beats
  C=16 and fused. New TU
  gdn_delta_chunk64. spin=0:
  C=64 ~95 ms, spin=4000 would
  serialize past 5m.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_chunk64.
  gpu-run --card 0. T=256 C=64
  nv=48. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_chunk64_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 95413.974 pipe_host
  95419.883. 0.17 GB/s. timed
  act=cur=2800 throttle=0. vs
  C=16 3210 (~29.7x) vs fused
  1086 (~87.9x).

VERDICT -> ESIMD chunk/WY C=64
  T=256 is 95420 us pipe_host
  card0 at 2800, numeric closed,
  ~88x fused 1086, ~30x C=16.
  Napkin miss. Stop C=64 vs
  fused. Stop this WY path.
  One-card. Rank pipe_host.
  Next: fused T=256 SLM-K.

### 2026-09-03gi - K7 ESIMD fused delta T=256 SLM-K card0

CONTEXT -> fused T=256 1086 us
  throttle=1. WY lost. Napkin
  SLM block-16 k/q beats HBM
  reload. Serial S. spin=0
  until us known.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmk.
  gpu-run --card 0. T=256 blk=16
  nv=48. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 840.529 pipe_host
  847.280. 18.6 GB/s. timed_begin
  act=2783 throttle=0. timed_end
  act=2767 throttle=1. vs fused
  1086 (~1.28x).

VERDICT -> ESIMD SLM-K T=256 is
  847 us pipe_host card0, ~1.28x
  fused 1086. Numeric closed.
  throttle=1 at end. Do not freeze
  847 as 2800. Possible leftover
  cut. One-card. Sibling before
  promote. Rank pipe_host.

### 2026-09-03gj - K7 ESIMD fused delta T=256 row-block card1

CONTEXT -> fused 1086. SLM-K 847.
  Napkin 4-row k-reuse beats
  1-row occupancy. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_rowb.
  gpu-run --card 1. T=256 rb=4
  nv=48. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_rowb_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 1034.237 pipe_host
  1034.092. 15.3 GB/s. timed
  act=cur=2800 throttle=0. vs
  fused 1086 (~1.05x) vs SLM-K
  847.

VERDICT -> ESIMD row-block rb=4
  T=256 is 1034 us pipe_host
  card1 at 2800, ~1.05x fused
  1086, loses to SLM-K 847.
  Numeric closed. One-card. Rank
  pipe_host. Next: sibling SLM-K
  vs rb=8.

### 2026-09-03gk - K7 ESIMD fused delta T=256 SLM-K sibling card1

CONTEXT -> card0 SLM-K was 847 us
  pipe_host, throttle=1. Possible
  leftover cut. Sibling. spin=4000
  hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk. gpu-run
  --card 1. T=256 blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmk_t256.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 859.891 pipe_host
  858.215 vs card0 847.280.
  Spread ~1.3%. 18.4 GB/s. timed
  act=2700-2717 cur=2800
  throttle=1. vs fused 1086.

VERDICT -> Sibling matches. ESIMD
  SLM-K T=256 is 847-858 us
  pipe_host both cards, ~1.27x
  fused 1086. throttle=1 both.
  Do not freeze 847 as 2800. New
  leftover class. Rank pipe_host.

### 2026-09-03gl - K7 ESIMD fused delta T=256 row-block rb=8 card0

CONTEXT -> rb=4 1034. SLM-K 847.
  Napkin 8-row k-reuse vs
  occupancy. One WG per head.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_rowb8.
  gpu-run --card 0. T=256 rb=8
  nv=48. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_rowb8_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 2058.990 pipe_host
  2060.439. 7.7 GB/s. timed
  act=cur=2800 throttle=0. vs
  rb=4 1034 (~1.99x) vs SLM-K
  847.

VERDICT -> ESIMD row-block rb=8
  T=256 is 2060 us pipe_host
  card0 at 2800, ~2x rb=4, ~2.4x
  SLM-K. Occupancy loss. Stop
  rb=8 vs rb=4 and SLM-K. Rank
  pipe_host. Next: SLM-K+rb=4 vs
  SLM-K blk=32.

### 2026-09-03gm - K7 ESIMD fused delta T=256 SLM-K+rb=4 card0

CONTEXT -> SLM-K 847. rb=4 1034.
  Napkin combine beats 847.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmk_rb4.
  gpu-run --card 0. T=256 blk=16
  rb=4. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk_rb4_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 998.307 pipe_host
  998.817. 15.8 GB/s. timed
  act=cur=2800 throttle=0. vs
  SLM-K 847 (~1.18x) vs rb=4
  1034.

VERDICT -> ESIMD SLM-K+rb=4 T=256
  is 999 us pipe_host card0 at
  2800, loses to SLM-K 847,
  slight beat of rb=4 1034.
  Occupancy. Stop combine vs
  SLM-K. Rank pipe_host.

### 2026-09-03gn - K7 ESIMD fused delta T=256 SLM-K blk=32 card1

CONTEXT -> SLM-K blk=16 847-858.
  Napkin blk=32 fewer barriers.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmk32.
  gpu-run --card 1. T=256 blk=32.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmk32_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 824.901 pipe_host
  831.953. 19.0 GB/s. timed act
  2783-2750 throttle=1. vs
  blk=16 847 (~1.02x) vs fused
  1086.

VERDICT -> ESIMD SLM-K blk=32
  T=256 is 832 us pipe_host
  card1, ~1.02x blk=16 847,
  ~1.31x fused 1086. throttle=1.
  Do not freeze 832 as 2800.
  Possible leftover cut. One-card.
  Sibling before promote. Rank
  pipe_host. Next: sibling blk=32
  vs blk=64.

### 2026-09-03go - K7 ESIMD fused delta T=256 SLM-K blk=32 sibling card0

CONTEXT -> card1 blk=32 was 832 us
  pipe_host, throttle=1. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk32. gpu-run
  --card 0. T=256 blk=32. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk32_t256.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 863.659 pipe_host
  862.027 vs card1 831.953.
  Spread ~3.6%. 18.3 GB/s. timed
  act=2650 cur=2800 throttle=1.
  vs blk=16 847-858.

VERDICT -> Sibling matches under
  5%. ESIMD SLM-K blk=32 is
  832-862 us pipe_host both
  cards, throttle=1. Wash vs
  blk=16 847-858 (clocks). Do
  not freeze 832 as 2800. Rank
  pipe_host.

### 2026-09-03gp - K7 ESIMD fused delta T=256 SLM-K blk=64 card1

CONTEXT -> blk=32 832-862. Napkin
  blk=64 fewer barriers. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmk64.
  gpu-run --card 1. T=256 blk=64.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmk64_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 827.867 pipe_host
  834.985. 18.9 GB/s. timed_begin
  act=2800 throttle=0. timed_end
  act=2733 throttle=1. vs blk=32
  832.

VERDICT -> ESIMD SLM-K blk=64
  T=256 is 835 us pipe_host
  card1, wash vs blk=32 832.
  throttle=1 at end. Stop larger
  blk vs 16/32. Do not freeze
  835. Rank pipe_host. Next:
  SLM-K T=64 vs SLM a/b.

### 2026-09-03gq - K7 ESIMD fused delta T=64 SLM-K card0

CONTEXT -> fused T=64 265-271
  throttle=1. SLM-K T=256 847.
  Napkin ~212 if T-linear. Same
  TU gdn_delta_slmk --t 64.
  spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk. gpu-run
  --card 0. T=64 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk_t64.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 213.732 pipe_host
  214.083. 29.5 GB/s. timed
  act=cur=2800 throttle=0. vs
  fused 265 (~1.24x) vs napkin
  212.

VERDICT -> ESIMD SLM-K T=64 is
  214 us pipe_host card0 at 2800,
  ~1.24x fused 265, T-linear vs
  847. throttle=0. Do not freeze
  214 until sibling. Possible
  T=64 leftover cut. Rank
  pipe_host.

### 2026-09-03gr - K7 ESIMD fused delta T=256 SLM a/b card1

CONTEXT -> SLM-K 847-858 leftover.
  Napkin a/b SLM cuts HBM scalars.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmab.
  gpu-run --card 1. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmab_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 848.302 pipe_host
  853.663. 18.5 GB/s. timed act
  2783-2750 throttle=1. vs SLM-K
  847-858.

VERDICT -> ESIMD SLM a/b T=256 is
  854 us pipe_host card1, wash vs
  SLM-K 847-858. Napkin miss.
  Stop a/b SLM vs k/q-only. Rank
  pipe_host. Next: sibling T=64
  vs v-prefetch T=256.

### 2026-09-03gs - K7 ESIMD fused delta T=64 SLM-K sibling card1

CONTEXT -> card0 SLM-K T=64 was
  214 us pipe_host at 2800.
  Possible leftover cut. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk. gpu-run
  --card 1. T=64 blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmk_t64.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 219.055 pipe_host
  217.843 vs card0 214.083.
  Spread ~1.8%. 28.9 GB/s. timed
  act=2750 cur=2800 throttle=1.
  vs fused 265.

VERDICT -> Sibling matches. ESIMD
  SLM-K T=64 is 214-218 us
  pipe_host both cards, ~1.23x
  fused 265. card0 2800
  throttle=0, card1 throttle=1.
  Do not freeze 214 as 2800.
  New T=64 leftover class. Rank
  pipe_host.

### 2026-09-03gt - K7 ESIMD fused delta T=256 v-prefetch card0

CONTEXT -> SLM-K 847 leftover.
  Napkin hoist 16 v loads out of
  inner loop. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmv.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmv_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 866.299 pipe_host
  873.078. 18.1 GB/s. timed act
  2783-2750 throttle=1. vs SLM-K
  847-858 (~1.03x).

VERDICT -> ESIMD v-prefetch T=256
  is 873 us pipe_host card0,
  ~1.03x SLM-K 847. Napkin miss.
  Stop v-prefetch vs SLM-K. Rank
  pipe_host. Next: SLM-K T=1 vs
  inner unroll T=256.

### 2026-09-03gu - K7 ESIMD fused delta T=1 SLM-K blk=1 card0

CONTEXT -> fused T=1 7.1 at 2800.
  slmk needs T%16. New blk=1 TU.
  spin=4000 hold.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmk1.
  gpu-run --card 0. T=1 blk=1.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk1_t1.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.2e-4
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 7.836 pipe_host
  8.149. 392 GB/s. timed
  act=cur=2800 throttle=0. vs
  fused 7.1 (~1.16x). event wash
  vs fused 7.825.

VERDICT -> ESIMD SLM-K T=1 blk=1
  is 8.15 us pipe_host card0 at
  2800, ~1.16x fused 7.1.
  Barrier tax. Stop SLM-K vs
  fused at decode. Rank
  pipe_host.

### 2026-09-03gv - K7 ESIMD fused delta T=256 SLM-K inner unroll card1

CONTEXT -> SLM-K 847 leftover.
  Napkin unroll tt=16 hides SLM.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmku.
  gpu-run --card 1. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmku_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 853.003 pipe_host
  856.296. 18.4 GB/s. timed
  act 2800-2783 throttle=1. vs
  SLM-K 847-858.

VERDICT -> ESIMD inner unroll
  T=256 is 856 us pipe_host
  card1, wash vs SLM-K 847-858.
  Napkin miss. Stop inner unroll
  vs SLM-K. Rank pipe_host.
  Next: SLM f32 k/q vs SLM
  double-buffer T=256.

### 2026-09-03gw - K7 ESIMD fused delta T=256 SLM f32 k/q card0

CONTEXT -> SLM-K 847 leftover.
  Napkin convert k/q once to f32
  in SLM. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmf32.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmf32_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 860.399 pipe_host
  867.995. 18.2 GB/s. timed
  act 2800-2767 throttle=1. vs
  SLM-K 847-858 (~1.02x).

VERDICT -> ESIMD SLM f32 k/q
  T=256 is 868 us pipe_host
  card0, ~1.02x SLM-K 847.
  Napkin miss. Stop f32 SLM vs
  half. Rank pipe_host.

### 2026-09-03gx - K7 ESIMD fused delta T=256 SLM double-buffer card1

CONTEXT -> SLM-K 847 leftover.
  Napkin ping-pong SLM, one
  barrier per blk. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmdb.
  gpu-run --card 1. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmdb_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 833.992 pipe_host
  842.973. 18.7 GB/s. timed
  act 2783-2733 throttle=1. vs
  SLM-K 847-858.

VERDICT -> ESIMD SLM db T=256 is
  843 us pipe_host card1,
  throttle=1, wash vs SLM-K
  847-858. Do not freeze 843 as
  2800. Stop double-buffer vs
  SLM-K. Rank pipe_host. Next:
  tree hsum T=256 vs SLM-K T=16.

### 2026-09-03gy - K7 ESIMD fused delta T=256 SLM-K tree hsum card0

CONTEXT -> SLM-K 847 leftover.
  Napkin esimd::reduce hsum vs
  scalar loop. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmh.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmh_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 416.516 pipe_host
  425.689. 37.1 GB/s. timed
  act 2800-2700 throttle=1. vs
  SLM-K 847 (~1.99x).

VERDICT -> ESIMD tree hsum T=256
  is 426 us pipe_host card0,
  ~1.99x SLM-K 847. throttle=1.
  New leftover class. Do not
  freeze 426 as 2800. Sibling
  before promote. Rank
  pipe_host.

### 2026-09-03gz - K7 ESIMD fused delta T=16 SLM-K card1

CONTEXT -> SLM-K T=256 847. T=64
  214. Napkin ~53 if T-linear.
  One blk. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk. gpu-run
  --card 1. T=16 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmk_t16.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 200.466 pipe_host
  58.130. 67.7 GB/s. timed_begin
  act=cur=550 throttle=0.
  timed_end act=cur=2800
  throttle=0. event min 57. vs
  napkin 53.

VERDICT -> ESIMD SLM-K T=16 is
  58 us pipe_host card1, near
  T-linear 53. Clocks ramped
  550 to 2800. Do not freeze 58
  as 2800. Hold retry. Rank
  pipe_host. Next: sibling tree
  hsum vs T=16 hold.

### 2026-09-03ha - K7 ESIMD fused delta T=16 SLM-K hold card0

CONTEXT -> card1 T=16 was 58 us
  pipe_host, start 550. Hold
  retry. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmk. gpu-run
  --card 0. T=16 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmk_t16.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 58.286 pipe_host
  58.600 vs card1 58.130.
  Spread ~0.8%. 67.2 GB/s.
  timed act=2767 cur=2800
  throttle=1. vs napkin 53.

VERDICT -> Sibling matches.
  ESIMD SLM-K T=16 is 58 us
  pipe_host both cards, near
  T-linear 53. throttle=1. Do
  not freeze 58 as 2800. Rank
  pipe_host.

### 2026-09-03hb - K7 ESIMD fused delta T=256 tree hsum sibling card1

CONTEXT -> card0 tree hsum was
  426 us pipe_host, throttle=1.
  New leftover class. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmh. gpu-run
  --card 1. T=256 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmh_t256.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 478.950 pipe_host
  477.332 vs card0 425.689.
  Spread ~12%. 33.1 GB/s. timed
  act=2417 cur=2800 throttle=1.
  vs SLM-K 847-858.

VERDICT -> Sibling clock-spread.
  ESIMD tree hsum T=256 is
  426-477 us pipe_host both
  cards, throttle=1. card1 act
  2417 not 2800. Do not freeze
  426 as 2800. New leftover
  class both cards. Rank
  pipe_host. Next: tree hsum
  T=64 vs tree hsum T=1.

### 2026-09-03hc - K7 ESIMD fused delta T=64 SLM-K tree hsum card0

CONTEXT -> SLM-K T=64 214 at
  2800. tree hsum T=256 426.
  Napkin ~107 if 2x. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmh. gpu-run
  --card 0. T=64 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmh_t64.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 109.112 pipe_host
  108.823. 57.9 GB/s. timed
  act=cur=2800 throttle=0. vs
  SLM-K 214 (~1.97x) vs napkin
  107.

VERDICT -> ESIMD tree hsum T=64
  is 109 us pipe_host card0 at
  2800, ~1.97x SLM-K 214.
  Napkin hit. T-linear vs 426.
  Possible leftover cut. Sibling
  before promote. Rank
  pipe_host.

### 2026-09-03hd - K7 ESIMD fused delta T=1 tree hsum card1

CONTEXT -> fused T=1 7.1 at
  2800. Scalar hsum. spin=4000
  hold.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_h.
  gpu-run --card 1. T=1.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_h.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0.0625
  cosine_o=1 max_abs_o=2 ok=1.
  event 7.237 pipe_host 6.085.
  525 GB/s. timed act=cur=2800
  throttle=0. vs fused 7.1
  (~1.16x). fused max_abs_o=0.

VERDICT -> ESIMD tree hsum T=1
  is 6.09 us pipe_host card1 at
  2800, ~1.16x fused 7.1.
  max_abs_o=2 vs fused 0.
  Numeric looser. Do not replace
  fused 7.1. Rank pipe_host.
  Next: sibling T=64 vs tree
  hsum T=16.

### 2026-09-03he - K7 ESIMD fused delta T=16 SLM-K tree hsum card0

CONTEXT -> SLM-K T=16 58. tree
  hsum T=64 109. Napkin ~29 if
  2x. spin=4000 hold.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmh. gpu-run
  --card 0. T=16 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmh_t16.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 34.380 pipe_host
  33.801. 116 GB/s. timed
  act=2583 cur=2800 throttle=1.
  vs SLM-K 58 (~1.72x) vs napkin
  29.

VERDICT -> ESIMD tree hsum T=16
  is 34 us pipe_host card0,
  ~1.72x SLM-K 58. throttle=1
  act=2583. Do not freeze 34 as
  2800. Possible leftover cut.
  Sibling before promote. Rank
  pipe_host.

### 2026-09-03hf - K7 ESIMD fused delta T=64 tree hsum sibling card1

CONTEXT -> card0 T=64 was 109 us
  pipe_host at 2800. Possible
  leftover cut. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmh. gpu-run
  --card 1. T=64 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmh_t64.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 124.950 pipe_host
  124.803 vs card0 108.823.
  Spread ~15%. 50.5 GB/s. timed
  act=2450 cur=2800 throttle=1.
  vs SLM-K 214.

VERDICT -> Sibling clock-spread.
  ESIMD tree hsum T=64 is
  109-125 us pipe_host both
  cards. card0 2800 throttle=0,
  card1 act=2450 throttle=1. Do
  not freeze 109 as 2800. New
  leftover class both cards.
  Rank pipe_host. Next: sibling
  T=16 vs tile-fused reduce
  T=256.

### 2026-09-03hg - K7 ESIMD fused delta T=256 tile-fused reduce card0

CONTEXT -> tree hsum T=256 426
  leftover. Napkin one 16-wide
  acc then one reduce. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmht.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 263.562 pipe_host
  260.132. 60.7 GB/s. timed
  act=2600 cur=2800 throttle=0.
  vs tree hsum 426 (~1.64x) vs
  SLM-K 847 (~3.25x).

VERDICT -> ESIMD tile-fused
  reduce T=256 is 260 us
  pipe_host card0, ~1.64x tree
  hsum 426. New leftover class.
  act=2600 not 2800. Do not
  freeze 260 as 2800. Sibling
  before promote. Rank
  pipe_host.

### 2026-09-03hh - K7 ESIMD fused delta T=16 tree hsum sibling card1

CONTEXT -> card0 T=16 was 34 us
  pipe_host, throttle=1. Possible
  leftover cut. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmh. gpu-run
  --card 1. T=16 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmh_t16.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 34.336 pipe_host
  33.683 vs card0 33.801.
  Spread ~0.3%. 117 GB/s. timed
  act=2633 cur=2800 throttle=1.
  vs SLM-K 58.

VERDICT -> Sibling matches.
  ESIMD tree hsum T=16 is 34 us
  pipe_host both cards, ~1.72x
  SLM-K 58. throttle=1. Do not
  freeze 34 as 2800. Rank
  pipe_host. Next: sibling
  tile-fused T=256 vs tile-fused
  T=64.

### 2026-09-03hi - K7 ESIMD fused delta T=64 tile-fused reduce card0

CONTEXT -> tree hsum T=64 109.
  tile-fused T=256 260. Napkin
  ~66 if 1.64x. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 0. T=64 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht_t64.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 154.266 pipe_host
  66.704. 94.5 GB/s. timed_begin
  act=cur=550. timed_end
  act=2700 cur=2800 throttle=0.
  event min 64. vs tree hsum 109
  (~1.63x) vs napkin 66.

VERDICT -> ESIMD tile-fused T=64
  is 67 us pipe_host card0, near
  napkin 66. Clocks ramped 550
  to 2700. Do not freeze 67 as
  2800. Hold retry. Rank
  pipe_host.

### 2026-09-03hj - K7 ESIMD fused delta T=256 tile-fused sibling card1

CONTEXT -> card0 tile-fused T=256
  was 260 us pipe_host. New
  leftover class. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 1. T=256 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht_t256.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 296.854 pipe_host
  294.043 vs card0 260.132.
  Spread ~13%. 53.7 GB/s. timed
  act 2283-2300 cur=2800
  throttle=1. vs tree hsum 426.

VERDICT -> Sibling clock-spread.
  ESIMD tile-fused T=256 is
  260-294 us pipe_host both
  cards. card0 act=2600, card1
  2283 throttle=1. Do not freeze
  260 as 2800. New leftover
  class both cards. Rank
  pipe_host. Next: T=64 hold vs
  tile-fused T=16.

### 2026-09-03hk - K7 ESIMD fused delta T=16 tile-fused reduce card0

CONTEXT -> tree hsum T=16 34.
  tile-fused T=64 67. Napkin ~21
  if 1.64x. spin=4000 hold.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 0. T=16 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht_t16.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 22.143 pipe_host
  21.712. 181 GB/s. timed
  act 2617-2600 cur=2800
  throttle=1. vs tree hsum 34
  (~1.56x) vs napkin 21.

VERDICT -> ESIMD tile-fused T=16
  is 22 us pipe_host card0,
  ~1.56x tree hsum 34. Napkin
  hit. throttle=1 act=2600. Do
  not freeze 22 as 2800.
  Possible leftover cut. Sibling
  before promote. Rank
  pipe_host.

### 2026-09-03hl - K7 ESIMD fused delta T=64 tile-fused hold card1

CONTEXT -> card0 T=64 was 67 us
  pipe_host, start 550. Hold
  retry / sibling. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 1. T=64 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht_t64.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 77.055 pipe_host
  76.650 vs card0 66.704.
  Spread ~15%. 82.2 GB/s. timed
  act=2367 cur=2800 throttle=1.
  vs tree hsum 109.

VERDICT -> Sibling clock-spread.
  ESIMD tile-fused T=64 is
  67-77 us pipe_host both cards.
  card1 act=2367 throttle=1. Do
  not freeze 67 as 2800. New
  leftover class both cards.
  Rank pipe_host. Next: sibling
  T=16 vs inner unroll T=256.

### 2026-09-03hm - K7 ESIMD fused delta T=256 slmht tt unroll card0

CONTEXT -> slmht T=256 260
  leftover. Napkin unroll inner
  tt=16. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmhtu.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmhtu_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 272.823 pipe_host
  275.567. 57.3 GB/s. timed
  act 2800-2767 throttle=1. vs
  slmht 260 (~1.06x).

VERDICT -> ESIMD slmht tt unroll
  T=256 is 276 us pipe_host
  card0, ~1.06x slmht 260.
  Napkin miss. Stop inner unroll
  vs slmht. Rank pipe_host.

### 2026-09-03hn - K7 ESIMD fused delta T=16 tile-fused sibling card1

CONTEXT -> card0 T=16 was 22 us
  pipe_host, throttle=1. Possible
  leftover cut. Sibling.
  spin=4000 hold. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 1. T=16 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht_t16.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 23.773 pipe_host
  21.286 vs card0 21.712.
  Spread ~2%. 185 GB/s. timed
  act=2633 cur=2800 throttle=1.
  vs tree hsum 34.

VERDICT -> Sibling matches.
  ESIMD tile-fused T=16 is 22 us
  pipe_host both cards, ~1.56x
  tree hsum 34. throttle=1. Do
  not freeze 22 as 2800. Rank
  pipe_host. Next: pack a/b/v
  T=256 vs tile-fused T=1.

### 2026-09-03ho - K7 ESIMD fused delta T=256 slmht pack a/b/v card0

CONTEXT -> slmht T=256 260
  leftover. Scalar a/b/v. Napkin
  SLM a/b plus 16-wide v along
  Dv. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmhtp.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmhtp_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 267.690 pipe_host
  266.427. 59.2 GB/s. timed
  act 2600-2650 cur=2800
  throttle=0. vs slmht 260
  (~1.02x).

VERDICT -> ESIMD slmht pack a/b/v
  T=256 is 266 us pipe_host
  card0, ~1.02x slmht 260.
  Napkin miss. Stop pack a/b/v
  vs slmht. Rank pipe_host.

### 2026-09-03hp - K7 ESIMD fused delta T=1 tile-fused reduce card1

CONTEXT -> fused T=1 7.1 at
  2800. tree hsum T=1 6.09
  max_abs_o=2. Napkin one
  16-wide acc then one reduce.
  spin=4000 hold.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_ht.
  gpu-run --card 1. T=1.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_ht.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0.03125
  cosine_o=1 max_abs_o=2 ok=1.
  event 6.437 pipe_host 5.542.
  577 GB/s. timed act=cur=2800
  throttle=0. vs fused 7.1
  (~1.28x) vs tree hsum 6.09.

VERDICT -> ESIMD tile-fused T=1
  is 5.54 us pipe_host card1 at
  2800, ~1.28x fused 7.1.
  max_abs_o=2 vs fused 0. Same
  numeric class as tree hsum
  T=1. Do not replace fused
  7.1. Rank pipe_host. Next:
  slmht blk=8 T=256 vs T=1
  tile-fused scalar hsum.

### 2026-09-03hq - K7 ESIMD fused delta T=256 slmht blk=8 card0

CONTEXT -> slmht T=256 260
  leftover blk=16. Napkin half
  block, more barriers. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmht8.
  gpu-run --card 0. T=256 blk=8.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht8_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 274.242 pipe_host
  269.210. 58.6 GB/s. timed
  act 2700-2667 cur=2800
  throttle=0. vs slmht 260
  (~1.04x).

VERDICT -> ESIMD slmht blk=8
  T=256 is 269 us pipe_host
  card0, ~1.04x slmht 260.
  Napkin miss. Stop blk=8 vs
  slmht. Rank pipe_host.

### 2026-09-03hr - K7 ESIMD fused delta T=1 tile-fused scalar hsum card1

CONTEXT -> ht T=1 5.54 at 2800
  max_abs_o=2. Napkin scalar
  hsum vs esimd::reduce. spin=4000
  hold.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_hts.
  gpu-run --card 1. T=1.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_hts.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0.0625
  cosine_o=1 max_abs_o=2 ok=1.
  event 6.630 pipe_host 6.088.
  525 GB/s. timed act=cur=2800
  throttle=0. vs ht 5.54 vs
  fused 7.1 vs tree hsum 6.09.

VERDICT -> ESIMD tile-fused
  scalar hsum T=1 is 6.09 us
  pipe_host card1 at 2800.
  Wash vs tree hsum. ~1.10x ht
  reduce 5.54. max_abs_o=2 vs
  fused 0. Numeric is the fused
  acc, not esimd::reduce. Stop
  scalar hsum vs reduce. Do not
  replace fused 7.1. Rank
  pipe_host. Next: slmht blk=32
  T=256 vs packed-o T=256.

### 2026-09-03hs - K7 ESIMD fused delta T=256 slmht blk=32 card0

CONTEXT -> slmht T=256 260
  leftover blk=16. Napkin
  double block, fewer barriers.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmht32.
  gpu-run --card 0. T=256 blk=32.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht32_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 256.893 pipe_host
  252.173. 62.6 GB/s. timed
  act=2600 cur=2800 throttle=0.
  vs slmht 260 at 2600 (~1.03x).

VERDICT -> ESIMD slmht blk=32
  T=256 is 252 us pipe_host
  card0 at 2600, ~1.03x slmht
  260. Possible leftover cut.
  Do not freeze 252 as 2800.
  Sibling before promote. Rank
  pipe_host.

### 2026-09-03ht - K7 ESIMD fused delta T=256 slmht packed-o card1

CONTEXT -> slmht T=256 260
  leftover. Scalar o store.
  Napkin 16-wide o along Dv.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmhto.
  gpu-run --card 1. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmhto_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 251.826 pipe_host
  247.158. 63.8 GB/s. timed
  act=2700 cur=2800 throttle=0.
  vs slmht 260 at 2600 (~1.05x).

VERDICT -> ESIMD slmht packed-o
  T=256 is 247 us pipe_host
  card1 at 2700, ~1.05x slmht
  260. Clock 2700 vs 2600.
  Possible leftover cut. Do not
  freeze 247 as 2800. Sibling
  before promote. Rank
  pipe_host. Next: sibling
  packed-o vs sibling blk=32.

### 2026-09-03hu - K7 ESIMD fused delta T=256 packed-o sibling card0

CONTEXT -> card1 packed-o was
  247 us at 2700. Possible cut.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmhto. gpu-run
  --card 0. T=256 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmhto_t256.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 298.659 pipe_host
  296.793 vs card1 247.158.
  Spread ~20%. 53.2 GB/s. timed
  act=2233 cur=2800 throttle=1.
  vs slmht sibling 294.

VERDICT -> Sibling clock-spread.
  ESIMD packed-o T=256 is 247-
  297 us pipe_host both cards.
  card0 act=2233 throttle=1.
  Clock-linear vs 247 at 2700.
  Do not freeze 247 as 2800.
  Not a kernel cut. Stop packed-o
  vs slmht. Rank pipe_host.

### 2026-09-03hv - K7 ESIMD fused delta T=256 slmht blk=32 sibling card1

CONTEXT -> card0 blk=32 was 252
  us at 2600. Possible cut.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht32. gpu-run
  --card 1. T=256 blk=32.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht32_t256.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 294.375 pipe_host
  292.134 vs card0 252.173.
  Spread ~16%. 54.0 GB/s. timed
  act 2233-2250 cur=2800
  throttle=1. vs slmht sibling
  294.

VERDICT -> Sibling clock-spread.
  ESIMD slmht blk=32 T=256 is
  252-292 us pipe_host both
  cards. card1 act=2233
  throttle=1. Clock-linear vs
  252 at 2600. Do not freeze
  252 as 2800. ~1.03x at
  matched 2600 is too small to
  promote. Rank pipe_host.
  Next: slmht 2-row T=256 vs
  blk=32 T=64. spin=0 on T=256.

### 2026-09-03hw - K7 ESIMD fused delta T=256 slmht 2-row card0

CONTEXT -> slmht T=256 260
  leftover. Napkin two i-rows
  per WI. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmht2.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht2_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 327.638 pipe_host
  327.459. 48.2 GB/s. timed
  act=cur=2800 throttle=0. vs
  slmht 260 (~1.26x).

VERDICT -> ESIMD slmht 2-row
  T=256 is 327 us pipe_host
  card0 at 2800, ~1.26x slmht
  260. Napkin miss. Stop 2-row
  vs slmht. Rank pipe_host.

### 2026-09-03hx - K7 ESIMD fused delta T=64 slmht blk=32 card1

CONTEXT -> slmht T=64 67.
  blk=32 T=256 ~1.03x at 2600.
  Napkin scale. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht32. gpu-run
  --card 1. T=64 blk=32. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht32_t64.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 155.943 pipe_host
  66.502. 94.8 GB/s. timed_begin
  act=cur=550. timed_end
  act=2700 cur=2800 throttle=0.
  event min 64. vs slmht T=64 67.

VERDICT -> ESIMD slmht blk=32
  T=64 is 67 us pipe_host card1,
  wash vs slmht 67. 1.03x does
  not scale. Clocks ramped 550
  to 2700. Do not freeze 67 as
  2800. Stop blk=32 at T=64.
  Rank pipe_host. Next: slmht
  SLM-db T=256 vs slmht8 T=8.

### 2026-09-03hy - K7 ESIMD fused delta T=256 slmht SLM-db card0

CONTEXT -> slmht T=256 260
  leftover. Napkin ping-pong
  k/q SLM, overlap next fill.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_slmhtdb.
  gpu-run --card 0. T=256 blk=16.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmhtdb_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 268.232 pipe_host
  268.173. 58.8 GB/s. timed
  act 2600-2550 cur=2800
  throttle=0. vs slmht 260
  (~1.03x).

VERDICT -> ESIMD slmht SLM-db
  T=256 is 268 us pipe_host
  card0, ~1.03x slmht 260.
  Napkin miss. Stop SLM-db vs
  slmht. Rank pipe_host.

### 2026-09-03hz - K7 ESIMD fused delta T=8 tile-fused blk=8 card1

CONTEXT -> fused T=1 7.1.
  tile-fused T=16 22. Napkin
  map T=8. spin=4000 hold.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht8. gpu-run
  --card 1. T=8 blk=8. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht8_t8.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 14.859 pipe_host
  12.526. 283 GB/s. timed
  act=2750 cur=2800 throttle=1.
  vs fused 7.1 vs T=16 22.

VERDICT -> ESIMD tile-fused T=8
  is 13 us pipe_host card1 at
  2750, ~1.76x fused 7.1, ~1.74x
  under T=16 22. Near half T=16.
  throttle=1. Do not freeze 13
  as 2800. Sibling before citing
  the map. Rank pipe_host. Next:
  sibling T=8 vs slmht T=32.

### 2026-09-03ia - K7 ESIMD fused delta T=8 tile-fused sibling card0

CONTEXT -> card1 T=8 was 13 us
  at 2750 throttle=1. Map.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht8. gpu-run
  --card 0. T=8 blk=8. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht8_t8.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 13.482 pipe_host
  12.393 vs card1 12.526.
  Spread ~1%. 286 GB/s. timed
  act=2733 cur=2800 throttle=1.
  vs fused 7.1 vs T=16 22.

VERDICT -> Sibling matches.
  ESIMD tile-fused T=8 is 13 us
  pipe_host both cards, ~1.76x
  fused 7.1, near half T=16 22.
  throttle=1. Do not freeze 13
  as 2800. Rank pipe_host.

### 2026-09-03ib - K7 ESIMD fused delta T=32 tile-fused card1

CONTEXT -> T=16 22. T=64 67.
  Napkin ~40. spin=4000 hold.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 1. T=32 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht_t32.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 40.987 pipe_host
  38.968. 121 GB/s. timed
  act 2483-2467 cur=2800
  throttle=1. vs T=16 22
  (~1.77x) vs T=64 67 vs napkin
  40.

VERDICT -> ESIMD tile-fused T=32
  is 39 us pipe_host card1,
  napkin 40. throttle=1 act=2470.
  Do not freeze 39 as 2800.
  Sibling before citing the map.
  Rank pipe_host. Next: sibling
  T=32 vs slmht32 T=128.

### 2026-09-03ic - K7 ESIMD fused delta T=32 tile-fused sibling card0

CONTEXT -> card1 T=32 was 39 us
  at 2470 throttle=1. Map.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 0. T=32 blk=16.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht_t32.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 40.727 pipe_host
  39.398 vs card1 38.968.
  Spread ~1%. 120 GB/s. timed
  act=2450 cur=2800 throttle=1.
  vs T=16 22 vs T=64 67.

VERDICT -> Sibling matches.
  ESIMD tile-fused T=32 is 39 us
  pipe_host both cards, napkin
  40. throttle=1. Do not freeze
  39 as 2800. Rank pipe_host.

### 2026-09-03id - K7 ESIMD fused delta T=128 slmht32 card1

CONTEXT -> T=64 67. T=256 260.
  Napkin ~130. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht32. gpu-run
  --card 1. T=128 blk=32. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht32_t128.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 126.724 pipe_host
  124.610. 75.9 GB/s. timed
  act=cur=2700 throttle=0. vs
  T=64 67 (~1.87x) vs T=256 260
  vs napkin 130.

VERDICT -> ESIMD slmht32 T=128
  is 125 us pipe_host card1 at
  2700, napkin 130. Do not
  freeze 125 as 2800. Sibling
  before citing the map. Rank
  pipe_host. Next: sibling
  slmht32 T=128 vs slmht T=128.

### 2026-09-03ie - K7 ESIMD fused delta T=128 slmht32 sibling card0

CONTEXT -> card1 slmht32 T=128
  was 125 us at 2700. Map.
  Sibling. spin=0. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht32. gpu-run
  --card 0. T=128 blk=32. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht32_t128.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 130.378 pipe_host
  129.673 vs card1 124.610.
  Spread ~4%. 73.0 GB/s. timed
  act=2600 cur=2800 throttle=0.
  vs napkin 130.

VERDICT -> Sibling clock-spread
  4%. ESIMD slmht32 T=128 is
  125-130 us pipe_host both
  cards at 2600-2700. Napkin
  130. Do not freeze 125 as
  2800. Rank pipe_host.

### 2026-09-03if - K7 ESIMD fused delta T=128 tile-fused card1

CONTEXT -> T=64 67. T=256 260.
  slmht32 T=128 125. Napkin
  ~130. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 1. T=128 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_slmht_t128.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 129.042 pipe_host
  126.655. 74.7 GB/s. timed
  act=cur=2700 throttle=0. vs
  slmht32 125 (~1.02x) vs T=64
  67 vs T=256 260.

VERDICT -> ESIMD tile-fused
  T=128 is 127 us pipe_host
  card1 at 2700, wash vs slmht32
  125. Napkin 130. Stop blk=32
  at T=128. Do not freeze 127 as
  2800. Sibling before citing
  the map. Rank pipe_host. Next:
  sibling slmht T=128 vs mixer
  T=256 retry.

### 2026-09-03ig - K7 ESIMD fused delta T=128 tile-fused sibling card0

CONTEXT -> card1 T=128 was 127
  us at 2700. Map. Sibling.
  spin=0. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_delta_slmht. gpu-run
  --card 0. T=128 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_delta_slmht_t128.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 132.412 pipe_host
  131.440 vs card1 126.655.
  Spread ~4%. 72.0 GB/s. timed
  act=2600 cur=2800 throttle=0.
  vs T=64 67 vs T=256 260.

VERDICT -> Sibling clock-spread
  4%. ESIMD tile-fused T=128 is
  127-131 us pipe_host both
  cards at 2600-2700. Do not
  freeze 127 as 2800. T-map
  closed. Rank pipe_host.

### 2026-09-03ih - K7 ESIMD mixer T=256 card1

CONTEXT -> mixer T=64 395 vs
  seq 275. Leftover now conv 38
  + slmht 260 ~298. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_t. gpu-run
  --card 1. T=256 C=10240.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=7.6e-6
  cosine_o=1 max_abs_o=2.4e-4
  ok=1. event 1539.995 pipe_host
  1557.055. timed act 2750-2667
  cur=2800 throttle=1. vs seq
  ~298 (~5.2x) vs mixer T=64
  395 (~3.9x).

VERDICT -> ESIMD mixer T=256 is
  1557 us pipe_host card1,
  ~5.2x seq 298. Old delta
  path. Stop packed mixer at
  T=256. Do not freeze 1557 as
  2800. Rank pipe_host. Next:
  mixer-slmht T=256 vs skip-hi
  T=256.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ii - K7 ESIMD mixer-slmht T=256 card0

CONTEXT -> packed mixer T=256
  1557 vs seq conv 38 + slmht
  260 ~298. New TU
  gdn_mixer_slmht. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_slmht.
  gpu-run --card 0. T=256
  C=10240 nv=48 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_slmht_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 470.435 pipe_host
  470.656. timed act 2800-2783
  cur=2800 throttle=0. vs packed
  1557 (~3.31x) vs seq 298
  (~1.58x).

VERDICT -> ESIMD mixer-slmht
  T=256 is 471 us pipe_host
  card0 at 2800, ~3.31x packed
  mixer 1557, ~1.58x seq 298.
  First fuse. Sibling before
  promote. Rank pipe_host. Next:
  sibling mixer-slmht T=256 vs
  mixer-slmht T=64.

### 2026-09-03ij - K7 ESIMD skip-hi T=256 card1

CONTEXT -> slmht leftover 260.
  Gated b: even-t b=0 skip_frac
  0.5, skip vold+rank1. Napkin
  ~180 if skip is real. New TU
  gdn_delta_skiphi. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_delta_skiphi.
  gpu-run --card 1. T=256 blk=16
  nv=48 skip_frac=0.5 even_t_b=0.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_delta_skiphi_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=1.2e-4
  ok=1. event 329.021 pipe_host
  329.899. 47.8 GB/s. timed
  act=cur=2800 throttle=0. vs
  slmht 260 (~1.27x) vs napkin
  180.

VERDICT -> ESIMD skip-hi T=256
  is 330 us pipe_host card1 at
  2800, ~1.27x slmht 260.
  Napkin 180 died. Branch tax.
  Stop skip-hi vs slmht leftover.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ik - K7 ESIMD mixer-slmht T=64 card0

CONTEXT -> packed mixer T=64
  395 vs seq conv 10.5 + slmht
  67 ~77. T=256 mixer-slmht
  471. Same TU. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 0. T=64 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_slmht_t64.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 117.503 pipe_host
  117.467. timed act=2683
  cur=2800 throttle=1. vs packed
  395 (~3.36x) vs seq 77
  (~1.53x) vs T=256 471
  (~4.01x).

VERDICT -> ESIMD mixer-slmht
  T=64 is 117 us pipe_host
  card0, ~3.36x packed 395,
  ~1.53x seq 77, T-linear vs
  471. throttle=1. Do not freeze
  117 as 2800. Sibling before
  citing the map. Rank
  pipe_host.

### 2026-09-03il - K7 ESIMD mixer-slmht T=256 sibling card1

CONTEXT -> card0 mixer-slmht
  T=256 was 471 us at 2800.
  First fuse. Sibling. spin=0.
  Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 1. T=256 C=10240
  nv=48 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_slmht_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 471.367 pipe_host
  470.966 vs card0 470.656.
  Spread ~0.07%. timed
  act=cur=2800 throttle=0.

VERDICT -> Sibling matches.
  ESIMD mixer-slmht T=256 is
  471 us pipe_host both cards
  at 2800, ~3.31x packed 1557,
  ~1.58x seq 298. Promote.
  Rank pipe_host. Next: sibling
  mixer-slmht T=64 vs T=128.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03im - K7 ESIMD mixer-slmht T=128 card0

CONTEXT -> T=64 mixer-slmht 117.
  T=256 471. Napkin T-linear
  235. seq slmht 127 + conv ~20.
  Same TU. spin=0.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 0. T=128 C=10240 nv=48
  blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_slmht_t128.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 261.862 pipe_host
  232.321. timed act 1600-2800
  cur 1583-2800 throttle=0. vs
  napkin 235 vs T=64 117
  (~1.98x) vs T=256 471
  (~2.03x) vs seq ~147
  (~1.58x).

VERDICT -> ESIMD mixer-slmht
  T=128 is 232 us pipe_host
  card0, napkin 235, T-linear.
  Clocks ramped 1600 to 2800.
  Do not freeze 232 as 2800.
  Sibling before citing the map.
  Rank pipe_host.

### 2026-09-03in - K7 ESIMD mixer-slmht T=64 sibling card1

CONTEXT -> card0 T=64 was 117 us
  at 2683 throttle=1. Map.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 1. T=64 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_slmht_t64.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 116.143 pipe_host
  116.017 vs card0 117.467.
  Spread ~1.2%. timed act=2717
  cur=2800 throttle=1. vs packed
  395 vs seq 77.

VERDICT -> Sibling matches.
  ESIMD mixer-slmht T=64 is
  116-117 us pipe_host both
  cards, throttle=1. Do not
  freeze 117 as 2800. Rank
  pipe_host. Next: sibling
  mixer-slmht T=128 vs T=32.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03io - K7 ESIMD mixer-slmht T=32 card0

CONTEXT -> T=64 mixer-slmht 117.
  slmht T=32 39. Napkin T-linear
  ~58. Same TU. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 0. T=32 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_slmht_t32.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 59.057 pipe_host
  59.779. timed act=2683
  cur=2800 throttle=1. vs napkin
  58 vs T=64 117 (~1.96x) vs
  slmht 39 (~1.53x).

VERDICT -> ESIMD mixer-slmht
  T=32 is 60 us pipe_host
  card0, napkin 58, T-linear vs
  117. throttle=1. Do not freeze
  60 as 2800. Sibling before
  citing the map. Rank
  pipe_host.

### 2026-09-03ip - K7 ESIMD mixer-slmht T=128 sibling card1

CONTEXT -> card0 T=128 was 232
  us, clocks ramped 1600 to
  2800. Map. Sibling. spin=0.
  Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 1. T=128 C=10240 nv=48
  blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_slmht_t128.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 232.526 pipe_host
  232.288 vs card0 232.321.
  Spread ~0.01%. timed
  act=cur=2800 throttle=0.

VERDICT -> Sibling matches.
  ESIMD mixer-slmht T=128 is
  232 us pipe_host both cards,
  card1 at 2800. Napkin 235.
  T-linear. Rank pipe_host.
  Next: sibling mixer-slmht
  T=32 vs T=16.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03iq - K7 ESIMD mixer-slmht T=16 card0

CONTEXT -> T=32 mixer-slmht 60.
  slmht T=16 22. Napkin T-linear
  ~29. Same TU. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 0. T=16 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_slmht_t16.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 31.188 pipe_host
  31.295. timed act=2733
  cur=2800 throttle=1. vs napkin
  29 vs T=32 60 (~1.92x) vs
  slmht 22 (~1.42x).

VERDICT -> ESIMD mixer-slmht
  T=16 is 31 us pipe_host
  card0, napkin 29. throttle=1.
  Do not freeze 31 as 2800.
  Sibling before citing the map.
  Rank pipe_host.

### 2026-09-03ir - K7 ESIMD mixer-slmht T=32 sibling card1

CONTEXT -> card0 T=32 was 60 us
  at 2683 throttle=1. Map.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 1. T=32 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_slmht_t32.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 59.713 pipe_host
  59.233 vs card0 59.779.
  Spread ~0.9%. timed act=2717
  cur=2800 throttle=1. vs napkin
  58 vs slmht 39.

VERDICT -> Sibling matches.
  ESIMD mixer-slmht T=32 is
  59-60 us pipe_host both
  cards, throttle=1. Do not
  freeze 60 as 2800. Rank
  pipe_host. Next: sibling
  mixer-slmht T=16 vs conv
  T=16 C=10240.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03is - K7 ESIMD conv1d T=16 C=10240 card0

CONTEXT -> C=10240 T=64 10.5.
  slmht T=16 22. mixer-slmht
  T=16 31. seq control. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. T=16 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t16_c10240.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 5.242 pipe_host
  5.710. 129 GB/s. timed
  act=1650 cur=1617 throttle=0.
  vs T=64 10.5 (~1.84x) vs
  mixer 31 vs slmht 22. seq
  ~28 vs mixer 31 (~1.12x).

VERDICT -> ESIMD conv T=16
  C=10240 is 5.7 us pipe_host
  card0 at 1650, T-linear vs
  10.5. Clocks not held. Do not
  freeze 5.7 as 2800. Sibling
  hold. Rank pipe_host.

### 2026-09-03it - K7 ESIMD mixer-slmht T=16 sibling card1

CONTEXT -> card0 T=16 was 31 us
  at 2733 throttle=1. Map.
  Sibling hold. spin=4000. Same
  TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_slmht. gpu-run
  --card 1. T=16 C=10240 nv=48
  blk=16. spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_slmht_t16.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=3.1e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 32.039 pipe_host
  31.184 vs card0 31.295.
  Spread ~0.4%. timed act=2750
  cur=2800 throttle=1. vs
  slmht 22 vs seq ~28.

VERDICT -> Sibling matches.
  ESIMD mixer-slmht T=16 is
  31 us pipe_host both cards,
  throttle=1. T-map blk=16
  closed. Do not freeze 31 as
  2800. Rank pipe_host. Next:
  conv T=16 hold vs conv T=32
  C=10240.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03iu - K7 ESIMD conv1d T=32 C=10240 card0

CONTEXT -> T=16 C=10240 5.7 at
  1650. T=64 10.5. slmht T=32
  39. seq control. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. T=32 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t32_c10240.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 5.401 pipe_host
  5.937. 235 GB/s. timed
  act=cur=2800 throttle=0. vs
  T=16 4.8 (~1.23x) vs T=64
  10.5 (~1.77x) vs slmht 39.
  seq ~45 vs mixer 60 (~1.34x).

VERDICT -> ESIMD conv T=32
  C=10240 is 5.9 us pipe_host
  card0 at 2800. seq ~45 vs
  mixer 60. Sibling before
  citing the map. Rank
  pipe_host.

### 2026-09-03iv - K7 ESIMD conv1d T=16 C=10240 hold card1

CONTEXT -> card0 T=16 was 5.7 us
  at 1650. Hold sibling.
  spin=4000. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. T=16 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t16_c10240.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 3.065 pipe_host
  4.833. 153 GB/s. timed
  act=cur=2800 throttle=0. vs
  card0 5.710 at 1650. Spread
  ~18% clock. vs T=64 10.5
  (~2.17x). seq ~27 vs mixer 31
  (~1.16x).

VERDICT -> Hold matches at 2800.
  ESIMD conv T=16 C=10240 is
  4.8 us pipe_host card1 at
  2800. Card0 5.7 at 1650.
  Clock-spread. Do not freeze
  5.7 as 2800. Rank pipe_host.
  Next: sibling conv T=32 vs
  conv T=128 C=10240.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03iw - K7 ESIMD conv1d T=128 C=10240 card0

CONTEXT -> T=32 5.9. T=64 10.5.
  T=256 40.7. Napkin T-linear
  ~20. slmht T=128 127. seq
  control. spin=4000.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 0. T=128 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_conv1d_t128_c10240.sh 0 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 19.542 pipe_host
  19.946. 267 GB/s. timed
  act=cur=2800 throttle=0. vs
  napkin 20 vs T=64 10.5
  (~1.90x) vs T=256 40.7
  (~2.04x). seq ~147 vs mixer
  232 (~1.58x).

VERDICT -> ESIMD conv T=128
  C=10240 is 20 us pipe_host
  card0 at 2800, napkin 20,
  T-linear. Sibling before
  citing the map. Rank
  pipe_host.

### 2026-09-03ix - K7 ESIMD conv1d T=32 C=10240 sibling card1

CONTEXT -> card0 T=32 was 5.9 us
  at 2800. Map. Sibling.
  spin=4000. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. T=32 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t32_c10240.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 5.406 pipe_host
  5.771 vs card0 5.937. Spread
  ~2.8%. timed act=cur=2800
  throttle=0. 235-241 GB/s. seq
  ~45 vs mixer 60.

VERDICT -> Sibling matches.
  ESIMD conv T=32 C=10240 is
  5.8-5.9 us pipe_host both
  cards at 2800. seq ~45 vs
  mixer 60. Rank pipe_host.
  Next: sibling conv T=128 vs
  mixer L2-out T=256.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03iy - K7 ESIMD mixer L2-out T=256 card0

CONTEXT -> mixer-slmht 471 vs
  seq conv 38 + slmht 260 ~298.
  New TU gdn_mixer_l2out: host
  L2, packed delta skips device
  L2. spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_l2out.
  gpu-run --card 0. T=256
  C=10240 nv=48 blk=16 host-L2.
  spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_l2out_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 271.320 pipe_host
  270.767. timed act 2700-2650
  cur=2800 throttle=0. vs slmht
  260 (~1.04x) vs mixer 471 vs
  conv+l2out ~309.

VERDICT -> ESIMD mixer L2-out
  T=256 is 271 us pipe_host
  card0, wash vs slmht 260.
  Packed tax ~4%. Device L2 is
  the mixer leftover. First
  fuse. Do not freeze 271 as
  2800. Sibling before promote.
  Rank pipe_host.

### 2026-09-03iz - K7 ESIMD conv1d T=128 C=10240 sibling card1

CONTEXT -> card0 T=128 was 20 us
  at 2800. Map. Sibling.
  spin=4000. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_conv1d_t. gpu-run
  --card 1. T=128 C=10240 k=4.
  spin=4000.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_conv1d_t128_c10240.sh 1 4000
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 19.539 pipe_host
  19.929 vs card0 19.946.
  Spread ~0.09%. timed
  act=cur=2800 throttle=0. 267
  GB/s. seq ~147 vs mixer 232.

VERDICT -> Sibling matches.
  ESIMD conv T=128 C=10240 is
  20 us pipe_host both cards at
  2800. T-map C=10240 closed.
  Rank pipe_host. Next: sibling
  mixer L2-out T=256 vs mixer
  L2-once T=256.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ja - K7 ESIMD mixer L2-once T=256 card0

CONTEXT -> mixer-slmht 471.
  L2-out 271. L2 once per
  (t,kh) then packed delta.
  New TU gdn_mixer_l2once.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_l2once.
  gpu-run --card 0. T=256
  C=10240 nv=48 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_l2once_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 319.096 pipe_host
  326.779. timed act 2700-2667
  cur=2800 throttle=0. vs mixer
  471 (~1.44x) vs seq 298
  (~1.10x) vs conv+l2out ~309.

VERDICT -> ESIMD mixer L2-once
  T=256 is 327 us pipe_host
  card0, beats mixer-slmht 471,
  loses to seq 298. Extra
  launch. First fuse. Do not
  freeze 327 as 2800. Sibling
  before promote. Rank
  pipe_host.

### 2026-09-03jb - K7 ESIMD mixer L2-out T=256 sibling card1

CONTEXT -> card0 L2-out was 271
  us at 2650-2700. First fuse.
  Sibling. spin=0. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_l2out. gpu-run
  --card 1. T=256 C=10240 nv=48
  blk=16 host-L2. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_l2out_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 271.094 pipe_host
  266.844 vs card0 270.767.
  Spread ~1.5%. timed act
  2700-2767 cur=2800 throttle=1.
  vs slmht 260.

VERDICT -> Sibling matches.
  ESIMD mixer L2-out T=256 is
  267-271 us pipe_host both
  cards, wash vs slmht 260.
  Packed tax ~4%. Do not freeze
  271 as 2800. Rank pipe_host.
  Next: sibling mixer L2-once
  T=256 vs mixer conv-L2 fuse.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jc - K7 ESIMD mixer L2-once T=256 sibling card1

CONTEXT -> card0 L2-once was 327
  us at 2700-2667. First fuse.
  Sibling. spin=0. Same TU.

CONFIG -> backend sycl+l0, same
  AOT gdn_mixer_l2once. gpu-run
  --card 1. T=256 C=10240 nv=48
  blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_mixer_l2once_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 317.284 pipe_host
  312.274 vs card0 326.779.
  Spread ~4.4%. timed act
  2700-2767 cur=2800 throttle=1.
  vs mixer-slmht 471 vs seq 298
  vs L2-out 267-271.

VERDICT -> Sibling matches.
  ESIMD mixer L2-once T=256 is
  312-327 us pipe_host both
  cards, 327-class. Beats mixer
  471, loses to seq 298. Extra
  launch. Do not freeze 327 as
  2800. Rank pipe_host. Next:
  mixer conv-L2 fuse T=256.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jd - K7 ESIMD mixer conv-L2 fuse T=256 card0

CONTEXT -> L2-once 327 is three
  kernels (conv, L2, packed
  delta). Drop L2 launch: fuse
  conv+L2 in one kernel via
  per-t SLM reduce on q/k, then
  packed delta. Napkin: L2 free
  in conv epilogue, ~298.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_convl2.
  gpu-run --card 0. T=256
  C=10240 nv=48 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_convl2_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 357.836 pipe_host
  358.198. timed act=2700
  cur=2800 throttle=0. vs
  L2-once 327 (~1.10x) vs seq
  298 (~1.20x) vs mixer 471.

VERDICT -> ESIMD mixer conv-L2
  T=256 is 358 us pipe_host
  card0, a loss vs L2-once 327
  and seq 298. Per-t SLM
  barriers in conv are the tax,
  not the extra launch. Stop
  this fuse. Sequential
  conv+slmht ~298 stays the
  T=256 leftover. Do not freeze
  358 as 2800. Rank pipe_host.
  Next: packed qkv ESIMD s8 vs
  W8A8 96/140/164.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03je - K7 ESIMD packed qkv s8 M=1 card0

CONTEXT -> oneDNN packed qkv
  W8A8 M=1 is 96 us. Square s8
  scale-to-f16 is 34 us at
  N=5120. Same dpas_s8_sc tile
  at packed n=10240 k=5120.
  Napkin N-linear 68 us.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 0. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s8_qkv_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 73.078
  pipe_host 73.945. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 96
  (~0.77x, a beat) vs square
  s8 34 (~2.17x, N=2x napkin
  68).

VERDICT -> ESIMD packed qkv s8
  M=1 is 73.945 us pipe_host
  card0 at 2800. Beats oneDNN
  packed qkv W8A8 96. ~2.17x
  square s8 34, near N-linear
  68. Numeric closed. One-card.
  Rank pipe_host. Next: packed
  qkv ESIMD s8 M=64 vs W8A8 140.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jf - K7 ESIMD s8 packed qkv M=64 card1

CONTEXT -> oneDNN packed qkv W8A8
  M=64 is 138-142 us. square s8
  4x8 A-db is 75 us at N=5120.
  Napkin N-linear 75*2 ~150 us.
  Same tile at packed n=10240.

CONFIG -> backend sycl+l0,
  standalone dpas_s8_sc8db48.
  gpu-run --card 1. NT=2 U=16
  spin=512 m=64 n=10240 k=5120
  wg=4x8 A-db out=f16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s8_qkv_m64.sh 1 512
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 212.604 pipe_host
  214.369. timed act=cur=2800
  throttle=0. vs W8A8 138-142
  (~1.53x) vs square 75 (~2.86x)
  vs napkin 150.

VERDICT -> ESIMD s8 packed qkv
  M=64 is 214 us pipe_host
  card1 at 2800, a loss vs
  oneDNN 140. Worse than
  N-linear. One-card. Do not
  freeze 214 until sibling.
  Rank pipe_host. Next: sibling
  M=64 card0 vs packed qkv
  ESIMD s8 M=256 vs W8A8 164.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jg - K7 ESIMD packed qkv s8 M=1 sibling card1

CONTEXT -> card0 packed qkv s8
  M=1 was 73.945 us pipe_host
  at 2800 (2026-09-03je).
  oneDNN W8A8 96. square s8 34.
  Sibling hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s8_qkv_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 73.214
  pipe_host 73.782. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs card0 73.945.
  Spread ~0.2%. vs W8A8 96
  (~0.77x) vs square s8 34
  (~2.17x).

VERDICT -> Sibling matches.
  Packed qkv s8 M=1 is 74-class
  us pipe_host both cards at
  2800. Beats oneDNN packed qkv
  W8A8 96. Numeric closed.
  Rank pipe_host. Next: packed
  qkv ESIMD s8 M=256 vs W8A8
  164, and sibling M=64 card0.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jh - K7 ESIMD s8 packed qkv M=256 card0

CONTEXT -> oneDNN packed qkv W8A8
  M=256 is 164 us. square s8
  4-acc 4x8 is 128 us at N=5120.
  Napkin N-linear 128*2 ~256 us.
  Same tile at packed n=10240.

CONFIG -> backend sycl+l0,
  standalone dpas_s8_sc8w48m4.
  gpu-run --card 0. NT=2 U=8
  spin=512 m=256 n=10240 k=5120
  wg=4x8 4acc out=f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s8_qkv_m256.sh 0 512
  ```

RESULT -> cosine=1 max_abs=0
  ok=1. event 272.672 pipe_host
  274.205. timed act 2733-2717
  cur=2800 throttle=1. vs W8A8
  164 (~1.67x) vs square 128
  (~2.14x) vs napkin 256.

VERDICT -> ESIMD s8 packed qkv
  M=256 is 274 us pipe_host
  card0, a loss vs oneDNN 164.
  Near N-linear 256. Numeric
  closed. One-card. Stop this
  tile vs packed qkv 164. Do
  not freeze 274 as 2800. Rank
  pipe_host. Do not promote.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ji - K7 ESIMD mixer conv-L2-register T=256 card0

CONTEXT -> conv-L2 SLM 358 lost
  to L2-once 327 and seq 298
  on per-t SLM barriers
  (2026-09-03jd). Register-head
  FIR+L2 on Q/K, no per-t SLM
  barriers, v channel-major,
  then packed delta. Napkin:
  L2 free in conv, beat 298.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_convl2r.
  gpu-run --card 0. T=256
  C=10240 nv=48 blk=16. spin=0.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_mixer_convl2r_t256.sh 0 0
  ```

RESULT -> cosine=1 max_abs=1.5e-5
  cosine_o=1 max_abs_o=9.8e-4
  ok=1. event 522.625 pipe_host
  530.777. timed act 2700-2600
  cur=2800 throttle=0. vs
  L2-once 327 (~1.62x) vs seq
  298 (~1.78x) vs conv-L2 SLM
  358 (~1.48x) vs mixer 471
  (~1.13x).

VERDICT -> ESIMD mixer conv-L2r
  T=256 is 531 us pipe_host
  card0, a loss vs L2-once 327
  and seq 298. Worse than
  conv-L2 SLM 358 and mixer
  471. Register-head FIR is
  the tax, not the extra
  launch. Stop this mapping.
  One-card is enough. Sequential
  conv+slmht ~298 stays the
  T=256 leftover. Do not freeze
  531 as 2800. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jj - K7 ESIMD s8 o-proj M=1 card1

CONTEXT -> oneDNN o-proj W8A8
  M=1 is 46-47 us. Square s8
  scale-to-f16 is 34 us at
  N=K=5120. Same dpas_s8_sc
  tile at n=5120 k=6144.
  Napkin K-linear 34*(6144/5120)
  ~41 us.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=5120 k=6144. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s8_oproj_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 61.719
  pipe_host 62.285. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 46-47
  (~1.33x, a loss) vs square
  s8 34 (~1.83x, K=1.2x napkin
  41).

VERDICT -> ESIMD o-proj s8
  M=1 is 62.285 us pipe_host
  card1 at 2800, a loss vs
  oneDNN o-proj W8A8 47.
  Worse than K-linear 41.
  Numeric closed. One-card.
  Stop this tile vs 47. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jk - K7 ESIMD s8 q-proj M=1 card0

CONTEXT -> oneDNN q-proj W8A8
  M=1 is 45-58 us. Square s8
  scale-to-f16 is 34 us at
  N=K=5120. Packed qkv s8 M=1
  is 74 us at n=10240. Same
  dpas_s8_sc tile at n=2048
  k=5120. Napkin N-linear
  34*(2048/5120)~14 us.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 0. NT=2 U=16
  m=1 n=2048 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s8_q_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 27.234
  pipe_host 27.714. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 45-58
  (~0.61x of 45, a beat) vs
  packed qkv s8 74 (~0.375x)
  vs square s8 34 (~0.82x) vs
  napkin 14 (~2.0x).

VERDICT -> ESIMD q-proj s8
  M=1 is 27.714 us pipe_host
  card0 at 2800, a beat vs
  oneDNN q-proj W8A8 45-58.
  Worse than N-linear 14.
  Numeric closed. One-card.
  Sibling before promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jl - K7 ESIMD s8 v-proj M=1 card1

CONTEXT -> oneDNN v-proj W8A8
  M=1 is 46 us. Square s8
  scale-to-f16 is 34 us at
  N=K=5120. Same dpas_s8_sc
  tile at n=6144 k=5120.
  Napkin N-linear 34*(6144/5120)
  ~41 us. Packed qkv s8 74.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=6144 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s8_v_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 52.609
  pipe_host 53.226. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 46
  (~1.16x, a loss) vs packed
  qkv s8 74 (~0.72x) vs square
  s8 34 (~1.57x, N=1.2x napkin
  41).

VERDICT -> ESIMD v-proj s8
  M=1 is 53.226 us pipe_host
  card1 at 2800, a loss vs
  oneDNN v-proj W8A8 46.
  Worse than N-linear 41.
  Numeric closed. One-card.
  Stop this tile vs 46. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jm - K7 ESIMD s8 o-proj NT=4 M=1 card0

CONTEXT -> oneDNN o-proj W8A8
  M=1 is 46-47 us. NT=2 same
  dpas_s8_sc tile is 62.285 us
  (loss). Square s8 34 at
  N=K=5120. Napkin K-linear
  34*(6144/5120) ~41 us. NT=4
  unroll=8 innerK=512.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 0. NT=4 U=8
  m=1 n=5120 k=6144. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s8_oproj_nt4_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 102.187
  pipe_host 103.086. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 47
  (~2.19x, a loss) vs NT=2 62
  (~1.66x) vs square s8 34
  (~3.03x, K=1.2x napkin 41).

VERDICT -> ESIMD o-proj s8
  NT=4 M=1 is 103.086 us
  pipe_host card0 at 2800, a
  loss vs oneDNN o-proj W8A8
  47 and vs NT=2 62. Worse
  than K-linear 41. Numeric
  closed. One-card. Stop this
  steal vs 47. Do not sibling.
  Do not promote. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jn - K7 ESIMD s8 q-proj M=1 sibling card1

CONTEXT -> card0 q-proj s8
  M=1 was 27.714 us pipe_host
  at 2800 (2026-09-03jk).
  oneDNN q W8A8 45-58. Same
  tile as k-proj (n=2048).
  packed qkv s8 74. Sibling
  hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=2048 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s8_q_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 27.258
  pipe_host 27.666. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs card0 27.714.
  Spread ~0.2%. vs W8A8 45-58
  (~0.61x of 45, a beat).
  Split vs packed leftover:
  q 27.7, k same shape as q
  so 27.7, v 53.2 (jl), sum
  ~108.6 vs packed qkv s8 74.

VERDICT -> Sibling matches.
  q/k s8 n=2048 is 28-class
  us pipe_host both cards at
  2800. Beats oneDNN q. Packed
  74 still beats sequential
  q+k+v ~109. Numeric closed.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jo - K7 ESIMD s8 packed qkv NT=4 M=1 card0

CONTEXT -> NT=2 packed qkv s8
  M=1 n=10240 k=5120 is 74 us
  at 2800 (beats W8A8 96).
  NT=4 o-proj was 103 (loss);
  this N is 2x wider. Same
  dpas_s8_sc tile NT=4 U=8.
  n=10240 % 64 == 0.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s8_sc.
  gpu-run --card 0. NT=4 U=8
  m=1 n=10240 k=5120. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s8_qkv_nt4_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 105.193
  pipe_host 105.746. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs NT=2 74
  (~1.43x, a loss) vs W8A8 96
  (~1.10x, a loss). NT=4
  o-proj was 103.

VERDICT -> ESIMD packed qkv s8
  NT=4 M=1 is 105.746 us
  pipe_host card0 at 2800, a
  loss vs NT=2 74 and vs W8A8
  96. Numeric closed. One-card.
  Stop this steal vs 74. Do
  not sibling. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jp - K7 ESIMD s4 packed qkv M=1 card1

CONTEXT -> A=s4 square scale-to-f16
  is 16.5 us. A=s4 packed qkv
  napkin N-linear 16.5*2~33 us.
  A=s4 ranks vs packed s8 74
  and W8A8 96 as wall time
  only, not those contracts.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s4 [-8,7] pack=2 scales
  0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s4_qkv_m1.sh 1 4000
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  16.250 pipe_host 16.637.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs napkin 33 (~0.50x,
  occupancy not N-linear).
  A=s4 vs packed s8 74 (~0.23x
  wall). A=s4 vs W8A8 96
  (~0.17x wall, not a W8A8
  beat). A=s4 vs square 16.5
  (wash).

VERDICT -> A=s4 packed qkv M=1
  is 16.637 us pipe_host card1
  at 2800. A=s4 wash vs square
  16.5, not napkin 33. A=s4
  wall beats packed s8 74.
  A=s4 is not a W8A8-contract
  beat of 96. A=s4 numeric
  closed. A=s4 one-card.
  Sibling before FINDINGS.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jq - K7 ESIMD s4 packed qkv M=1 sibling card0

CONTEXT -> A=s4 packed qkv M=1
  card1 was 16.637 us pipe_host
  at act=cur=2800 (2026-09-03jp).
  A=s4 cosine=1 max_abs=0. A=s4
  wash vs square 16.5, occupancy
  not N-linear. Sibling hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_sc.
  gpu-run --card 0. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s4 [-8,7] pack=2 scales
  0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s4_qkv_m1.sh 0 4000
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  16.229 pipe_host 16.607.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs card1 16.637. Spread
  ~0.2%. A=s4 vs napkin 33
  (~0.50x, occupancy not
  N-linear). A=s4 vs packed s8
  74 (~0.22x wall). A=s4 vs
  W8A8 96 (~0.17x wall, not a
  W8A8 beat). A=s4 vs square
  16.5 (wash).

VERDICT -> Sibling matches.
  A=s4 packed qkv M=1 is
  17-class us pipe_host both
  cards at 2800. A=s4 wash vs
  square 16.5. Occupancy not
  N-linear. A=s4 is not a
  W8A8-contract beat. vs packed
  s8 74 is wall-time only.
  A=s4 numeric closed. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jr - K7 ESIMD s4 packed qkv M=64 card1

CONTEXT -> A=s4 square 4x8 A-db
  M=64 is 33.6 us. A=s4 packed
  qkv napkin N-linear 33.6*2~67
  us. W8A8 packed M=64 is 140.
  s8 4x8 packed was 214 (loss).
  A=s4 ranks vs 67/140/214 as
  wall time only, not those
  contracts.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_db48.
  gpu-run --card 1. NT=2 U=16
  wg=4x8 A-db m=64 n=10240
  k=5120. spin=512. pack=2
  scales 0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s4_qkv_m64.sh 1 512
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  63.130 pipe_host 63.452.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs napkin 67 (~0.95x,
  near N-linear). A=s4 vs
  W8A8 140 (~0.45x wall, not
  a W8A8 beat). A=s4 vs s8
  214 (~0.30x wall). A=s4 vs
  square 33.6 (~1.89x).

VERDICT -> A=s4 packed qkv M=64
  is 63.452 us pipe_host card1
  at 2800. A=s4 near napkin 67
  not occupancy-bound. A=s4
  wall beats W8A8 140 and s8
  214. A=s4 is not a
  W8A8-contract beat of 140.
  A=s4 numeric closed. A=s4
  one-card. New s4 packed-qkv
  M=64 floor. Sibling before
  FINDINGS. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03js - K7 ESIMD s4 packed qkv M=64 sibling card0

CONTEXT -> A=s4 packed qkv M=64
  card1 was 63.452 us pipe_host
  at act=cur=2800 (2026-09-03jr).
  A=s4 cosine=1 max_abs=0. A=s4
  near napkin 67, near N-linear
  vs square 33.6. Sibling hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_db48.
  gpu-run --card 0. NT=2 U=16
  wg=4x8 A-db m=64 n=10240
  k=5120. spin=512. pack=2
  scales 0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s4_qkv_m64.sh 0 512
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  62.750 pipe_host 63.290.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs card1 63.452. Spread
  ~0.3%. A=s4 vs napkin 67
  (~0.94x, near N-linear).
  A=s4 vs W8A8 140 (~0.45x
  wall, not a W8A8 beat).
  A=s4 vs s8 214 (~0.30x
  wall). A=s4 vs square 33.6
  (~1.88x).

VERDICT -> Sibling matches.
  A=s4 packed qkv M=64 is
  63-class us pipe_host both
  cards at 2800. A=s4 near
  N-linear vs square 33.6.
  A=s4 is not a W8A8-contract
  beat. vs W8A8 140 is
  wall-time only. A=s4 numeric
  closed. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jt - K7 ESIMD s4 packed qkv M=256 card1

CONTEXT -> A=s4 square 4-acc
  4x8 M=256 is 48.6 us. A=s4
  packed qkv napkin N-linear
  48.6*2~97 us. W8A8 packed
  M=256 is 164. s8 4-acc packed
  was 274 (loss). A=s4 ranks
  vs 97/164/274 as wall time
  only, not those contracts.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_w48m4.
  gpu-run --card 1. NT=2 U=8
  wg=4x8 4-acc m=256 n=10240
  k=5120. spin=512. pack=2
  scales 0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s4_qkv_m256.sh 1 512
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  95.198 pipe_host 95.262.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs napkin 97 (~0.98x,
  near N-linear). A=s4 vs
  W8A8 164 (~0.58x wall, not
  a W8A8 beat). A=s4 vs s8
  274 (~0.35x wall). A=s4 vs
  square 48.6 (~1.96x).

VERDICT -> A=s4 packed qkv M=256
  is 95.262 us pipe_host card1
  at 2800. A=s4 near napkin 97
  not occupancy-bound. A=s4
  wall beats W8A8 164 and s8
  274. A=s4 is not a
  W8A8-contract beat of 164.
  A=s4 numeric closed. A=s4
  one-card. New s4 packed-qkv
  M=256 floor. Sibling before
  FINDINGS. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jv - K7 ESIMD s4 o-proj M=1 card1

CONTEXT -> A=s4 square scale-to-f16
  is 16.5 us. A=s4 o-proj napkin
  K-linear 16.5*(6144/5120)~20 us.
  s8 o-proj is 62 vs W8A8 47 (s8
  lost). A=s4 ranks vs 20/62/47
  as wall time only, not those
  contracts.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=5120 k=6144. spin=4000.
  Fill s4 [-8,7] pack=2 scales
  0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s4_oproj_m1.sh 1 4000
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  19.039 pipe_host 19.394.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs napkin 20 (~0.97x,
  near K-linear). A=s4 vs s8
  62 (~0.31x wall). A=s4 vs
  W8A8 47 (~0.41x wall, not a
  W8A8 beat). A=s4 vs square
  16.5 (~1.18x, K=1.2x).

VERDICT -> A=s4 o-proj M=1 is
  19.394 us pipe_host card1 at
  2800. A=s4 near napkin 20
  not occupancy-bound. A=s4
  wall beats s8 62 and W8A8 47.
  A=s4 is not a W8A8-contract
  beat of 47. A=s4 numeric
  closed. A=s4 one-card. New
  s4 o-proj floor. Sibling
  before FINDINGS. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ju - K7 ESIMD s4 packed qkv M=256 sibling card0

CONTEXT -> A=s4 packed qkv M=256
  card1 was 95.262 us pipe_host
  at act=cur=2800 (2026-09-03jt).
  A=s4 cosine=1 max_abs=0. A=s4
  near napkin 97, near N-linear
  vs square 48.6. Sibling hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_w48m4.
  gpu-run --card 0. NT=2 U=8
  wg=4x8 4-acc m=256 n=10240
  k=5120. spin=512. pack=2
  scales 0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s4_qkv_m256.sh 0 512
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  93.984 pipe_host 93.706.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs card1 95.262. Spread
  ~1.7%. A=s4 vs napkin 97
  (~0.97x, near N-linear).
  A=s4 vs W8A8 164 (~0.57x
  wall, not a W8A8 beat).
  A=s4 vs s8 274 (~0.34x
  wall). A=s4 vs square 48.6
  (~1.93x).

VERDICT -> Sibling matches.
  A=s4 packed qkv M=256 is
  95-class us pipe_host both
  cards at 2800. A=s4 near
  N-linear vs square 48.6.
  A=s4 is not a W8A8-contract
  beat. vs W8A8 164 is
  wall-time only. A=s4 numeric
  closed. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jw - K7 ESIMD s4 o-proj M=1 sibling card0

CONTEXT -> A=s4 o-proj M=1
  card1 was 19.394 us pipe_host
  at act=cur=2800 (2026-09-03jv).
  A=s4 cosine=1 max_abs=0. A=s4
  near napkin 20, near K-linear
  vs square 16.5. Sibling hold.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s4_sc.
  gpu-run --card 0. NT=2 U=16
  m=1 n=5120 k=6144. spin=4000.
  Fill s4 [-8,7] pack=2 scales
  0.02 out f16. A=s4.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s4_oproj_m1.sh 0 4000
  ```

RESULT -> A=s4 cosine=1.000000
  max_abs=0 ok=1. A=s4 event
  19.005 pipe_host 19.381.
  A=s4 timed act=cur=2800
  throttle=0. A=s4 spin_done
  act=cur=2800 throttle=0.
  A=s4 vs card1 19.394. Spread
  ~0.07%. A=s4 vs napkin 20
  (~0.97x, near K-linear).
  A=s4 vs s8 62 (~0.31x wall).
  A=s4 vs W8A8 47 (~0.41x
  wall, not a W8A8 beat).
  A=s4 vs square 16.5 (~1.17x,
  K=1.2x).

VERDICT -> Sibling matches.
  A=s4 o-proj M=1 is 19-class
  us pipe_host both cards at
  2800. A=s4 near K-linear vs
  square 16.5. A=s4 is not a
  W8A8-contract beat. vs W8A8
  47 is wall-time only. A=s4
  numeric closed. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jx - K7 ESIMD s2 packed qkv M=1 card1

CONTEXT -> A=s2 square scale-to-f16
  is 11.5 us. s4 packed 16.6
  wash vs square (occupancy).
  Napkin occupancy ~11.5. A=s2
  ranks vs square 11.5, s4
  packed 16.6, packed s8 74 as
  wall time only, not those
  contracts. IGC s2 range
  [-2,1]. A=s2 not W8A8.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s2_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s2 [-2,1] pack=4 scales
  0.02 out f16. A=s2.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s2_qkv_m1.sh 1 4000
  ```

RESULT -> A=s2 cosine=1.000000
  max_abs=0 ok=1. A=s2 event
  11.276 pipe_host 11.675.
  A=s2 timed act=cur=2800
  throttle=0. A=s2 spin_done
  act=cur=2800 throttle=0.
  A=s2 vs square 11.5 (wash,
  occupancy not N-linear).
  A=s2 vs s4 packed 16.6
  (~0.70x wall). A=s2 vs packed
  s8 74 (~0.16x wall). A=s2 vs
  napkin 11.5 (wash).

VERDICT -> A=s2 packed qkv M=1
  is 11.675 us pipe_host card1
  at 2800. A=s2 wash vs square
  11.5, occupancy not N-linear.
  A=s2 wall beats s4 packed
  16.6 and packed s8 74. A=s2
  is not a W8A8-contract beat.
  A=s2 numeric closed. A=s2
  one-card. Sibling before
  FINDINGS. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jy - K7 ESIMD s2 packed qkv M=1 sibling card0

CONTEXT -> A=s2 packed qkv M=1
  card1 is 11.675 us pipe_host
  at 2800 (2026-09-03jx),
  cosine=1 max_abs=0. Wash vs
  square 11.5. Occupancy not
  N-linear. IGC s2 range
  [-2,1]. A=s2 not W8A8.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s2_sc.
  gpu-run --card 0. NT=2 U=16
  m=1 n=10240 k=5120. spin=4000.
  Fill s2 [-2,1] pack=4 scales
  0.02 out f16. A=s2.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s2_qkv_m1.sh 0 4000
  ```

RESULT -> A=s2 cosine=1.000000
  max_abs=0 ok=1. A=s2 event
  11.299 pipe_host 11.644 vs
  card1 11.675. Spread ~0.27%.
  A=s2 timed act=cur=2800
  throttle=0. A=s2 spin_done
  act=cur=2800 throttle=0.
  A=s2 vs square 11.5 (wash,
  occupancy not N-linear).
  A=s2 vs s4 packed 16.6
  (~0.70x wall). A=s2 vs packed
  s8 74 (~0.16x wall). A=s2 vs
  napkin 11.5 (wash).

VERDICT -> A=s2 packed qkv M=1
  is 12-class us pipe_host both
  cards at 2800. A=s2 wash vs
  square 11.5, occupancy not
  N-linear. A=s2 wall beats s4
  packed 16.6 and packed s8 74.
  A=s2 is not a W8A8-contract
  beat. A=s2 numeric closed.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03jz - K7 ESIMD s2 o-proj M=1 card1

CONTEXT -> A=s2 square scale-to-f16
  is 11.5 us. A=s2 o-proj napkin
  K-linear 11.5*(6144/5120)~14 us.
  s4 o-proj is 19 vs W8A8 47.
  A=s2 ranks vs 14/19/47 as
  wall time only, not those
  contracts. IGC s2 range
  [-2,1]. A=s2 not W8A8.

CONFIG -> backend sycl+l0,
  standalone AOT dpas_s2_sc.
  gpu-run --card 1. NT=2 U=16
  m=1 n=5120 k=6144. spin=4000.
  Fill s2 [-2,1] pack=4 scales
  0.02 out f16. A=s2.

COMMAND ->
  ```
  gpu-run --card 1 kernels/gdn/run_esimd_s2_oproj_m1.sh 1 4000
  ```

RESULT -> A=s2 cosine=1.000000
  max_abs=0 ok=1. A=s2 event
  13.159 pipe_host 13.545.
  A=s2 timed act=cur=2800
  throttle=0. A=s2 spin_done
  act=cur=2800 throttle=0.
  A=s2 vs napkin 14 (~0.97x,
  near K-linear). A=s2 vs s4
  19 (~0.71x wall). A=s2 vs
  W8A8 47 (~0.29x wall, not a
  W8A8 beat). A=s2 vs square
  11.5 (~1.18x, K=1.2x).

VERDICT -> A=s2 o-proj M=1 is
  13.545 us pipe_host card1 at
  2800. A=s2 near napkin 14
  not occupancy-bound. A=s2
  wall beats s4 19 and W8A8
  47. A=s2 is not a
  W8A8-contract beat of 47.
  A=s2 numeric closed. A=s2
  one-card. New s2 o-proj
  floor. Sibling before
  FINDINGS. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-03ka - K7 ESIMD s2 o-proj M=1 sibling card0

CONTEXT -> card1 A=s2 o-proj was
  13.545 us at 2800
  (2026-09-03jz). First floor.
  Sibling. spin=4000. Same TU.
  A=s2 not W8A8.

CONFIG -> backend sycl+l0, same
  AOT dpas_s2_sc. gpu-run
  --card 0. NT=2 U=16 m=1
  n=5120 k=6144. spin=4000.
  A=s2.

COMMAND ->
  ```
  gpu-run --card 0 kernels/gdn/run_esimd_s2_oproj_m1.sh 0 4000
  ```

RESULT -> A=s2 cosine=1 max_abs=0
  ok=1. A=s2 event 13.167
  pipe_host 13.519 vs card1
  13.545. Spread ~0.2%. A=s2
  timed act=cur=2800 throttle=0.
  A=s2 spin_done act=cur=2800
  throttle=0.

VERDICT -> Sibling matches.
  A=s2 o-proj M=1 is 14-class
  us pipe_host both cards at
  2800. Near K-linear vs square
  11.5. Not a W8A8-contract
  beat. vs W8A8 47 is wall-time
  only. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04a - K7 ESIMD s8 o-proj NT=1 M=1 card0

CONTEXT -> oneDNN o-proj W8A8
  M=1 is 46-47 us. sc NT=2
  62 us. NT=4 103 us. Square
  s8 34. Napkin K-linear ~41.
  NT=1 is 20 WGs vs NT=2 10
  WGs. Leftover GEMM, not
  fused mixer 7.1.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1. gpu-run
  --card 0. NT=1 U=16 m=1
  n=5120 k=6144. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_oproj_nt1_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 54.443
  pipe_host 55.016. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 47
  (~1.17x, a loss) vs NT=2 62
  (~0.89x) vs NT=4 103
  (~0.53x) vs square s8 34
  (~1.62x, K=1.2x napkin 41).

VERDICT -> ESIMD o-proj s8
  NT=1 M=1 is 55.016 us
  pipe_host card0 at 2800.
  Occupancy steal vs NT=2 62
  (20 WGs vs 10). Same s8
  scale-to-f16 contract as
  W8A8 47, not faster, not a
  W8A8-contract beat. Worse
  than K-linear 41. Numeric
  closed. One-card. Leftover
  GEMM. Do not replace fused
  mixer 7.1. Sibling later
  before FINDINGS. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04b - K7 ESIMD s8 o-proj B-pipeline hail-mary persist/mainloop M=1 card1

CONTEXT -> New s8 B-software-
  pipeline leftover steal
  (hail-mary persist/mainloop)
  vs sc NT=2. At M=1 B is the
  traffic.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_bp, hail-mary
  persist/mainloop. gpu-run
  --card 1. NT=2 m=1 n=5120
  k=6144. spin=4000. Fill s8
  [-64,64] scales 0.02 out
  f16. Priors: W8A8 o-proj
  46-47 us; sc NT=2 62 us;
  NT=4 103 us; square s8 34;
  napkin K-linear ~41. At M=1
  B is the traffic.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_oproj_bp_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 66.250
  pipe_host 66.892. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0.

VERDICT -> hail-mary
  persist/mainloop. ESIMD
  o-proj s8 B-pipeline M=1
  is 66.892 us pipe_host
  card1 at 2800, a loss vs
  sc NT=2 62 and vs oneDNN
  o-proj W8A8 47. Not a
  W8A8-contract beat of 47.
  Worse than K-linear 41.
  Numeric closed. One-card.
  STOP this B-pipeline on
  decode o-proj. Do not
  sibling. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04c - K7 ESIMD s8 o-proj NT=1 M=1 sibling card1

CONTEXT -> card0 NT=1 o-proj was
  55.016 us at 2800
  (2026-09-04a). First floor.
  Sibling. spin=4000. Same TU
  dpas_s8_sc_nt1.

CONFIG -> backend sycl+l0, same
  AOT dpas_s8_sc_nt1. gpu-run
  --card 1. NT=1 U=16 m=1
  n=5120 k=6144. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_oproj_nt1_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 54.784
  pipe_host 55.323 vs card0
  55.016. Spread ~0.56%. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 47
  (~1.18x, a loss) vs NT=2 62
  (~0.89x).

VERDICT -> Sibling matches.
  ESIMD o-proj s8 NT=1 M=1 is
  55-class us pipe_host both
  cards at 2800. Occupancy
  steal vs sc NT=2 62. Same
  s8 scale-to-f16 contract as
  W8A8 47, not faster, not a
  W8A8-contract beat. Not a
  leftover close. Worse than
  K-linear 41. Numeric closed.
  Leftover GEMM. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04d - K7 ESIMD s8 4-acc wg 8x4 k128 packed qkv M=64 leftover steal card0

CONTEXT -> NEW s8 4-acc wg 8x4
  k128 leftover steal on packed
  qkv M=64 n=10240 k=5120.
  Priors: W8A8 packed qkv M=64
  is 138-142 us. 4x8 A-db 214
  lost. square 4-acc 75.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w84m4. gpu-run
  --card 0. NT=2 m=64 n=10240
  k=5120. wg=8x4 4acc k128.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_w84m4_m64.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 153.906
  pipe_host 154.074. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs W8A8 138-142
  (~1.10x, a loss) vs 4x8 A-db
  214 (~0.72x) vs square 4-acc
  75 (~2.05x).

VERDICT -> ESIMD packed qkv s8
  4-acc wg 8x4 k128 M=64 is
  154.074 us pipe_host card0
  at 2800, a loss vs oneDNN
  W8A8 140. Not a W8A8-contract
  beat. Faster than 4x8 A-db
  214, still ~2.05x square
  4-acc 75. Numeric closed.
  One-card. STOP this wg 8x4
  at M=64. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04e - K7 ESIMD s8 4-acc wg 8x4 k128 packed qkv M=256 leftover steal card0

CONTEXT -> NEW s8 4-acc wg 8x4
  k128 leftover steal on packed
  qkv M=256 n=10240 k=5120.
  Priors: W8A8 packed qkv M=256
  is 164 us. 4-acc 4x8 274 lost.
  square 4-acc 128. Same-family
  M=64 was 154 vs W8A8 140
  (2026-09-04d). STOP at M=64
  already; this is a different
  leftover.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w84m4. gpu-run
  --card 0. NT=2 m=256 n=10240
  k=5120. wg=8x4 4acc k128.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_w84m4_m256.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 281.568
  pipe_host 278.725. timed
  act=2717 cur=2800 throttle=1.
  spin_done act=2717 cur=2800
  throttle=1. vs W8A8 164
  (~1.70x, a loss) vs 4-acc
  4x8 274 (~1.02x) vs square
  4-acc 128 (~2.18x). gpu-run
  119s.

VERDICT -> ESIMD packed qkv s8
  4-acc wg 8x4 k128 M=256 is
  278.725 us pipe_host card0
  at act=2717 cur=2800
  throttle=1, a loss vs oneDNN
  W8A8 164. Not a W8A8-contract
  beat. Same class as 4-acc
  4x8 274, still ~2.18x square
  4-acc 128. Numeric closed.
  One-card. Do not freeze (not
  2800). STOP this wg 8x4 at
  M=256. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04f - K7 ESIMD s8 4-acc B-pipeline hail-mary persist/mainloop packed qkv M=256 card1

CONTEXT -> hail-mary
  persist/mainloop 4-acc
  B-pipeline packed-qkv M=256
  n=10240 k=5120. Priors:
  W8A8 packed 164. 4-acc 4x8
  274. Decode B-pipeline
  o-proj LOST (66.9 vs 62).
  This is prefill 4-acc, still
  label hail-mary.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4bp
  hail-mary persist/mainloop.
  gpu-run --card 1. NT=2 m=256
  n=10240 k=5120 wg=4x8 4acc
  B-pipeline spin=512. Fill s8
  [-64,64] scales 0.02 out
  f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_qkv_w48m4bp_m256.sh 1 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 344.135
  pipe_host 343.995. timed
  act=2700 cur=2800 throttle=1.
  spin_done act=2700 cur=2800
  throttle=1. vs W8A8 164
  (~2.10x, a loss) vs 4-acc
  4x8 274 (~1.25x, a loss).
  gpu-run 123s.

VERDICT -> hail-mary
  persist/mainloop. ESIMD
  packed qkv s8 4-acc
  B-pipeline M=256 is 343.995
  us pipe_host card1, a loss
  vs 4-acc 4x8 274 and vs
  oneDNN W8A8 164. Not a
  W8A8-contract beat. Numeric
  closed. One-card. Clocks not
  2800 (throttle=1). Do not
  freeze. STOP this persist
  B-pipeline on packed
  prefill. Do not sibling. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04g - K7 ESIMD s8 o-proj NT=1 wg 4x2 M=1 card0

CONTEXT -> NEW s8 NT=1 wg 4x2
  o-proj (40 WGs vs NT=1 8x2
  20 WGs). NT=1 wg 8x2 is
  55.016/55.323 both at 2800
  vs W8A8 47 vs sc 62.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1w42. NT=1
  wg=4x2 m=1 n=5120 k=6144
  spin=4000. gpu-run --card 0.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_oproj_nt1w42_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 74.010
  pipe_host 74.278. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs NT=1 8x2 55
  (~1.35x, a loss) vs W8A8 47
  (~1.58x, a loss) vs sc NT=2
  62 (~1.20x, a loss). gpu-run
  2s.

VERDICT -> ESIMD o-proj s8
  NT=1 wg 4x2 M=1 is 74.278 us
  pipe_host card0 at 2800, a
  loss vs NT=1 wg 8x2 55.
  Occupancy steal 40 WGs vs 20
  lost. Not a W8A8-contract
  beat of 47. Numeric closed.
  One-card. STOP smaller WG on
  o-proj. Do not sibling. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04h - K7 ESIMD s8 o-proj NT=1 wg 4x2 M=1 card1

CONTEXT -> NEW s8 NT=1 wg 4x2
  o-proj leftover steal vs NT=1
  8x2 55 both. 40 WGs vs 20.
  Both-card new WG map. card0
  74.278 at 2800 (2026-09-04g).

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1w42. gpu-run
  --card 1. NT=1 wg=4x2 m=1
  n=5120 k=6144. spin=4000.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_oproj_nt1w42_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 72.901
  pipe_host 74.636 vs card0
  74.278. Spread ~0.48%. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs NT=1 8x2 55
  (~1.36x, a loss) vs W8A8 47
  (~1.59x, a loss) vs sc NT=2
  62 (~1.20x, a loss). gpu-run
  2s.

VERDICT -> Sibling matches the
  loss. ESIMD o-proj s8 NT=1
  wg 4x2 M=1 is 74.636 us
  pipe_host card1 at 2800, a
  loss vs NT=1 8x2 55.
  Occupancy steal 40 WGs vs 20
  lost. Not a W8A8-contract
  beat of 47. Numeric closed.
  Both cards 74-class at 2800.
  STOP this wg 4x2. Do not
  promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04i - P2 synthetic XCCL P2P-off us sweep both cards

CONTEXT -> Parked charter P2. Kernel leftover
  GEMM still open (o-proj NT=1 55 vs
  W8A8 47). Pause one-card. P2P off.
  Pre-health green. Decode through
  64-token payloads, then hang.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric xccl, p2p=0.
  Image b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch 2.13.0+xpu.
  Payloads decode_h 5120 bf16 through
  8MiB. No CCL_TOPO_P2P_ACCESS=1.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251
  gpu-run xpu-collective-health --p2p 0 --timeout 240
  gpu-run bash parallel/tp2/run_xccl_p2p0.sh
  # hang at prefill_256h; docker killed; post-health
  ```

RESULT -> Pre: card0 HEALTHY, card1
  HEALTHY, COLLECTIVE_HEALTH_OK
  world=2 shape=4x5120 p2p=0.
  Fire 1 (hang137): decode_h
  all_reduce 137.119 us ok=1,
  all_gather 210.358, sendrecv
  848.166. health_4h AR 125.922
  AG 172.312 SR 649.646.
  prefill_64h AR 535.338 AG
  544.275 SR 948.873. Then no
  RESULT for ~18 min at act=2800.
  docker rm unruffled_wing.
  Fire 2 (instrumented retry):
  decode_h AR 98.846 AG 127.938
  SR 538.775. prefill_64h AR
  562.496 AG 563.377 SR 890.072.
  prefill_256h all_reduce 2081.428
  ok=1, then hang on all_gather
  2.5 MiB. docker rm
  zealous_blackburn. Post-health
  after both kills: card0/card1
  HEALTHY, COLLECTIVE_HEALTH_OK
  4x5120 p2p=0.

VERDICT -> Decode-sized XCCL
  P2P-off all_reduce is ~99-137 us,
  all_gather ~128-210, sendrecv
  ping-pong ~539-848. 64-token
  AR ~535-562. 256-token AR can
  finish (~2081 us) but all_gather
  at 2.5 MiB hangs. STOP XCCL
  all_gather >=2.5 MiB P2P-off
  until a new arm with a timeout.
  Teardown recovered. Not a full
  P2 exit. P4 stays blocked.
  Rank us. Do not enable P2P.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04j - P3 host-staged PP=2 activation handoff both cards

CONTEXT -> Parked charter P3 after
  P2 hang+teardown health. Host
  bounce stage0 xpu:0 -> DRAM ->
  stage1 xpu:1. P2P off. Identity.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric host_staged_pp2,
  p2p=0. One process, both XPUs.
  gpu-run both cards. No peer
  access. No clock spin.

COMMAND ->
  ```
  gpu-run bash parallel/pp2/run_handoff_host.sh
  ```

RESULT -> ok_all=1. T1_H5120
  host_handoff 76.848 us,
  samecard 22.047, bubble 0.713,
  ok=1. T4 76.204 bubble 0.826.
  T64 303.123 bubble 0.961.
  T256 1026.181 bubble 0.986.
  T1_H6144 50.307 bubble 0.739.
  1MiB 467.278 bubble 0.971.
  Start clocks D3hot/2550, end
  2800. Do not freeze 77 as 2800.
  Post2 health after P2 kills
  still HEALTHY + COLLECTIVE_OK.

VERDICT -> Host-staged PP=2
  handoff is correct. Decode
  T=1 hidden 5120 is 77 us class
  with ~71% bubble vs same-card
  copy 22 us. Bubble dominates.
  Identity closed. First xe2x2
  PP=2 synthetic. Device-P2P
  handoff not measured (P2P off).
  P4 stays blocked on P2 bulk
  hang. Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04k - K7 ESIMD s8 o-proj NT=1 B-pipeline leftover steal M=1 card0

CONTEXT -> NEW s8 NT=1 B-pipeline
  o-proj leftover steal
  (prologue next k64 A/B into
  GRF before current dpas) vs
  NT=1 55 both vs W8A8 47.
  NT=2 B-pipeline 67 vs sc 62.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1bp. NT=1
  B-pipeline m=1 n=5120 k=6144
  spin=4000. gpu-run --card 0.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_oproj_nt1bp_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 59.672
  pipe_host 60.244. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs NT=1 55
  (~1.10x, a loss) vs W8A8 47
  (~1.28x, a loss) vs NT=2
  B-pipeline 67 (~0.90x) vs sc
  NT=2 62 (~0.97x). gpu-run
  3s.

VERDICT -> ESIMD o-proj s8
  NT=1 B-pipeline M=1 is
  60.244 us pipe_host card0
  at 2800, a loss vs NT=1 55.
  Occupancy plus B-pipe steal
  lost. Not a W8A8-contract
  beat of 47. Numeric closed.
  One-card. STOP NT=1
  B-pipeline on o-proj. Do not
  sibling. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04l - K7 ESIMD s8 4-acc split-K=2 packed qkv M=64 leftover steal card1

CONTEXT -> NEW s8 4-acc split-K=2
  leftover steal on packed qkv
  M=64 n=10240 k=5120. Priors:
  W8A8 packed M=64 is 138-142.
  4x8 A-db 214. wg 8x4 154.
  Square 4-acc 75. Rank
  pipe_host of gemm+reduce.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4sk.
  gpu-run --card 1. NT=2 m=64
  n=10240 k=5120. splitK=2
  wg=4x8 4acc k128 unroll=5.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_qkv_sk_m64.sh 1 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 7.766
  (reduce-only). pipe_host
  115.081. wait_host 130.993.
  timed act=2783 cur=2800
  throttle=1. spin_done
  act=2783 cur=2800 throttle=1.
  vs W8A8 138-142 (~0.82x, a
  win) vs 4x8 A-db 214 (~0.54x)
  vs wg 8x4 154 (~0.75x) vs
  square 4-acc 75 (~1.53x).
  gpu-run 30s.

VERDICT -> ESIMD packed qkv s8
  4-acc split-K=2 M=64 is
  115.081 us pipe_host card1
  at act=2783 cur=2800
  throttle=1, a win vs oneDNN
  W8A8 140. W8A8-contract beat
  of 140 on this card. Numeric
  closed. One-card. Do not
  freeze (not 2800). Still fire
  M=256 (win vs 140, within
  10% or better). Sibling
  before FINDINGS. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04m - K7 ESIMD s8 4-acc split-K=2 packed qkv M=64 sibling card0

CONTEXT -> card1 split-K=2 packed
  qkv M=64 was 115.081 us
  pipe_host at act=2783
  throttle=1 (2026-09-04l).
  First floor. Sibling. Same
  TU dpas_s8_sc8w48m4sk.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4sk.
  gpu-run --card 0. NT=2 m=64
  n=10240 k=5120. splitK=2
  wg=4x8 4acc k128 unroll=5.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_sk_m64.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 6.531
  (reduce-only). pipe_host
  117.264 vs card1 115.081.
  Spread ~1.90%. wait_host
  131.224. timed act=2783
  cur=2800 throttle=1.
  spin_done act=2783 cur=2800
  throttle=1. vs W8A8 138-142
  (~0.84x, a win) vs 4x8 A-db
  214 (~0.55x) vs wg 8x4 154
  (~0.76x) vs square 4-acc 75
  (~1.56x). gpu-run 34s.

VERDICT -> Sibling matches.
  ESIMD packed qkv s8 4-acc
  split-K=2 M=64 is 115-class
  us pipe_host both cards
  (117.264 / 115.081) at
  act=2783 cur=2800
  throttle=1, a win vs oneDNN
  W8A8 140. W8A8-contract beat
  of 140 both cards. Numeric
  closed. Spread ~1.90%. Do
  not freeze (not 2800). Not a
  decode leftover. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04n - K7 ESIMD s8 4-acc split-K=2 packed qkv M=256 leftover steal card1

CONTEXT -> NEW s8 4-acc split-K=2
  leftover steal on packed qkv
  M=256 n=10240 k=5120. Priors:
  W8A8 packed M=256 is 164 us.
  4-acc 4x8 274 lost. persist
  344 lost. Same-family M=64
  was 115 vs W8A8 140
  (2026-09-04l/m). Rank
  pipe_host of gemm+reduce.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4sk.
  gpu-run --card 1. NT=2 m=256
  n=10240 k=5120. splitK=2
  wg=4x8 4acc k128 unroll=5.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_qkv_sk_m256.sh 1 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 48.417
  (reduce-only). pipe_host
  294.564. wait_host 309.415.
  timed act=2650/2633 cur=2800
  throttle=1. spin_done
  act=2650 cur=2800 throttle=1.
  vs W8A8 164 (~1.80x, a
  loss) vs 4-acc 4x8 274
  (~1.07x, a loss) vs persist
  344 (~0.86x) vs square
  4-acc 128 (~2.30x). gpu-run
  127s.

VERDICT -> ESIMD packed qkv s8
  4-acc split-K=2 M=256 is
  294.564 us pipe_host card1
  at act=2650/2633 cur=2800
  throttle=1, a loss vs oneDNN
  W8A8 164. Not a W8A8-contract
  beat. Same class as 4-acc
  4x8 274. Numeric closed.
  One-card. Do not freeze (not
  2800). STOP split-K at
  M=256. Do not sibling. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04o - P2 host-staged all-reduce P2P-off both cards

CONTEXT -> Parked charter P2 leftover
  after XCCL P2P-off decode AR
  99-137 us and hang on 2.5 MiB
  all_gather (2026-09-04i).
  Host-staged AR: XCCL barrier
  only, payload add on host shm.
  P2P off. Pause one-card.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric host_staged_ar,
  p2p=0. Image
  b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch
  2.13.0+xpu. Outer timeout 180s.
  No CCL_TOPO_P2P_ACCESS=1.
  No clock spin.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run bash parallel/tp2/run_host_ar.sh
  # post xpu-health same as pre
  ```

RESULT -> Pre: card0 HEALTHY,
  card1 HEALTHY. ok_all=1.
  decode_h 438.944 us ok=1
  GBs=0.023. health_4h 470.356.
  prefill_64h 2765.903.
  prefill_256h 9493.851
  (2.5 MiB finished). 1MiB
  3983.281. vs XCCL decode AR
  99-137 (~3.2-4.4x slower).
  vs XCCL 256h AR 2081
  (~4.6x slower). gpu-run 19s.
  Timeout not hit. Start clocks
  D0 cur=1133/1333, end 2800
  throttle=0. Do not freeze
  439 as 2800. Post: card0
  HEALTHY, card1 HEALTHY.

VERDICT -> Host-staged P2P-off
  AR is correct through 2.5 MiB.
  Decode 439 us loses to XCCL
  AR 99-137. 256h 9494 us
  finishes where XCCL all_gather
  hung. Identity closed.
  Teardown health recovered.
  Not a full P2 exit. Does not
  unblock P4. Do not enable P2P.
  Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04p - K7 ESIMD s8 o-proj NT=1 lsc_prefetch_2d M=1 card0

CONTEXT -> NEW s8 NT=1 +
  lsc_prefetch_2d o-proj
  leftover steal vs NT=1 55
  both vs W8A8 47. ngen M=1
  has ff prefetch. Old 1D
  prefetch lost decode.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1ff.
  prefetch=lsc_prefetch_2d
  m=1 n=5120 k=6144 spin=4000.
  gpu-run --card 0. Fill s8
  [-64,64] scales 0.02 out
  f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_oproj_nt1ff_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 54.211
  pipe_host 54.797. timed
  act=cur=2800 throttle=0.
  spin_done act=cur=2800
  throttle=0. vs NT=1 55.016
  (~0.40% faster, 55-class,
  not a steal) vs W8A8 47
  (~1.17x, a loss). gpu-run
  2s.

VERDICT -> ESIMD o-proj s8
  NT=1 lsc_prefetch_2d M=1 is
  54.797 us pipe_host card0
  at 2800, 55-class vs NT=1
  55. Does not beat 55. Not a
  W8A8-contract beat of 47.
  Numeric closed. One-card.
  STOP prefetch on NT=1
  o-proj. Do not sibling. Do
  not promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04q - K7 ESIMD s8 4-acc split-K=5 unroll=8 packed qkv M=256 leftover steal card1

CONTEXT -> NEW s8 4-acc split-K=5
  unroll=8 leftover steal on
  packed qkv M=256 n=10240
  k=5120. Priors: W8A8 packed
  M=256 is 164 us. 4-acc 4x8
  274 lost. SK=2 unroll=5 295
  lost (2026-09-04n). M=64
  SK=2 115 beat 140. Rank
  pipe_host of gemm+reduce.
  Event is reduce-only.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4sk5.
  gpu-run --card 1. NT=2 m=256
  n=10240 k=5120. splitK=5
  wg=4x8 4acc k128 unroll=8.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_qkv_sk5_m256.sh 1 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 129.562
  (reduce-only). pipe_host
  392.811. wait_host 406.883.
  timed act=2650 cur=2800
  throttle=1. spin_done
  act=2650 cur=2800 throttle=1.
  vs W8A8 164 (~2.40x, a
  loss) vs 4-acc 4x8 274
  (~1.43x, a loss) vs SK=2
  295 (~1.33x, a loss).
  gpu-run 120s.

VERDICT -> ESIMD packed qkv s8
  4-acc split-K=5 unroll=8
  M=256 is 392.811 us
  pipe_host card1 at act=2650
  cur=2800 throttle=1, a loss
  vs oneDNN W8A8 164. Not a
  W8A8-contract beat. Worse
  than SK=2 295 and 4-acc
  274. Numeric closed.
  One-card. Do not freeze (not
  2800). STOP SK=5 at M=256.
  Do not sibling. Do not
  promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04r - K7 ESIMD s8 o-proj NT=1 split-K=2 M=1 card0

CONTEXT -> NEW s8 NT=1 split-K=2
  o-proj leftover steal vs
  NT=1 55 both vs W8A8 47.
  Packed SK=2 won M=64.
  Prefetch 54.8 wash vs 55.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1sk. NT=1
  splitK=2 m=1 n=5120 k=6144
  spin=4000. gpu-run --card 0.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host of
  gemm+reduce. Event is
  reduce-only.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_oproj_nt1sk_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 1.474
  (reduce-only). pipe_host
  44.348. wait_host 59.971.
  timed act=cur=2800
  throttle=0. spin_done
  act=cur=2800 throttle=0.
  vs NT=1 55.016 (~0.81x, a
  win) vs W8A8 47 (~0.94x, a
  win) vs prefetch 54.797
  (~0.81x). gpu-run 2s.

VERDICT -> ESIMD o-proj s8
  NT=1 split-K=2 M=1 is
  44.348 us pipe_host card0
  at 2800, a win vs NT=1 55
  and vs oneDNN W8A8 47.
  W8A8-contract beat of 47
  on this card. Numeric
  closed. One-card. Sibling
  before FINDINGS. Do not
  promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04s - K7 ESIMD s8 o-proj NT=1 split-K=2 M=1 sibling card1

CONTEXT -> card0 NT=1 split-K=2
  o-proj was 44.348 us
  pipe_host at 2800
  (2026-09-04r). First floor.
  Sibling. New SK mapping on
  decode. spin=4000. Same TU
  dpas_s8_sc_nt1sk.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc_nt1sk.
  gpu-run --card 1. NT=1 m=1
  n=5120 k=6144. splitK=2
  wg=8x2_alongN U=16.
  spin=4000. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host of gemm+reduce.
  Event is reduce-only.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_oproj_nt1sk_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 1.474
  (reduce-only). pipe_host
  44.114 vs card0 44.348.
  Spread ~0.53%. wait_host
  59.259. timed act=cur=2800
  throttle=0. spin_done
  act=cur=2800 throttle=0.
  vs NT=1 55.323 (~0.80x, a
  win) vs W8A8 47 (~0.94x, a
  win) vs prefetch 54.797
  (~0.80x). gpu-run 2s.

VERDICT -> Sibling matches.
  ESIMD o-proj s8 NT=1
  split-K=2 M=1 is 44-class
  us pipe_host both cards
  (44.348 / 44.114) at 2800.
  Occupancy steal vs NT=1 55
  (40 WGs along K vs 20).
  W8A8-contract beat of 47
  both cards. Numeric closed.
  Spread ~0.53%. Leftover
  GEMM. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04t - P2 XCCL all_gather 2.5 MiB timeout-bounded both cards

CONTEXT -> Parked charter P2 leftover
  after unbounded 2.5 MiB XCCL
  all_gather hang (2026-09-04i)
  and host-staged AR finish
  (2026-09-04o). Timeout arm.
  Hang within 45s is a RESULT.
  P2P off. Pause one-card.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric xccl, p2p=0
  timeout=45s. Image
  b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch
  2.13.0+xpu. Outer timeout 45s
  TERM then kill-after 8s.
  No CCL_TOPO_P2P_ACCESS=1.
  Payload prefill_256h 256*5120
  bf16 = 2621440 bytes.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run bash parallel/tp2/run_ag256.sh
  # post xpu-health same as pre
  ```

RESULT -> Pre: card0 HEALTHY,
  card1 HEALTHY. First identity
  all_gather: rank=1 ok=1.
  Rank 0 never printed OK.
  No RESULT us line. No
  sendrecv. TIMEOUT_OR_EXIT
  rc=124 at 45s. RESULT
  op=all_gather name=prefill_256h
  HANG timeout=45s p2p=0.
  gpu-run 49s. CCL_TOPO_P2P_ACCESS
  0. Topology: PCIe between
  devices. Post: card0 HEALTHY
  (24s), card1 HEALTHY (26s).
  Not WEDGED.

VERDICT -> Bounded hang is a
  RESULT. XCCL P2P-off all_gather
  at 2.5 MiB hangs within 45s.
  Teardown health recovered.
  Not a full P2 exit. P4 stays
  blocked. Do not enable P2P.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04v - K7 ESIMD s8 2-acc packed qkv M=64 leftover steal card1

CONTEXT -> NEW s8 2-acc wg 4x8
  k128 packed qkv M=64 n=10240
  k=5120. Priors: SK=2 115
  beats W8A8 140 (2026-09-04l).
  4x8 A-db 214. wg 8x4 154.
  Square 4-acc 75. Rank
  pipe_host.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m2.
  gpu-run --card 1. NT=2 m=64
  n=10240 k=5120. wg=4x8 2acc
  k128 unroll=8. spin=512.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_s8_qkv_m2_m64.sh 1 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 140.521
  pipe_host 141.647. wait_host
  157.001. timed act=cur=2800
  throttle=0. spin_done
  act=cur=2800 throttle=0.
  vs SK=2 115 (~1.23x, a
  loss) vs W8A8 140 (~1.01x, a
  loss) vs 4x8 A-db 214
  (~0.66x) vs wg 8x4 154
  (~0.92x) vs square 4-acc 75
  (~1.89x). gpu-run 32s.

VERDICT -> ESIMD packed qkv s8
  2-acc wg 4x8 k128 M=64 is
  141.647 us pipe_host card1
  at 2800, a loss vs SK=2 115
  and vs oneDNN W8A8 140. Not
  a W8A8-contract beat. Numeric
  closed. One-card. Held 2800.
  STOP 2-acc at M=64. Do not
  sibling. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04u - K7 ESIMD s8 2-acc wg 4x8 k128 packed qkv M=256 leftover steal card0

CONTEXT -> NEW s8 2-acc wg 4x8
  k128 packed qkv M=256 n=10240
  k=5120. Priors: W8A8 packed
  M=256 is 164 us. 4-acc 4x8
  274 lost. SK=2 295 lost
  (2026-09-04n). SK=5 393
  lost (2026-09-04q). Same-
  family M=64 2-acc 142 lost
  vs W8A8 140 (2026-09-04v).
  2-acc = 2x M WGs vs 4-acc.
  Rank pipe_host.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m2.
  gpu-run --card 0. NT=2 m=256
  n=10240 k=5120. wg=4x8 2acc
  k128 unroll=8. spin=512.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_m2_m256.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 325.896
  pipe_host 327.053. wait_host
  341.648. timed act=2667
  cur=2800 throttle=1.
  spin_done act=2667 cur=2800
  throttle=1. vs W8A8 164
  (~1.99x, a loss) vs 4-acc
  4x8 274 (~1.19x, a loss)
  vs SK=2 295 (~1.11x, a
  loss) vs SK=5 393 (~0.83x).
  gpu-run 120s.

VERDICT -> ESIMD packed qkv s8
  2-acc wg 4x8 k128 M=256 is
  327.053 us pipe_host card0
  at act=2667 cur=2800
  throttle=1, a loss vs oneDNN
  W8A8 164. Not a W8A8-contract
  beat. Worse than 4-acc 274
  and SK=2 295. Numeric closed.
  One-card. Do not freeze (not
  2800). STOP 2-acc at M=256.
  Do not sibling. Do not
  promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04w - P2 chunked XCCL all_gather 4x64h both cards

CONTEXT -> Parked charter P2 leftover
  after one-shot 2.5 MiB XCCL
  all_gather hang (2026-09-04i/t)
  and host-staged AR finish
  (2026-09-04o). 64h already
  ~544 us. 4 sequential 64h
  gathers, P2P off. Timeout
  90s. Pause one-card.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric xccl, p2p=0
  chunked 4x64h. Image
  b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch
  2.13.0+xpu. Outer timeout 90s
  TERM then kill-after 10s.
  No CCL_TOPO_P2P_ACCESS=1.
  Payload prefill_256h as 4x
  64*5120 bf16 = 2621440 bytes.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run bash parallel/tp2/run_ag_chunk64.sh
  # post xpu-health same as pre
  ```

RESULT -> Pre: card0 HEALTHY
  (25s), card1 HEALTHY (25s).
  Identity: rank=0 ok=1,
  rank=1 ok=1. RESULT
  op=chunked_all_gather
  name=prefill_256h_as_4x64h
  chunk_numel=327680 nchunk=4
  bytes=2621440 us=2161.738
  ok=1. VERDICT_LINE ok_all=1
  path=chunked_ag_64h p2p=0.
  TIMEOUT_OR_EXIT rc=0.
  gpu-run 20s. Timeout not hit.
  CCL_TOPO_P2P_ACCESS 0.
  Topology: PCIe between
  devices. vs one-shot hang
  (04t rc=124 at 45s). vs 4x
  64h AG ~544 us (~2176 us
  linear). vs host-staged AR
  256h 9494 us. Post: card0
  HEALTHY (21s), card1 HEALTHY
  (21s). Not WEDGED.

VERDICT -> Chunked XCCL P2P-off
  all_gather is a passing bulk
  path. 2161.738 us, ok_all=1.
  One-shot 2.5 MiB still hangs;
  STOP one-shot AG >=2.5 MiB.
  Teardown health recovered.
  P4 may unblock on this
  chunked path. Do not enable
  P2P. Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04x - P4 mixed 2x2 decode sendrecv+AR both cards

CONTEXT -> Parked charter P4 after
  P3 host-staged passed (04j) and
  P2 chunked AG 4x64h passed (04w,
  2162 us). Decode-sized PP
  sendrecv + TP all_reduce. Timeout
  90s. Pause one-card. P2P off.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric 2x2_decode, p2p=0.
  Image
  b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch
  2.13.0+xpu. Outer timeout 90s
  TERM then kill-after 10s.
  No CCL_TOPO_P2P_ACCESS=1.
  Payload decode_h 5120 bf16 =
  10240 bytes.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run bash parallel/2x2/run_decode.sh
  # post xpu-health same as pre
  ```

RESULT -> Pre: card0 HEALTHY
  (19s), card1 HEALTHY (20s).
  Identity: rank=0 pp_ok=1 tp_ok=1,
  rank=1 pp_ok=1 tp_ok=1. RESULT
  op=2x2_decode name=decode_h
  numel=5120 bytes=10240
  us=689.721 pp_ok=1 tp_ok=1.
  VERDICT_LINE ok_all=1
  path=pp_sendrecv+tp_ar p2p=0.
  TIMEOUT_OR_EXIT rc=0.
  gpu-run 18s. Timeout not hit.
  CCL_TOPO_P2P_ACCESS 0.
  Topology: PCIe between
  devices. vs P2 AR 99-137 +
  P2 sendrecv 539-848 (additive
  ~638-985). Post: card0
  HEALTHY (24s), card1 HEALTHY
  (24s). Not WEDGED.

VERDICT -> First xe2x2 mixed 2x2
  synthetic. Decode sendrecv+AR
  is 689.721 us, ok_all=1,
  sendrecv-dominated. Identity
  closed. Teardown health
  recovered. One-shot 2.5 MiB AG
  still hangs (not this arm).
  Do not enable P2P. Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04z - K7 ESIMD mixer-slmhtc T=256 card1

CONTEXT -> NEW one-kernel mixer
  FIR conv + register L2 +
  slmht. Priors: seq conv 38 +
  slmht 260 ~298. mixer-slmht
  471. L2-once 327. conv-L2
  SLM 358. conv-L2r 531.
  Napkin: drop conv launch, L2
  free in registers, beat 298.
  spin=0.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_slmhtc.
  gpu-run --card 1. T=256
  C=10240 nv=48 blk=16. spin=0.
  Rank pipe_host vs seq 298.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_mixer_slmhtc_t256.sh 1 0
  ```

RESULT -> cosine=1 max_abs=7.6e-6
  cosine_o=1 max_abs_o=1.2e-4
  ok=1. event 563.070 pipe_host
  570.074. wait_host 586.271.
  timed_begin act=2800 cur=2800
  throttle=0. timed_end act=2717
  cur=2800 throttle=1. vs seq
  298 (~1.91x, a loss) vs
  mixer-slmht 471 (~1.21x) vs
  L2-once 327 (~1.74x) vs
  conv-L2 358 (~1.59x) vs
  conv-L2r 531 (~1.07x).
  gpu-run 1s.

VERDICT -> ESIMD mixer-slmhtc
  T=256 is 570.074 us pipe_host
  card1, a loss vs seq conv+slmht
  298. Numeric closed. One-card.
  STOP this fuse. Do not sibling.
  Sequential conv+slmht ~298
  stays the T=256 leftover. Do
  not freeze 570 as 2800. Rank
  pipe_host.

### 2026-09-04y - K7 ESIMD s8 4-acc + lsc_prefetch packed qkv M=256 card0

CONTEXT -> NEW s8 4-acc +
  lsc_prefetch_2d packed qkv
  M=256 n=10240 k=5120. Priors:
  W8A8 packed M=256 is 164 us.
  4-acc 4x8 274 lost. ngen
  issues ff prefetch. Steal
  lsc_prefetch_2d of next k128
  A/B on the 4-acc tile. NT=1
  prefetch o-proj STOP (04p).
  Rank pipe_host.

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4ff.
  prefetch=lsc_prefetch_2d.
  gpu-run --card 0. NT=2 m=256
  n=10240 k=5120. wg=4x8 4acc
  k128 unroll=8. spin=512.
  Fill s8 [-64,64] scales 0.02
  out f16. Rank pipe_host.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_m4ff_m256.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 269.198
  pipe_host 267.199. wait_host
  285.027. spin_done act=2650
  cur=2800 throttle=0. timed
  act=2650-2733 cur=2800
  throttle=1. vs W8A8 164
  (~1.63x, a loss) vs 4-acc
  274 (~0.97x, 274-class, not
  a steal). gpu-run 119s.

VERDICT -> ESIMD packed qkv s8
  4-acc + lsc_prefetch_2d M=256
  is 267.199 us pipe_host card0
  at act=2650-2733 cur=2800
  throttle=1, a loss vs oneDNN
  W8A8 164. Not a W8A8-contract
  beat. 274-class vs 4-acc 274,
  not a steal. Numeric closed.
  One-card. Do not freeze (not
  2800). STOP prefetch on 4-acc
  M=256. Do not sibling. Do not
  promote. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04aa - P2 XCCL sendrecv 2.5 MiB

CONTEXT -> Parked charter P2 leftover
  after one-shot 2.5 MiB XCCL
  all_gather hang (2026-09-04i/t)
  and AR 256h 2081 us (04i).
  sendrecv ping-pong only.
  P2P off. Timeout 45s.
  Pause one-card.

CONFIG -> backend pytorch-xpu on
  sycl+l0, fabric xccl, p2p=0
  timeout=45s. Image
  b70-sglang-xpu-int8-runtime:20260826-mtp6.
  gpu-run both cards. torch
  2.13.0+xpu. Outer timeout 45s
  TERM then kill-after 8s.
  No CCL_TOPO_P2P_ACCESS=1.
  Payload prefill_256h 256*5120
  bf16 = 2621440 bytes.

COMMAND ->
  ```
  gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 --timeout 180
  gpu-run xpu-collective-health --p2p 0 --timeout 240
  gpu-run bash parallel/tp2/run_sr256.sh
  # post same three
  ```

RESULT -> Pre: card0 HEALTHY
  (16s), card1 HEALTHY (13s),
  COLLECTIVE_HEALTH_OK 4x5120
  p2p=0 (30s). Identity:
  rank=0 ok=1, rank=1 ok=1.
  RESULT op=sendrecv
  name=prefill_256h numel=1310720
  bytes=2621440 us=2640.586
  ok=1. VERDICT_LINE ok_all=1
  path=sendrecv_256h p2p=0.
  TIMEOUT_OR_EXIT rc=0.
  gpu-run 19s. Timeout not hit.
  CCL_TOPO_P2P_ACCESS 0.
  Topology: PCIe between
  devices. throttle=0. act
  briefly 2800, cur mixed
  2150-2800, not held. Do not
  freeze 2641 as 2800. vs 64h
  sendrecv 890-948 us. vs AR
  256h 2081 us. vs chunked AG
  2162 us. vs one-shot AG hang
  (04t rc=124 at 45s). Post:
  card0 HEALTHY (15s), card1
  HEALTHY (15s),
  COLLECTIVE_HEALTH_OK (27s).
  Not WEDGED.

VERDICT -> XCCL P2P-off sendrecv
  at 2.5 MiB is 2640.586 us,
  ok_all=1. Identity closed.
  Teardown health recovered.
  One-shot AG >=2.5 MiB still
  hangs; STOP that path.
  Do not enable P2P. Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ac - K7 ESIMD mixer T-chunk two-queue pipe T=256 card1

CONTEXT -> NEW leftover steal:
  T-chunk two-queue conv+slmht
  pipeline. NOT a fuse. Priors:
  seq conv~38 + slmht~260 ~298.
  mixer-slmht 471. L2-once 327.
  conv-L2 358. conv-L2r 531.
  slmhtc 570 STOP. Combined
  two-kernel mixer-slmht was
  471. Rank pipe_host.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_mixer_pipe.
  gpu-run --card 1. T=256
  C=10240 nv=48 blk=16 nchunk=4
  tchunk=64. spin=0. Rank
  pipe_host vs seq 298.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/gdn/run_esimd_mixer_pipe_t256.sh 1 0
  ```

RESULT -> cosine=0.984986
  max_abs=0.0038319
  cosine_o=0.957795
  max_abs_o=0.13934 ok=0.
  event 398.872 pipe_host
  463.529. median 460.969
  min 455.999 max 567.182.
  timed_begin act=2800 cur=2800
  throttle=0. timed_end act=2800
  cur=2800 throttle=0. vs seq
  298 (~1.56x, a loss) vs
  mixer-slmht 471 (~0.98x us,
  not a steal: ok=0). gpu-run
  1s exit 1.

VERDICT -> ESIMD mixer T-chunk
  two-queue pipe T=256 is
  463.529 us pipe_host card1,
  numeric not closed (s 0.985
  o 0.958, both <0.99). Loss
  vs seq conv+slmht 298 even
  on us. STOP this pipe. Do
  not sibling. Sequential
  conv+slmht ~298 stays the
  T=256 leftover. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ab - K7 ESIMD s8 4-acc split-K=2 + lsc_prefetch packed qkv M=256 card0

CONTEXT -> NEW leftover steal:
  ESIMD s8 4-acc split-K=2 +
  lsc_prefetch_2d packed qkv
  M=256 n=10240 k=5120. Priors:
  W8A8 packed M=256 is 164 us.
  4-acc 274, prefetch-only
  267.199 STOP, SK=2 295, SK=5
  393, 2-acc 327, persist 344,
  wg 8x4 279. M=64 SK=2 115
  stands vs W8A8 140. This arm
  is SK=2 occupancy PLUS
  prefetch. Not a stopped tile.
  Rank pipe_host (event is
  reduce-only on SK kernels).

CONFIG -> backend sycl+l0,
  arm dpas_s8_sc8w48m4skff.
  prefetch=lsc_prefetch_2d.
  gpu-run --card 0. NT=2 m=256
  n=10240 k=5120. splitK=2
  wg=4x8 4acc k128 unroll=5.
  spin=512. Fill s8 [-64,64]
  scales 0.02 out f16. Rank
  pipe_host of gemm+reduce.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/gdn/run_esimd_s8_qkv_skff_m256.sh 0 512
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event 47.120
  (reduce-only). pipe_host
  291.934. wait_host 307.370.
  spin_done act=2583 cur=2800
  throttle=1. timed act=2583
  cur=2800 throttle=1. vs W8A8
  164 (~1.78x, a loss) vs
  prefetch 267.199 (~1.09x, a
  loss) vs SK=2 295 (~0.99x,
  295-class, not a steal) vs
  4-acc 274 (~1.07x, a loss).
  gpu-run 119s.

VERDICT -> ESIMD packed qkv s8
  4-acc split-K=2 +
  lsc_prefetch_2d M=256 is
  291.934 us pipe_host card0
  at act=2583 cur=2800
  throttle=1, a loss vs oneDNN
  W8A8 164. Not a W8A8-contract
  beat. 295-class vs SK=2 295,
  worse than prefetch-only 267.
  Combining two stopped tiles
  is not a steal. Numeric
  closed. One-card. Do not
  freeze (not 2800). STOP
  SK=2+prefetch at M=256. Do
  not sibling. Do not promote.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ae - K8 oneDNN W8A8 Lightning routed expert UP-proj M=1 card1

CONTEXT -> K8 incumbent floor:
  oneDNN int8_gemm_w8a8 at
  Lightning routed expert
  UP-proj M=1 n=1856 k=2688.
  Prior: W8A8 square M=1 5120
  is 44 us. ONE expert
  GEMM-only. Not 6-expert
  grouped. Not a serve.

CONFIG -> backend pytorch-xpu
  on sycl+l0. gpu-run --card 1.
  Image b70-sglang-xpu-int8-
  runtime:20260826-mtp6.
  int8_gemm_w8a8. heat M=64
  spin=512. m=1 n=1856 k=2688.
  Rank us. No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/nemotron/run_w8a8_moe_up_m1.sh 1
  ```

RESULT -> cosine=1.000000
  max_abs=0.030334 ok=1.
  44.285 us. 112.655 GB/s.
  vs square 44. Napkin
  44*(1856/5120)~16 (CONFIG).
  GPU-window act=550-2800
  cur=550-2800 throttle=0.
  Two samples act=cur=2800.
  start act=0 cur=2800
  throttle=0 D3hot. end
  act=0 cur=2800 throttle=0
  D0. gpu-run 14s.

VERDICT -> oneDNN W8A8 Lightning
  routed expert UP-proj M=1 is
  44.285 us card1, 44-class
  launch like square 5120, not
  N-linear. Numeric closed.
  throttle=0. Do not freeze
  (act not held 2800). One-
  card enough (W8A8 family
  already matched both cards).
  Rank us.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ad - K8 ESIMD s8 RC=4 Lightning routed expert UP-proj M=1 card0

CONTEXT -> K8 first s8 floor:
  existing RC=4 8x2-N
  scale-to-f16 tile at
  Lightning routed expert
  UP-proj M=1 n=1856 k=2688.
  Priors: square s8 M=1 5120
  is 34 us. Napkin N-linear
  34*(1856/5120)~12 us.
  W8A8 sibling card1 is
  44.285 us (04ae). ONE
  expert, not grouped-6.
  Stock dpas_s8_sc NT=2 U=16
  refused: inner_k=1024,
  2688%1024=640. Same tile
  U=14 inner_k=896 divides
  2688. Rank pipe_host.

CONFIG -> backend sycl+l0,
  standalone AOT
  dpas_s8_sc_u14. gpu-run
  --card 0. NT=2 U=14 m=1
  n=1856 k=2688. spin=4000.
  Fill s8 [-64,64] scales
  0.02 out f16. RC=4
  wg=8x2_alongN dpas=56.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/nemotron/run_esimd_s8_moe_up_m1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event
  15.622 pipe_host 16.060.
  median 15.625 min 14.375
  max 16.458. TOPS 0.6387.
  spin_done act=cur=2800
  throttle=0. timed
  act=cur=2800 throttle=0
  both ends. vs W8A8 44.285
  (~0.36x, a beat) vs square
  s8 34 (~0.47x) vs napkin
  12 (~1.30x). Stock U=16
  check-only then rc=2.
  gpu-run 2s.

VERDICT -> ESIMD s8 RC=4
  Lightning routed expert
  UP-proj M=1 is 16.060 us
  pipe_host card0 at 2800,
  a beat of oneDNN W8A8
  44.285. Stock U=16 cannot
  run hidden 2688; U=14 is
  the same family. Numeric
  closed. One-card enough
  (matched s8 RC=4 family,
  cosine closed, clocks
  held). ONE expert, not
  grouped-6. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04af - K8 ESIMD Mamba-2 SSD SSU T=1 card0

CONTEXT -> K8 NEW math: first
  Mamba-2 SSD SSU decode T=1.
  Not GDN. Priors: GDN fused
  decode 7.1 us is the WRONG
  math. Sibling SSU B8/W4 is
  a community floor, not
  FINDINGS. Binary
  mamba_ssu_t1 COMPILE_OK.
  Rank pipe_host.

CONFIG -> backend sycl+l0,
  standalone AOT mamba_ssu_t1
  intel_gpu_bmg_g31. gpu-run
  --card 0. T=1 heads=64
  d_head=64 d_state=128
  groups=8 VL=16
  wi=one_per_head. spin=0.
  Rank pipe_host vs eager
  napkin. No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/nemotron/run_mamba_ssu_t1.sh 0 0
  ```

RESULT -> cosine=1.000000
  max_abs=4.7684e-07 ok=1.
  event 234.510 pipe_host
  190.028 wait_host 258.467.
  22.18 GB/s. median 234.219
  min 231.458 max 238.542.
  timed_begin act=950 cur=933
  throttle=0. timed_end
  act=cur=2800 throttle=0.
  start D3hot act=0 cur=2800
  throttle=0. end D0 act=0
  cur=2800 throttle=0. freq
  throttle=0. vs GDN delta
  7.1 (wrong math, not a
  steal). gpu-run 2s.

VERDICT -> ESIMD Mamba-2 SSD
  SSU T=1 is 190.028 us
  pipe_host card0, numeric
  closed. Clocks not held
  (spin=0, timed_begin
  act=950). Do not freeze
  190. One-card first light;
  sibling pending (new math,
  not a both-card floor).
  GDN 7.1 is the wrong math.
  Sibling SSU B8/W4 stays
  community. Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ag - K8 ESIMD grouped 6-expert s8 decode M=1 card1

CONTEXT -> K8 grouped decode:
  6 routed experts, each s8
  M=1 n=1856 k=2688, one
  in-order queue. Priors:
  one expert s8 U=14 is
  16.060 us card0 at 2800
  (04ad). W8A8 one expert
  44.285 us (04ae). Napkin
  6*16~96 vs 6*44~266.
  Rank pipe_host of all 6.

CONFIG -> backend sycl+l0,
  standalone AOT
  moe_group_s8_m1. gpu-run
  --card 1. experts=6 m=1
  n=1856 k=2688. spin=4000.
  RC=4 NT=2 kstep=64
  wg=8x2_alongN scale 0.02
  out f16. Six launches
  share A and one in-order
  queue. Not U=14.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/nemotron/run_moe_group_s8_m1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 ok=1. event
  164.633 last_event 33.398
  wait_host 231.041
  pipe_host 165.223. TOPS
  0.3623. median_sum 164.636
  min 161.770 max 168.749.
  spin_done act=cur=2800
  throttle=0. timed
  act=cur=2800 throttle=0
  both ends. freq 50 ms
  throttle=0 all 17 samples.
  GPU-window act=517,400 then
  7x 2800. start act=0
  cur=2800 throttle=0 D3hot.
  end act=0 cur=2800
  throttle=0 D0. vs
  6*16.060=96.360 (~1.72x)
  vs 6*44.285=265.710
  (~0.62x). gpu-run 2s.

VERDICT -> ESIMD grouped 6
  routed-expert s8 decode
  M=1 is 165.223 us
  pipe_host card1 at 2800.
  Beats 6x W8A8 266 napkin.
  Loses to 6x U=14 96 napkin
  (~1.72x): this binary is
  k64-loop not U=14; mean
  event 27.4 us/expert vs
  16.060. pipe_host ~ event
  sum, so the in-order queue
  packed the six launches.
  Numeric closed. Clocks
  held. One-card enough
  (matched s8 RC=4 family).
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ai - K8 ESIMD Mamba conv1d K=4 C=4096 T=1 card1

CONTEXT -> K8 Mamba depthwise
  conv K=4 C=4096 T=1 using
  existing gdn_conv1d --c 4096.
  Same FIR as GDN, different
  C (not 10240 leftover).
  Priors: GDN conv C=2048 T=1
  4.350/4.500 us at 1700/1400.
  C=6144 T=1 4.799/5.000.
  Fused qkv T=1 C=2048 ~4.4.
  Occupancy may wash vs
  C=2048. Rank pipe_host.

CONFIG -> backend sycl+l0,
  standalone AOT gdn_conv1d
  intel_gpu_bmg_g31. gpu-run
  --card 1. T=1 C=4096 k=4
  f16 VL=16 wg=16. spin=4000
  mhz=2400. Generic --c path.
  No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/nemotron/run_mamba_conv_c4096.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=0 cosine_st=1.000000
  max_abs_st=0 ok=1. event
  0.896 wait_host 14.891
  pipe_host 4.355. 22.57 GB/s.
  median 0.937 min 0.729 max
  0.938. spin_done act=cur=2800
  throttle=0. timed
  act=cur=2800 throttle=0
  both ends. freq 50 ms
  throttle=0 all 7 samples.
  GPU-window act=0,0,0,0,550,
  0,0 cur=2800,2800,2800,400,
  2800,2800,2800 (sampler
  miss, short kernel). start
  D3hot act=0 cur=2800
  throttle=0. end D0 act=0
  cur=2800 throttle=0. vs
  C=2048 4.350/4.500 (wash,
  not 2x). vs C=6144 4.799/
  5.000. vs fused qkv 4.4.
  gpu-run 2s.

VERDICT -> ESIMD Mamba conv1d
  K=4 C=4096 T=1 is 4.355 us
  pipe_host card1 at 2800
  (kernel sample), occupancy
  wash vs GDN C=2048 4.4 not
  2x. Numeric closed. Clocks
  held at kernel sample.
  One-card enough (conv
  family already both-card
  at other C). Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ah - K8 ESIMD Mamba-2 SSD SSU T=1 held-clock card0

CONTEXT -> K8 NEW math: first
  held-clock Mamba-2 SSD SSU
  decode T=1. Not GDN. Prior
  04af first light 190.028 us
  pipe_host spin=0 (timed_begin
  act=950, clocks not held).
  Do not freeze 190. Rank
  pipe_host.

CONFIG -> backend sycl+l0,
  standalone AOT mamba_ssu_t1
  intel_gpu_bmg_g31. gpu-run
  --card 0. T=1 heads=64
  d_head=64 d_state=128
  groups=8 VL=16
  wi=one_per_head. spin=4000.
  Rank pipe_host vs 04af 190
  and vs GDN 7.1 (wrong math).
  No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/nemotron/run_mamba_ssu_t1.sh 0 4000
  ```

RESULT -> cosine=1.000000
  max_abs=4.7684e-07 ok=1.
  event 79.531 pipe_host
  80.064 wait_host 93.563.
  52.65 GB/s. median 79.427
  min 78.541 max 80.833.
  spin_done act=cur=2800
  throttle=0. timed
  act=cur=2800 throttle=0
  both ends. start D0 act=0
  cur=2800 throttle=0. end D0
  act=cur=2800 throttle=0.
  freq 50 ms throttle=0 all
  11 samples. GPU-window
  act=0,0,0,400 then 4x 2800
  then 3x 0. vs 04af 190.028
  (~0.42x; 190 was ramp). vs
  GDN delta 7.1 (wrong math,
  not a steal). gpu-run 2s.

VERDICT -> ESIMD Mamba-2 SSD
  SSU T=1 is 80.064 us
  pipe_host card0 at 2800,
  numeric closed. Clocks held
  (spin=4000, timed
  act=cur=2800). 04af 190.028
  was spin=0 ramp; do not
  freeze 190. One-card held-
  clock; sibling pending (new
  math, not a both-card
  floor). GDN 7.1 is the
  wrong math. Sibling SSU
  B8/W4 stays community.
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04aj - K8 ESIMD Mamba-2 SSD SSU T=1 held-clock card1 sibling

CONTEXT -> K8 NEW math sibling
  of 04ah card0 held-clock
  80.064 us pipe_host. First
  both-card check for Mamba-2
  SSD SSU decode T=1. Not GDN.
  Rank pipe_host. Promote
  FINDINGS floor if spread vs
  80.064 is <5% and clocks
  held 2800.

CONFIG -> backend sycl+l0,
  standalone AOT mamba_ssu_t1
  intel_gpu_bmg_g31. gpu-run
  --card 1. T=1 heads=64
  d_head=64 d_state=128
  groups=8 VL=16
  wi=one_per_head. spin=4000.
  Rank pipe_host vs 04ah
  80.064. No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 1 bash kernels/nemotron/run_mamba_ssu_t1.sh 1 4000
  ```

RESULT -> cosine=1.000000
  max_abs=4.7684e-07 ok=1.
  event 79.536 pipe_host
  79.923 wait_host 93.906.
  52.74 GB/s. median 79.375
  min 78.541 max 80.937.
  spin_done act=cur=2800
  throttle=0. timed
  act=cur=2800 throttle=0
  both ends. start D3hot
  act=0 cur=2800 throttle=0.
  end D0 act=0 cur=2800
  throttle=0. freq 50 ms
  throttle=0 all 11 samples.
  GPU-window act=0,0,0,400
  then 4x 2800 then 2x 0.
  vs 04ah card0 80.064 spread
  0.176% (<5%). gpu-run 2s.

VERDICT -> ESIMD Mamba-2 SSD
  SSU T=1 is 79.923 us
  pipe_host card1 at 2800,
  numeric closed. Clocks held
  (spin=4000, timed
  act=cur=2800). Spread vs
  04ah 80.064 is 0.176% at
  2800. Both-card FINDINGS
  floor: 80 us pipe_host at
  2800. GDN 7.1 is the wrong
  math. Sibling SSU B8/W4
  stays community. Rank
  pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

### 2026-09-04ak - K8 NVFP4 nibble LUT Lightning routed expert UP-proj M=1 card0 REFUSED

CONTEXT -> K8 shape steal on
  held K6 merge LUT family at
  Lightning routed expert
  UP-proj M=1 n=1856 k=2688.
  Priors: nibble LUT 5120 is
  158 us. s8 expert-up 16.060
  (04ad U=14). W8A8 44.285
  (04ae). Stock nibble_lut_sc
  NT=2 U=16 inner_k=1024.
  Packed E2M1. Never bitcast
  s4. One-card LUT family.

CONFIG -> backend sycl+l0,
  standalone AOT nibble_lut_sc
  intel_gpu_bmg_g31. gpu-run
  --card 0. NT=2 U=16 m=1
  n=1856 k=2688. spin=4000.
  Packed E2M1 B 2/byte along
  K, simd nibble LUT, VNNI4,
  RC=4 8x2-N s8 scale-to-f16.
  No nibble_lut u14 binary.
  No P2P. No serve.

COMMAND ->
  ```
  gpu-run --card 0 bash kernels/nemotron/run_nibble_lut_moe_up_m1.sh 0 4000
  ```

RESULT -> REFUSED. stderr
  nibble_lut_sc: shape m=1
  n=1856 k=2688 nt=2 unroll=16.
  inner_k=16*64=1024,
  2688%1024=640. n=1856%32=0
  (N ok). Check-only 4x32x1024
  cosine=1.000000 max_abs=0
  ok=1 event 209.688 pipe_host
  204.179 at act=400/550 (not
  held; not the Lightning
  shape). timed Lightning
  pipe_host REFUSED. No
  nibble_lut_sc_u14 (NT=2
  launch is template U=16).
  start D3hot act=0 cur=2800
  throttle=0. gpu-run 2s
  exit 2.

VERDICT -> Stock nibble_lut_sc
  U=16 cannot run Lightning
  hidden 2688. pipe_host
  REFUSED. STOP. Do not
  rewrite (no one-line u14
  exists for nibble_lut).
  158 us stays the 5120
  FINDING, not this shape.
  One-card enough (LUT family
  already both-card at 5120;
  this is a mapping refuse).
  Rank pipe_host.
Do not drop below 5m: M=256 FFN spin=512
already 2-4 min GPU, overlapping fires
serialize on gpu-run.

