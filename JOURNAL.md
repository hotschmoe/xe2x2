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




