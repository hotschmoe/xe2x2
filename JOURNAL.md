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
