# P2 -- tensor parallel = 2

Question: what is a correct, healthy, and then *cheap* two-rank map
on this PCI tree?

Open. oneCCL / XCCL, host-staged copies, llama.cpp-style splits, and
push all-reduce are arms. P2P on is a labeled control after P2P-off
collectives are healthy. Do not start from a vLLM TP=2 serve.

## Why

The two B70s sit on separate Threadripper root complexes
(`pci0000:00` vs `pci0000:40`, Gen3 x16 each). Cross-die P2P is the
worst case, and it already sometimes works. The serving-tree failure
mode is process / queue handoff, not "P2P is impossible." Push
all-reduce plus fused RMSNorm was a useful prototype in
`b70_ai_things` (`xpu_push_ar_fused_rmsnorm.cpp`). Minimum number of collectives per token is the latency score. GB/s
of one fat allreduce is diagnostic. Some ops should stay TP=1
(replicate). Some should shard. Joining adjacent kernels so an
allreduce between them disappears is the TP=2 fusion play. P2P/AR
adds latency on this cross-die Gen3 tree; cutting calls beats
chasing peak payload bandwidth.

## Suggested arms (after P0 health)

- Synthetic all-reduce, all-gather, send-recv. Payload sizes from
  decode-hidden (5120 bf16) up to bulk (tens of MiB).
- P2P off, then P2P on as a control. Record ACS / IOMMU / health.
- Push all-reduce vs oneCCL default vs host-staged.
- Call-count: one fused AR vs many small ARs; AR fused into RMSNorm
  / residual vs standalone. Report us and calls/token first.
- Same op TP=1 (replicated) vs TP=2 (sharded) vs fused-neighbors
  TP=2. Do not assume TP=2 wins.
- Tiny sharded matmul only after the synthetic collective is correct.

Both cards, one job. Pause the one-card kernel matrix while this
runs. Do not also pin a K-workstream to card0.

## Record

Payload, us, GB/s, call count, P2P on/off, rank logs from both
cards, pre-health, post-health, teardown. Coherence of the reduction
vs a host oracle.

## Exit

A passing P2P-off collective + teardown health. Then a verdict on
push-AR vs oneCCL for decode-sized payloads. FINDINGS only with both
ranks and post-health. Speed is secondary until correctness holds.
