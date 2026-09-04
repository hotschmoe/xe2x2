# P4 mixed 2x2 decode sendrecv+AR 2026-09-04x

Backend pytorch-xpu on sycl+l0. fabric 2x2_decode.
p2p=0. gpu-run both cards. No serve. Rank us.
PP sendrecv + TP all_reduce at hidden 5120 bf16.
timeout=90s. No clock spin.

Pre-health: card0 HEALTHY, card1 HEALTHY.

| name | op | bytes | us | ok |
|---|---|---:|---:|---:|
| decode_h 10KiB | pp_sendrecv+tp_ar | 10240 | 689.721 | 1 |

ok_all=1. rank0 pp_ok=1 tp_ok=1, rank1 pp_ok=1 tp_ok=1.
TIMEOUT_OR_EXIT rc=0. gpu-run 18s. Timeout not hit.
CCL_TOPO_P2P_ACCESS 0. Topology: PCIe.
vs P2 AR 99-137 + P2 sendrecv 539-848
(additive ~638-985). Sendrecv-dominated.

Post-health: card0 HEALTHY, card1 HEALTHY.
Not WEDGED.

First xe2x2 mixed 2x2 synthetic. Identity closed.
One-shot XCCL all_gather >= 2.5 MiB still hangs
(not this arm). Do not enable P2P.
