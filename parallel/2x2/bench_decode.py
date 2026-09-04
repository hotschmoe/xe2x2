#!/usr/bin/env python3
"""P4 mixed 2x2 synthetic, decode-sized, P2P off.

PP=2 sendrecv of hidden 5120 bf16 plus TP=2 all_reduce of the same
payload, one timed loop. Bulk 2.5 MiB one-shot AG still hangs; this
arm stays at decode_h. Rank us. Backend pytorch-xpu on sycl+l0.
"""
from __future__ import annotations

from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist


def us_bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("CCL_TOPO_P2P_ACCESS", "?") != "0":
        raise RuntimeError("P4 refused: want p2p=0")
    torch.xpu.set_device(local_rank)
    dist.init_process_group(
        backend="xccl",
        timeout=timedelta(seconds=60),
        device_id=torch.device(f"xpu:{local_rank}"),
    )
    device = torch.device(f"xpu:{local_rank}")
    n = 5120
    if rank == 0:
        print(
            "CONFIG backend=pytorch-xpu on sycl+l0 fabric=2x2_decode p2p=0",
            "pp=sendrecv tp=all_reduce numel",
            n,
            flush=True,
        )

    pp = torch.full((n,), 7.0 + rank, dtype=torch.bfloat16, device=device)
    pp_in = torch.empty((n,), dtype=torch.bfloat16, device=device)
    tp = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=device)

    def step():
        if rank == 0:
            dist.send(pp, dst=1)
            dist.recv(pp_in, src=1)
        else:
            dist.recv(pp_in, src=0)
            dist.send(pp, dst=0)
        tp.fill_(float(rank + 1))
        dist.all_reduce(tp, op=dist.ReduceOp.SUM)

    step()
    torch.xpu.synchronize()
    pp_ok = 1 if abs(pp_in[0].float().item() - (7.0 + (1 - rank))) < 0.05 else 0
    tp_ok = 1 if abs(tp[0].float().item() - 3.0) < 0.05 else 0
    print(f"OK op=2x2_decode rank={rank} pp_ok={pp_ok} tp_ok={tp_ok}", flush=True)
    us = us_bench(step, warmup=8, iters=20)
    if rank == 0:
        print(
            f"RESULT op=2x2_decode name=decode_h numel={n} bytes={n * 2} "
            f"us={us:.3f} pp_ok={pp_ok} tp_ok={tp_ok}",
            flush=True,
        )
        print(
            f"VERDICT_LINE ok_all={int(pp_ok and tp_ok)} path=pp_sendrecv+tp_ar p2p=0",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if pp_ok and tp_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
