#!/usr/bin/env python3
"""P2 leftover: XCCL sendrecv at 2.5 MiB, P2P off.

all_reduce 256h worked (~2081 us). all_gather 256h hangs at 45s.
This arm is ping-pong sendrecv only. Outer timeout 45s. Rank us or HANG.
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
        raise RuntimeError("P2 refused: want p2p=0")
    torch.xpu.set_device(local_rank)
    dist.init_process_group(
        backend="xccl",
        timeout=timedelta(seconds=30),
        device_id=torch.device(f"xpu:{local_rank}"),
    )
    device = torch.device(f"xpu:{local_rank}")
    numel = 256 * 5120
    nbytes = numel * 2
    if rank == 0:
        print(
            "CONFIG backend=pytorch-xpu on sycl+l0 fabric=xccl p2p=0",
            "op=sendrecv name=prefill_256h bytes",
            nbytes,
            flush=True,
        )
    a = torch.full((numel,), 7.0 + rank, dtype=torch.bfloat16, device=device)
    b = torch.empty((numel,), dtype=torch.bfloat16, device=device)
    if rank == 0:
        dist.send(a, dst=1)
        dist.recv(b, src=1)
    else:
        dist.recv(b, src=0)
        dist.send(a, dst=0)
    torch.xpu.synchronize()
    ok = 1 if abs(b[0].float().item() - (7.0 + (1 - rank))) < 0.05 else 0
    print(f"OK op=sendrecv rank={rank} ok={ok}", flush=True)

    def do_sr():
        if rank == 0:
            dist.send(a, dst=1)
            dist.recv(b, src=1)
        else:
            dist.recv(b, src=0)
            dist.send(a, dst=0)

    us = us_bench(do_sr, warmup=2, iters=8)
    if rank == 0:
        print(
            f"RESULT op=sendrecv name=prefill_256h numel={numel} bytes={nbytes} "
            f"us={us:.3f} ok={ok}",
            flush=True,
        )
        print(f"VERDICT_LINE ok_all={ok} path=sendrecv_256h p2p=0", flush=True)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
