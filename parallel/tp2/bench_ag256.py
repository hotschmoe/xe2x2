#!/usr/bin/env python3
"""P2 leftover: XCCL all_gather and sendrecv at 2.5 MiB only, P2P off.

Decode through 64h already passed. This arm is the hang size.
Outer docker timeout must kill a hang. Rank us or TIMEOUT.
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
            "op=all_gather+sendrecv name=prefill_256h bytes",
            nbytes,
            flush=True,
        )

    src = torch.full((numel,), float(rank + 1), dtype=torch.bfloat16, device=device)
    out = [torch.empty((numel,), dtype=torch.bfloat16, device=device) for _ in range(2)]
    dist.all_gather(out, src)
    torch.xpu.synchronize()
    ag_ok = 1
    for r in range(2):
        if abs(out[r][0].float().item() - float(r + 1)) > 0.02:
            ag_ok = 0
    print(f"OK op=all_gather rank={rank} ok={ag_ok}", flush=True)

    def do_ag():
        s = torch.full((numel,), float(rank + 1), dtype=torch.bfloat16, device=device)
        o = [torch.empty((numel,), dtype=torch.bfloat16, device=device) for _ in range(2)]
        dist.all_gather(o, s)

    ag_us = us_bench(do_ag, warmup=2, iters=8)
    if rank == 0:
        print(
            f"RESULT op=all_gather name=prefill_256h numel={numel} bytes={nbytes} "
            f"us={ag_us:.3f} ok={ag_ok}",
            flush=True,
        )

    ping = torch.full((numel,), 7.0 + rank, dtype=torch.bfloat16, device=device)
    pong = torch.empty((numel,), dtype=torch.bfloat16, device=device)
    if rank == 0:
        dist.send(ping, dst=1)
        dist.recv(pong, src=1)
    else:
        dist.recv(pong, src=0)
        dist.send(ping, dst=0)
    torch.xpu.synchronize()
    sr_ok = 1 if abs(pong[0].float().item() - (7.0 + (1 - rank))) < 0.02 else 0
    print(f"OK op=sendrecv rank={rank} ok={sr_ok}", flush=True)

    def do_sr():
        a = torch.full((numel,), 7.0 + rank, dtype=torch.bfloat16, device=device)
        b = torch.empty((numel,), dtype=torch.bfloat16, device=device)
        if rank == 0:
            dist.send(a, dst=1)
            dist.recv(b, src=1)
        else:
            dist.recv(b, src=0)
            dist.send(a, dst=0)

    sr_us = us_bench(do_sr, warmup=2, iters=8)
    if rank == 0:
        print(
            f"RESULT op=sendrecv name=prefill_256h numel={numel} bytes={nbytes} "
            f"us={sr_us:.3f} ok={sr_ok}",
            flush=True,
        )
        print(f"VERDICT_LINE ok_ag={ag_ok} ok_sr={sr_ok} p2p=0", flush=True)
    dist.destroy_process_group()
    return 0 if ag_ok and sr_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
