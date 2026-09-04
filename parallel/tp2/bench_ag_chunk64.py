#!/usr/bin/env python3
"""P2 leftover: chunked XCCL all_gather. 2.5 MiB as 4x 64h.

64h all_gather already passed (~544 us). One-shot 256h hangs at 45s.
This arm is 4 sequential 64-token gathers, P2P off. Rank us.
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
        timeout=timedelta(seconds=60),
        device_id=torch.device(f"xpu:{local_rank}"),
    )
    device = torch.device(f"xpu:{local_rank}")
    chunk = 64 * 5120
    nchunk = 4
    total = chunk * nchunk
    if rank == 0:
        print(
            "CONFIG backend=pytorch-xpu on sycl+l0 fabric=xccl p2p=0",
            "op=chunked_all_gather chunk=64h nchunk=4 bytes",
            total * 2,
            flush=True,
        )

    src = [
        torch.full((chunk,), float(rank + 1 + c), dtype=torch.bfloat16, device=device)
        for c in range(nchunk)
    ]
    outs = [
        [torch.empty((chunk,), dtype=torch.bfloat16, device=device) for _ in range(2)]
        for _ in range(nchunk)
    ]
    ok = 1
    for c in range(nchunk):
        dist.all_gather(outs[c], src[c])
        torch.xpu.synchronize()
        for r in range(2):
            expect = float(r + 1 + c)
            if abs(outs[c][r][0].float().item() - expect) > 0.05:
                ok = 0
    print(f"OK op=chunked_all_gather rank={rank} ok={ok}", flush=True)

    def do_ag():
        for c in range(nchunk):
            dist.all_gather(outs[c], src[c])

    us = us_bench(do_ag, warmup=4, iters=10)
    if rank == 0:
        print(
            f"RESULT op=chunked_all_gather name=prefill_256h_as_4x64h "
            f"chunk_numel={chunk} nchunk={nchunk} bytes={total * 2} "
            f"us={us:.3f} ok={ok}",
            flush=True,
        )
        print(f"VERDICT_LINE ok_all={ok} path=chunked_ag_64h p2p=0", flush=True)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
