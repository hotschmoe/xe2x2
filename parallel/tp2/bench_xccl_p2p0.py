#!/usr/bin/env python3
"""P2 synthetic XCCL collectives, P2P off.

Times all_reduce, all_gather, and send/recv at decode-hidden through
bulk payloads. Correctness vs a host oracle. Rank us then GB/s.
Backend: pytorch-xpu on sycl+l0. No serve. No CCL_TOPO_P2P_ACCESS=1.
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
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def gbs(nbytes: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return (float(nbytes) / 1e9) / (us * 1e-6)


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError(f"expected world_size=2, got {world}")
    p2p = os.environ.get("CCL_TOPO_P2P_ACCESS", "?")
    if p2p != "0":
        raise RuntimeError(f"P2 refused: CCL_TOPO_P2P_ACCESS={p2p} (want 0)")

    torch.xpu.set_device(local_rank)
    dist.init_process_group(
        backend="xccl",
        timeout=timedelta(seconds=180),
        device_id=torch.device(f"xpu:{local_rank}"),
    )
    device = torch.device(f"xpu:{local_rank}")
    if rank == 0:
        print(
            "CONFIG backend=pytorch-xpu on sycl+l0 fabric=xccl p2p=0",
            "world=2",
            "torch",
            torch.__version__,
            "ZE_AFFINITY_MASK",
            os.environ.get("ZE_AFFINITY_MASK"),
            flush=True,
        )

    # decode hidden 5120 bf16 through bulk. nbytes = numel * 2.
    shapes = [
        ("decode_h", 5120),
        ("health_4h", 4 * 5120),
        ("prefill_64h", 64 * 5120),
        ("prefill_256h", 256 * 5120),
        ("1MiB", 1024 * 1024 // 2),
        ("8MiB", 8 * 1024 * 1024 // 2),
    ]

    ok_all = 1
    try:
        for name, numel in shapes:
            nbytes = numel * 2

            print(
                f"BEGIN name={name} numel={numel} bytes={nbytes} rank={rank}",
                flush=True,
            )

            # Reuse device buffers. Allocating a fresh tensor every timed
            # iteration can force a new XCCL/IGC compile per call.
            ar_buf = torch.empty((numel,), dtype=torch.bfloat16, device=device)
            ag_src = torch.empty((numel,), dtype=torch.bfloat16, device=device)
            ag_out = [
                torch.empty((numel,), dtype=torch.bfloat16, device=device)
                for _ in range(world)
            ]
            sr_a = torch.empty((numel,), dtype=torch.bfloat16, device=device)
            sr_b = torch.empty((numel,), dtype=torch.bfloat16, device=device)

            # all_reduce SUM. rank r contributes (r+1). expect 3.
            ar_buf.fill_(float(rank + 1))
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM)
            torch.xpu.synchronize()
            got = ar_buf[0].float().item()
            ar_ok = 1 if abs(got - 3.0) < 0.02 else 0
            if not ar_ok:
                ok_all = 0
            print(
                f"OK op=all_reduce name={name} rank={rank} got={got:.4f} ok={ar_ok}",
                flush=True,
            )

            def do_ar():
                dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM)

            ar_us = us_bench(do_ar, warmup=8, iters=20)
            if rank == 0:
                print(
                    f"RESULT op=all_reduce name={name} numel={numel} bytes={nbytes} "
                    f"us={ar_us:.3f} GBs={gbs(nbytes, ar_us):.3f} ok={ar_ok}",
                    flush=True,
                )

            # all_gather: each rank sends numel, out is 2*numel.
            ag_src.fill_(float(rank + 1))
            dist.all_gather(ag_out, ag_src)
            torch.xpu.synchronize()
            ag_ok = 1
            for r in range(world):
                if abs(ag_out[r][0].float().item() - float(r + 1)) > 0.02:
                    ag_ok = 0
                    ok_all = 0
            print(
                f"OK op=all_gather name={name} rank={rank} ok={ag_ok}",
                flush=True,
            )

            def do_ag():
                dist.all_gather(ag_out, ag_src)

            ag_us = us_bench(do_ag, warmup=8, iters=20)
            if rank == 0:
                print(
                    f"RESULT op=all_gather name={name} numel={numel} bytes={nbytes} "
                    f"us={ag_us:.3f} GBs={gbs(2 * nbytes, ag_us):.3f} ok={ag_ok}",
                    flush=True,
                )

            # send/recv: rank0 -> rank1 then rank1 -> rank0 (ping-pong pair).
            sr_a.fill_(7.0 + rank)
            if rank == 0:
                dist.send(sr_a, dst=1)
                dist.recv(sr_b, src=1)
            else:
                dist.recv(sr_b, src=0)
                dist.send(sr_a, dst=0)
            torch.xpu.synchronize()
            expect = 7.0 + (1 - rank)
            sr_ok = 1 if abs(sr_b[0].float().item() - expect) < 0.02 else 0
            if not sr_ok:
                ok_all = 0
            print(
                f"OK op=sendrecv name={name} rank={rank} ok={sr_ok}",
                flush=True,
            )

            def do_sr():
                if rank == 0:
                    dist.send(sr_a, dst=1)
                    dist.recv(sr_b, src=1)
                else:
                    dist.recv(sr_b, src=0)
                    dist.send(sr_a, dst=0)

            sr_us = us_bench(do_sr, warmup=8, iters=20)
            if rank == 0:
                print(
                    f"RESULT op=sendrecv name={name} numel={numel} bytes={nbytes} "
                    f"us={sr_us:.3f} GBs={gbs(nbytes, sr_us):.3f} ok={sr_ok}",
                    flush=True,
                )

        dist.barrier()
        if rank == 0:
            print(f"VERDICT_LINE ok_all={ok_all} p2p=0", flush=True)
        return 0 if ok_all else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
