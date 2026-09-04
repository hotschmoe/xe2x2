#!/usr/bin/env python3
"""P2 host-staged all-reduce, P2P off by construction.

Each rank copies bf16 to host, POSIX shm add, copy back.
Decode hidden through 2.5 MiB. No XCCL all_gather. Rank us.
Backend: pytorch-xpu on sycl+l0. No serve.
"""
from __future__ import annotations

from datetime import timedelta
import mmap
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
    p2p = os.environ.get("CCL_TOPO_P2P_ACCESS", "0")
    if p2p != "0":
        raise RuntimeError(f"P2 refused: CCL_TOPO_P2P_ACCESS={p2p}")

    torch.xpu.set_device(local_rank)
    dist.init_process_group(
        backend="xccl",
        timeout=timedelta(seconds=60),
        device_id=torch.device(f"xpu:{local_rank}"),
    )
    device = torch.device(f"xpu:{local_rank}")
    if rank == 0:
        print(
            "CONFIG backend=pytorch-xpu on sycl+l0 fabric=host_staged_ar p2p=0",
            "world=2 torch",
            torch.__version__,
            flush=True,
        )

    shapes = [
        ("decode_h", 5120),
        ("health_4h", 4 * 5120),
        ("prefill_64h", 64 * 5120),
        ("prefill_256h", 256 * 5120),
        ("1MiB", 1024 * 1024 // 2),
    ]
    shm_path = "/dev/shm/xe2x2_p2_host_ar"
    max_n = max(n for _, n in shapes)
    shm_bytes = 2 * max_n * 4
    if rank == 0:
        try:
            os.unlink(shm_path)
        except FileNotFoundError:
            pass
        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, shm_bytes)
    dist.barrier()
    if rank != 0:
        fd = os.open(shm_path, os.O_RDWR, 0o600)
    mm = mmap.mmap(fd, shm_bytes)
    ok_all = 1
    try:
        for name, numel in shapes:
            nbytes = numel * 2
            t = torch.full((numel,), float(rank + 1), dtype=torch.bfloat16, device=device)
            torch.xpu.synchronize()

            def do_ar():
                t.fill_(float(rank + 1))
                h = t.float().cpu().contiguous()
                dist.barrier()
                off = rank * max_n * 4
                mm[off : off + numel * 4] = h.numpy().tobytes()
                dist.barrier()
                import numpy as np

                a = np.frombuffer(mm, dtype=np.float32, count=max_n, offset=0)[:numel].copy()
                b = np.frombuffer(
                    mm, dtype=np.float32, count=max_n, offset=max_n * 4
                )[:numel].copy()
                s = torch.from_numpy(a + b)
                t.copy_(s.to(device=device, dtype=torch.bfloat16))

            do_ar()
            torch.xpu.synchronize()
            got = t[0].float().item()
            ok = 1 if abs(got - 3.0) < 0.05 else 0
            if not ok:
                ok_all = 0
            us = us_bench(do_ar, warmup=4, iters=10)
            if rank == 0:
                print(
                    f"RESULT op=host_ar name={name} numel={numel} bytes={nbytes} "
                    f"us={us:.3f} GBs={gbs(nbytes, us):.3f} ok={ok}",
                    flush=True,
                )
            t.fill_(float(rank + 1))
        dist.barrier()
        if rank == 0:
            print(f"VERDICT_LINE ok_all={ok_all} path=host_staged_ar p2p=off", flush=True)
        return 0 if ok_all else 1
    finally:
        mm.close()
        os.close(fd)
        dist.destroy_process_group()
        if rank == 0:
            try:
                os.unlink(shm_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
