#!/usr/bin/env python3
"""P3 synthetic PP=2 activation handoff, host-staged, P2P off.

stage0 (xpu:0) -> host DRAM -> stage1 (xpu:1). Identity vs a clone.
Also same-card copy as a control. Rank us and bubble vs payload.
Backend: pytorch-xpu on sycl+l0. One process, both cards visible.
No serve. Do not enable peer access.
"""
from __future__ import annotations

import os
import time

import torch


def us_bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def gbs(nbytes: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return (float(nbytes) / 1e9) / (us * 1e-6)


def main() -> int:
    p2p = os.environ.get("CCL_TOPO_P2P_ACCESS", "0")
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 fabric=host_staged_pp2 p2p",
        p2p,
        "torch",
        torch.__version__,
        "xpu_count",
        torch.xpu.device_count(),
        "ZE_AFFINITY_MASK",
        os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if torch.xpu.device_count() < 2:
        print("NEED_TWO_XPU", torch.xpu.device_count(), flush=True)
        return 2

    # Qwen3.8 hidden 5120 bf16. T=1 decode, T=64/256 prefill-ish.
    shapes = [
        ("T1_H5120", 1, 5120),
        ("T4_H5120", 4, 5120),
        ("T64_H5120", 64, 5120),
        ("T256_H5120", 256, 5120),
        ("T1_H6144", 1, 6144),
        ("1MiB", 1024 * 1024 // 2, 1),
    ]
    ok_all = 1
    for name, rows, cols in shapes:
        n = rows * cols
        nbytes = n * 2
        src = torch.arange(n, dtype=torch.int32, device="xpu:0").remainder_(31).to(
            torch.bfloat16
        ).reshape(rows, cols)
        torch.xpu.synchronize()

        host = src.cpu()
        dst = host.to("xpu:1")
        torch.xpu.synchronize()
        back = dst.cpu()
        ident_ok = 1 if torch.equal(host, back) else 0
        if not ident_ok:
            ok_all = 0

        def do_handoff():
            h = src.cpu()
            d = h.to("xpu:1")
            return d

        ho_us = us_bench(do_handoff, warmup=8, iters=20)

        def do_same():
            return src.clone()

        same_us = us_bench(do_same, warmup=8, iters=20)
        bubble = 0.0 if ho_us <= 0 else (ho_us - same_us) / ho_us
        print(
            f"RESULT op=host_handoff name={name} rows={rows} cols={cols} "
            f"bytes={nbytes} us={ho_us:.3f} samecard_us={same_us:.3f} "
            f"bubble={bubble:.3f} GBs={gbs(nbytes, ho_us):.3f} ok={ident_ok}",
            flush=True,
        )

    print(f"VERDICT_LINE ok_all={ok_all} path=host_staged p2p=off", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
