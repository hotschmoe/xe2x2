#!/usr/bin/env python3
"""Held-clock oneDNN int8_gemm_w8a8, M=256 N=17408 K=5120. pytorch-xpu on sycl+l0.

Spin M=256 then timed M=256 at named 2800. Rank us_bench vs w4a16 394,
s8 469.8, s4 140.0, W8A8 square 75. Oracle after timed.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F


def us_bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def read_int(path: str) -> int:
    try:
        with open(path, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def sample_gt(card: int) -> tuple[int, int, int]:
    base = f"/sys/class/drm/card{card}/device/tile0/gt0/freq0"
    return (
        read_int(f"{base}/act_freq"),
        read_int(f"{base}/cur_freq"),
        read_int(f"{base}/throttle/status"),
    )


def main() -> int:
    card = int(os.environ.get("ZE_AFFINITY_MASK") or "0")
    spin = int(os.environ.get("W8_SPIN") or "512")
    m = 256
    n = int(os.environ.get("W8_N") or "17408")
    k = int(os.environ.get("W8_K") or "5120")
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=int8_gemm_w8a8",
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        f"spin={spin} time=M{m} n={n} k={k}",
        flush=True,
    )
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
        print("imported vllm_xpu_kernels._xpu_C", flush=True)
    except Exception as e:
        print("import _xpu_C", type(e).__name__, e, flush=True)
        return 2
    if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
        print("MISSING int8_gemm_w8a8", flush=True)
        return 2
    a = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="xpu")
    b = torch.randint(-64, 64, (k, n), dtype=torch.int8, device="xpu").contiguous()
    a_s = torch.full((m, 1), 0.02, dtype=torch.float16, device="xpu")
    b_s = torch.full((1, n), 0.02, dtype=torch.float16, device="xpu")

    def go():
        return torch.ops._xpu_C.int8_gemm_w8a8(
            a, a_s, b, b_s, torch.float16, None
        )

    y = go()
    torch.xpu.synchronize()
    print("out_shape", tuple(y.shape), "dtype", y.dtype, flush=True)
    for i in range(spin):
        go()
        if (i % 256) == 255:
            torch.xpu.synchronize()
    torch.xpu.synchronize()
    act_s, cur_s, th_s = sample_gt(card)
    print(f"spin_done n={spin} act={act_s} cur={cur_s} throttle={th_s}",
          flush=True)
    act0, cur0, th0 = sample_gt(card)
    print(f"timed_begin act={act0} cur={cur0} throttle={th0}", flush=True)
    us = us_bench(go, 20, 30)
    act1, cur1, th1 = sample_gt(card)
    print(f"timed_end act={act1} cur={cur1} throttle={th1}", flush=True)
    y = go()
    torch.xpu.synchronize()
    ref = (a.cpu().float() * a_s.cpu().float()) @ (
        b.cpu().float() * b_s.cpu().float()
    )
    aa = y.float().reshape(-1).cpu()
    bb = ref.float().reshape(-1).cpu()
    cos = float(F.cosine_similarity(aa, bb, dim=0))
    mx = float((aa - bb).abs().max())
    gbs = (k * n / 1e9) / (us * 1e-6)
    print(
        f"int8_gemm_w8a8,{m},{n},{k},{us:.3f},{gbs:.3f},{cos:.6f},{mx:.5g}",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
