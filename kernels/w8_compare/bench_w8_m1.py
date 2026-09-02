#!/usr/bin/env python3
"""W8A8 M=1 after a M=64 occupancy burst. Matched to dpas_s8_sc hold."""
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


def main() -> int:
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=int8_gemm_w8a8",
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        "heat=M64 then time M1",
        flush=True,
    )
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except Exception as e:
        print("import _xpu_C", type(e).__name__, e, flush=True)
        return 2
    k = n = 5120

    def make(m: int):
        a = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="xpu")
        b = torch.randint(-64, 64, (k, n), dtype=torch.int8, device="xpu")
        a_s = torch.full((m, 1), 0.02, dtype=torch.float16, device="xpu")
        b_s = torch.full((1, n), 0.02, dtype=torch.float16, device="xpu")
        return a, a_s, b, b_s

    ah, ahs, bh, bhs = make(64)
    for _ in range(40):
        torch.ops._xpu_C.int8_gemm_w8a8(ah, ahs, bh, bhs, torch.float16, None)
    torch.xpu.synchronize()
    print("heat_m64_done", flush=True)

    a, a_s, b, b_s = make(1)
    y = torch.ops._xpu_C.int8_gemm_w8a8(a, a_s, b, b_s, torch.float16, None)
    torch.xpu.synchronize()
    ref = (a.cpu().float() * a_s.cpu().float()) @ (
        b.cpu().float() * b_s.cpu().float()
    )
    us = us_bench(
        lambda: torch.ops._xpu_C.int8_gemm_w8a8(
            a, a_s, b, b_s, torch.float16, None
        ),
        30,
        40,
    )
    aa = y.float().reshape(-1).cpu()
    bb = ref.float().reshape(-1).cpu()
    cos = float(F.cosine_similarity(aa, bb, dim=0))
    mx = float((aa - bb).abs().max())
    gbs = (k * n / 1e9) / (us * 1e-6)
    print(
        f"int8_gemm_w8a8,1,{n},{k},{us:.3f},{gbs:.3f},{cos:.6f},{mx:.5g}",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
