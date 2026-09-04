#!/usr/bin/env python3
"""K8: oneDNN nvfp4_gemm_w4a16 at Lightning expert shapes.

Backend pytorch-xpu on sycl+l0. Dump then beat. Hang/absent is a RESULT.
"""
from __future__ import annotations

import argparse
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


def try_shape(m: int, n: int, k: int, has: bool, has_f8: bool) -> None:
    gs = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="xpu").t()
    print("shape", m, n, k, "B", tuple(w.shape), "stride", tuple(w.stride()), flush=True)
    if has:
        scale = torch.ones((k // gs, n), dtype=torch.bfloat16, device="xpu") * 0.02

        def go():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16(x, w, None, scale, gs)

        try:
            y = go()
            torch.xpu.synchronize()
            print("out", tuple(y.shape), y.dtype, flush=True)
            us = us_bench(go, 10, 20)
            print(f"nvfp4_gemm_w4a16,{m},{n},{k},{us:.3f}", flush=True)
        except Exception as e:
            print("nvfp4_gemm_w4a16 EXC", type(e).__name__, str(e)[:300], flush=True)
    elif has_f8:
        print("HAS_F8SCALE_ONLY skip timed", flush=True)
    else:
        print("NVFP4_OP_ABSENT", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=1856)
    p.add_argument("--k", type=int, default=2688)
    p.add_argument("--name", default="lightning_moe")
    args = p.parse_args()
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=nvfp4_gemm_w4a16",
        "arm", args.name, "m", args.m, "n", args.n, "k", args.k,
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    so = os.environ.get("B70_XPU_C_SO")
    if so and os.path.exists(so):
        try:
            torch.ops.load_library(so)
            print("load_library", so, "ok", flush=True)
        except Exception as e:
            print("load_library", type(e).__name__, str(e)[:200], flush=True)
    else:
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
            print("imported vllm_xpu_kernels._xpu_C", flush=True)
        except Exception as e:
            print("import _xpu_C", type(e).__name__, e, flush=True)
    has = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16")
    has_f8 = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale")
    print("HAS nvfp4_gemm_w4a16", has, "HAS f8scale", has_f8, flush=True)
    try_shape(args.m, args.n, args.k, has, has_f8)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
