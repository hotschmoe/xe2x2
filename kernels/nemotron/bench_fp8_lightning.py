#!/usr/bin/env python3
"""K8: oneDNN fp8_gemm_w8a16 at Lightning mamba in/out shapes.

Official PTQ is FP8 on mamba in_proj/out_proj. Dump then beat.
Backend pytorch-xpu on sycl+l0. Hang/absent is a RESULT.
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


def try_shape(m: int, n: int, k: int, has: bool) -> None:
    print("shape", m, n, k, flush=True)
    if not has:
        print("FP8_OP_ABSENT", flush=True)
        return
    a = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    b = torch.randn(k, n, device="cpu").to(torch.float8_e4m3fn).to("xpu").contiguous()
    s = torch.ones(1, n, dtype=torch.float32, device="xpu")

    def go():
        return torch.ops._xpu_C.fp8_gemm_w8a16(a, b, s, None)

    try:
        y = go()
        torch.xpu.synchronize()
        print("out", tuple(y.shape), y.dtype, flush=True)
        us = us_bench(go, 10, 20)
        bytes_w = k * n
        gbs = (bytes_w / 1e9) / (us * 1e-6)
        print(f"fp8_gemm_w8a16,{m},{n},{k},{us:.3f},{gbs:.3f}", flush=True)
    except Exception as e:
        print("fp8_gemm_w8a16 EXC", type(e).__name__, str(e)[:300], flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=10304)
    p.add_argument("--k", type=int, default=2688)
    p.add_argument("--name", default="lightning_mamba")
    args = p.parse_args()
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=fp8_gemm_w8a16",
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
    has = hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")
    print("HAS fp8_gemm_w8a16", has, flush=True)
    try_shape(args.m, args.n, args.k, has)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
