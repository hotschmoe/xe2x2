#!/usr/bin/env python3
"""K1: dump + time incumbent _xpu_C GEMMs. Backend pytorch-xpu on sycl+l0.

Synthetic tensors. Rank us. IGC dump is via env IGC_ShaderDumpEnable /
IGC_DumpToCustomDir (set by the launcher). No tok/s.
"""
from __future__ import annotations

import argparse
import os
import sys
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


def list_ops() -> list[str]:
    names = []
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
        print("imported vllm_xpu_kernels._xpu_C", flush=True)
    except Exception as e:
        print("import vllm_xpu_kernels._xpu_C:", type(e).__name__, e, flush=True)
    try:
        ops = torch.ops._xpu_C
        names = sorted([n for n in dir(ops) if not n.startswith("_")])
    except Exception as e:
        print("torch.ops._xpu_C:", type(e).__name__, e, flush=True)
    return names


def time_int8(m: int, n: int, k: int, warmup: int, iters: int) -> None:
    if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a16"):
        print("MISSING int8_gemm_w8a16", flush=True)
        return
    a = torch.randn(m, k, dtype=torch.float16, device="xpu")
    b = torch.randint(-127, 127, (k, n), dtype=torch.int8, device="xpu").contiguous()
    s = torch.ones(n, dtype=torch.float16, device="xpu")
    y = torch.ops._xpu_C.int8_gemm_w8a16(a, b, s, None)
    torch.xpu.synchronize()
    us = us_bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(a, b, s, None), warmup, iters)
    bytes_w = k * n  # s8 weights
    gbs = (bytes_w / 1e9) / (us * 1e-6)
    ops = 2.0 * m * n * k
    tops = (ops / 1e12) / (us * 1e-6)
    print(
        f"int8_gemm_w8a16,m={m},n={n},k={k},us={us:.3f},GBs={gbs:.3f},"
        f"TOPS={tops:.4f},out={tuple(y.shape)},dtype={y.dtype}",
        flush=True,
    )


def time_int8a8(m: int, n: int, k: int, warmup: int, iters: int) -> None:
    if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
        print("MISSING int8_gemm_w8a8", flush=True)
        return
    a = torch.randint(-127, 127, (m, k), dtype=torch.int8, device="xpu")
    a_s = torch.ones(m, 1, dtype=torch.float16, device="xpu")
    b = torch.randint(-127, 127, (k, n), dtype=torch.int8, device="xpu").contiguous()
    b_s = torch.ones(1, n, dtype=torch.float16, device="xpu")
    y = torch.ops._xpu_C.int8_gemm_w8a8(a, a_s, b, b_s, torch.float16, None)
    torch.xpu.synchronize()
    us = us_bench(
        lambda: torch.ops._xpu_C.int8_gemm_w8a8(
            a, a_s, b, b_s, torch.float16, None
        ),
        warmup,
        iters,
    )
    bytes_w = k * n
    gbs = (bytes_w / 1e9) / (us * 1e-6)
    ops = 2.0 * m * n * k
    tops = (ops / 1e12) / (us * 1e-6)
    print(
        f"int8_gemm_w8a8,m={m},n={n},k={k},us={us:.3f},GBs={gbs:.3f},"
        f"TOPS={tops:.4f},out={tuple(y.shape)},dtype={y.dtype}",
        flush=True,
    )


def time_fp8(m: int, n: int, k: int, warmup: int, iters: int) -> None:
    if not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
        print("MISSING fp8_gemm_w8a16", flush=True)
        return
    a = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    b = torch.randn(k, n, device="cpu").to(torch.float8_e4m3fn).to("xpu").contiguous()
    s = torch.ones(1, n, dtype=torch.float32, device="xpu")
    try:
        y = torch.ops._xpu_C.fp8_gemm_w8a16(a, b, s, None)
        torch.xpu.synchronize()
    except Exception as e:
        print("fp8_gemm_w8a16 EXC", type(e).__name__, str(e)[:240], flush=True)
        return
    us = us_bench(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(a, b, s, None), warmup, iters)
    bytes_w = k * n
    gbs = (bytes_w / 1e9) / (us * 1e-6)
    ops = 2.0 * m * n * k
    tops = (ops / 1e12) / (us * 1e-6)
    print(
        f"fp8_gemm_w8a16,m={m},n={n},k={k},us={us:.3f},GBs={gbs:.3f},"
        f"TOPS={tops:.4f},out={tuple(y.shape)},dtype={y.dtype}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["int8", "int8a8", "fp8", "both"], default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    print(
        "CONFIG backend=pytorch-xpu sitting on sycl+l0",
        "torch", torch.__version__,
        "xpu", torch.xpu.is_available(),
        "count", torch.xpu.device_count() if torch.xpu.is_available() else 0,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if not torch.xpu.is_available():
        print("NO_XPU", flush=True)
        return 2
    names = list_ops()
    print("OPS", " ".join(names) if names else "(none)", flush=True)
    for needle in (
        "fp8_gemm_w8a16",
        "int8_gemm_w8a16",
        "int8_gemm_w8a8",
        "nvfp4_gemm_w4a16",
        "fp8_gemm",
    ):
        print(f"HAS {needle}={hasattr(torch.ops._xpu_C, needle)}", flush=True)

    # Qwen3.8-ish. M=1 decode, M=64 prefill-ish.
    shapes = [
        (1, 5120, 5120),
        (1, 17408, 5120),
        (64, 5120, 5120),
        (64, 17408, 5120),
        (256, 5120, 5120),
    ]
    for m, n, k in shapes:
        print(f"# shape M={m} N={n} K={k}", flush=True)
        if args.op in ("int8", "both"):
            try:
                time_int8(m, n, k, args.warmup, args.iters)
            except Exception as e:
                print("int8 EXC", type(e).__name__, str(e)[:240], flush=True)
        if args.op in ("int8a8", "both"):
            try:
                time_int8a8(m, n, k, args.warmup, args.iters)
            except Exception as e:
                print("int8a8 EXC", type(e).__name__, str(e)[:240], flush=True)
        if args.op in ("fp8", "both"):
            try:
                time_fp8(m, n, k, args.warmup, args.iters)
            except Exception as e:
                print("fp8 EXC", type(e).__name__, str(e)[:240], flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
