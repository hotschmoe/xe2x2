#!/usr/bin/env python3
"""K4: fp8_gemm_w8a16 vs int8_gemm_w8a8 GEMM-only at Qwen3.8 shapes.

Backend pytorch-xpu on sycl+l0. Rank us. Cosine vs host reference.
int8_gemm_w8a16 in the w8a16-tagged image is scalar ref_matmul;
do not use it as the XMX arm. Cq (quant launches) is a later row.
"""
from __future__ import annotations

import argparse
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


def cosine(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a = y.float().reshape(-1).cpu()
    b = ref.float().reshape(-1).cpu()
    cos = float(F.cosine_similarity(a, b, dim=0))
    mx = float((a - b).abs().max())
    return cos, mx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["fp8", "int8a8"], required=True)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    print(
        "CONFIG backend=pytorch-xpu sitting on sycl+l0",
        "op", args.op,
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if not torch.xpu.is_available():
        print("NO_XPU", flush=True)
        return 2
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except Exception as e:
        print("import _xpu_C", type(e).__name__, e, flush=True)

    shapes = [
        (1, 5120, 5120),
        (1, 17408, 5120),
        (2, 5120, 5120),
        (4, 5120, 5120),
        (64, 5120, 5120),
        (256, 5120, 5120),
        (1024, 5120, 5120),
    ]
    print("arm,m,n,k,us,GBs,TOPS,cosine,max_abs,ok", flush=True)
    for m, n, k in shapes:
        if args.op == "fp8":
            if not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
                print("MISSING fp8_gemm_w8a16", flush=True)
                return 2
            a = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
            b_cpu = torch.randn(k, n, device="cpu").to(torch.float8_e4m3fn)
            b = b_cpu.to("xpu").contiguous()
            s = torch.ones(1, n, dtype=torch.float32, device="xpu")
            y = torch.ops._xpu_C.fp8_gemm_w8a16(a, b, s, None)
            torch.xpu.synchronize()
            ref = a.cpu().float() @ b_cpu.float()
            us = us_bench(
                lambda: torch.ops._xpu_C.fp8_gemm_w8a16(a, b, s, None),
                args.warmup,
                args.iters,
            )
            arm = "fp8_gemm_w8a16"
        else:
            if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
                print("MISSING int8_gemm_w8a8", flush=True)
                return 2
            a = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="xpu")
            b = torch.randint(-64, 64, (k, n), dtype=torch.int8, device="xpu")
            a_s = torch.full((m, 1), 0.02, dtype=torch.float16, device="xpu")
            b_s = torch.full((1, n), 0.02, dtype=torch.float16, device="xpu")
            y = torch.ops._xpu_C.int8_gemm_w8a8(
                a, a_s, b, b_s, torch.float16, None
            )
            torch.xpu.synchronize()
            ref = (a.cpu().float() * a_s.cpu().float()) @ (
                b.cpu().float() * b_s.cpu().float()
            )
            us = us_bench(
                lambda: torch.ops._xpu_C.int8_gemm_w8a8(
                    a, a_s, b, b_s, torch.float16, None
                ),
                args.warmup,
                args.iters,
            )
            arm = "int8_gemm_w8a8"
        cos, mx = cosine(y, ref)
        ok = 1 if cos > 0.99 else 0
        gbs = (k * n / 1e9) / (us * 1e-6)
        tops = (2.0 * m * n * k / 1e12) / (us * 1e-6)
        print(
            f"{arm},{m},{n},{k},{us:.3f},{gbs:.3f},{tops:.4f},"
            f"{cos:.6f},{mx:.5g},{ok}",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
