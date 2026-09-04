#!/usr/bin/env python3
"""K8: Lightning GQA 32q/2kv d=128 decode vs Mamba SSU us.

Not a flash-attn campaign. Six layers, 2 KV heads. Rank us.
Backend pytorch-xpu on sycl+l0. ABSENT/EXC is a RESULT.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=int, default=1, help="KV length")
    p.add_argument("--name", default="lightning_gqa")
    args = p.parse_args()
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=sdpa",
        "arm", args.name, "q_heads=32 kv_heads=2 d=128 t", args.t,
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if not torch.xpu.is_available():
        print("NO_XPU", flush=True)
        return 2
    hq, hkv, d, t = 32, 2, 128, args.t
    q = torch.randn(1, hq, 1, d, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(1, hkv, t, d, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(1, hkv, t, d, dtype=torch.bfloat16, device="xpu")

    def go():
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    try:
        y = go()
        torch.xpu.synchronize()
        print("out", tuple(y.shape), y.dtype, flush=True)
        us = us_bench(go, 20, 40)
        print(f"sdpa_gqa,t={t},us={us:.3f}", flush=True)
        print("SSU_FLOOR_US 80.064", flush=True)
    except Exception as e:
        print("SDPA EXC", type(e).__name__, str(e)[:300], flush=True)
        return 0
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
