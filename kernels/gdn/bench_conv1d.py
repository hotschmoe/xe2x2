#!/usr/bin/env python3
"""K7: Qwen3.8-27B GDN depthwise conv1d K=4 incumbent.

Backend pytorch-xpu on sycl+l0. Rank us. No serve.
Dims from bf16 config.json text_config.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

H = 5120
KEY_DIM = 16 * 128  # linear_num_key_heads * linear_key_head_dim
VAL_DIM = 48 * 128  # linear_num_value_heads * linear_value_head_dim
KCONV = 4


def us_bench(fn, warmup: int, iters: int, spin: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    for _ in range(spin):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    print(
        "CONFIG backend=pytorch-xpu sitting on sycl+l0",
        "op=depthwise_conv1d_k4 dtype=bf16",
        "model=qwen3.8-27b H", H, "key_dim", KEY_DIM, "val_dim", VAL_DIM,
        "conv_k", KCONV,
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if not torch.xpu.is_available():
        print("NO_XPU", flush=True)
        return 2
    print(
        "INVENTORY layers=64 full_attention_interval=4 gdn_layers=48",
        "full_attn_layers=16",
        flush=True,
    )
    print("arm,channels,t,us,bytes,GBs,ok", flush=True)
    warmup, iters, spin = 10, 40, 512
    for name, c in (("q", KEY_DIM), ("k", KEY_DIM), ("v", VAL_DIM)):
        w = torch.randn(c, 1, KCONV, dtype=torch.bfloat16, device="xpu")
        for t in (1, 64, 256):
            x = torch.randn(1, c, t, dtype=torch.bfloat16, device="xpu")

            def fn(x=x, w=w, c=c):
                xpad = F.pad(x, (KCONV - 1, 0))
                return F.conv1d(xpad, w, groups=c)

            y = fn()
            torch.xpu.synchronize()
            ok = int(y.shape == (1, c, t))
            us = us_bench(fn, warmup, iters, spin)
            nbytes = x.numel() * 2 + w.numel() * 2 + y.numel() * 2
            gbs = (nbytes / 1e9) / (us * 1e-6) if us > 0 else 0.0
            print(
                f"conv1d_{name},{c},{t},{us:.3f},{nbytes},{gbs:.2f},{ok}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
