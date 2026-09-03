#!/usr/bin/env python3
"""K7: Qwen3.8-27B GDN fused-recurrent decode update incumbent.

Backend pytorch-xpu on sycl+l0. Rank us. No serve.
S_t = alpha * (S - beta * v_old k^T) + beta * v k^T; o = S q.
"""
from __future__ import annotations

import os
import time

import torch

NV = 48
DK = 128
DV = 128


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
        "op=gdn_delta_recurrent dtype=bf16",
        "n_v_heads", NV, "dk", DK, "dv", DV,
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    if not torch.xpu.is_available():
        print("NO_XPU", flush=True)
        return 2
    s = torch.randn(NV, DV, DK, dtype=torch.bfloat16, device="xpu")
    q = torch.randn(NV, DK, 1, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(NV, DK, 1, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(NV, DV, 1, dtype=torch.bfloat16, device="xpu")
    alpha = torch.rand(NV, 1, 1, dtype=torch.bfloat16, device="xpu") * 0.5 + 0.25
    beta = torch.rand(NV, 1, 1, dtype=torch.bfloat16, device="xpu") * 0.5 + 0.25

    def fn():
        vold = torch.bmm(s, k)
        kt = k.transpose(1, 2)
        s2 = alpha * (s - beta * torch.bmm(vold, kt)) + beta * torch.bmm(v, kt)
        o = torch.bmm(s2, q)
        return s2, o

    y_s, y_o = fn()
    torch.xpu.synchronize()
    ok = int(y_s.shape == s.shape and y_o.shape == (NV, DV, 1))
    us = us_bench(fn, 10, 40, 512)
    state_b = NV * DV * DK * 2
    nbytes = state_b * 2 + (q.numel() + k.numel() + v.numel()) * 2
    gbs = (nbytes / 1e9) / (us * 1e-6) if us > 0 else 0.0
    print("arm,heads,dv,dk,us,state_bytes,bytes,GBs,ok", flush=True)
    print(
        f"delta_recurrent,{NV},{DV},{DK},{us:.3f},{state_b},{nbytes},{gbs:.2f},{ok}",
        flush=True,
    )
    print(
        "NOTE state_bf16_per_layer_MiB", round(state_b / (1 << 20), 3),
        "x48_gdn_layers_MiB", round(48 * state_b / (1 << 20), 3),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
