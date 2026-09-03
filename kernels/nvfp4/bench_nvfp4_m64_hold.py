#!/usr/bin/env python3
"""Held-clock oneDNN nvfp4_gemm_w4a16, M=64 5120. pytorch-xpu on sycl+l0.

Spin M=64 then timed M=64 at named 2800. A is bf16, not s8. No E2M1 cosine.
Rank us_bench vs W8A8 46, s8 75, compose 68.7, LUT 331.6.
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
    spin = int(os.environ.get("NVFP4_SPIN") or "1000")
    m = 64
    n = int(os.environ.get("NVFP4_N") or "5120")
    k = int(os.environ.get("NVFP4_K") or "5120")
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=nvfp4_gemm_w4a16",
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        f"spin={spin} time=M{m} n={n} k={k}",
        flush=True,
    )
    so = os.environ.get("B70_XPU_C_SO")
    print("B70_XPU_C_SO", so, flush=True)
    if so and os.path.exists(so):
        try:
            torch.ops.load_library(so)
            print("load_library", so, "ok", flush=True)
        except Exception as e:
            print("load_library", so, type(e).__name__, str(e)[:400], flush=True)
    else:
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
            print("imported vllm_xpu_kernels._xpu_C", flush=True)
        except Exception as e:
            print("import _xpu_C", type(e).__name__, e, flush=True)
    has = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16")
    has_f8 = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale")
    print("HAS nvfp4_gemm_w4a16", has, "HAS f8scale", has_f8, flush=True)
    if not has and not has_f8:
        print("NVFP4_OP_ABSENT", flush=True)
        print("DONE", flush=True)
        return 0
    gs = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="xpu")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="xpu").t()
    print("B_shape", tuple(w.shape), "B_stride", tuple(w.stride()), flush=True)
    if has:
        scale = torch.ones((k // gs, n), dtype=torch.bfloat16, device="xpu") * 0.02

        def go():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16(x, w, None, scale, gs)

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
        print(f"nvfp4_gemm_w4a16,{m},{n},{k},{us:.3f}", flush=True)
    if has_f8:
        bscale = torch.ones((k // gs, n), dtype=torch.float8_e4m3fn, device="xpu")
        gscale = torch.ones(1, dtype=torch.float32, device="xpu") * 0.02

        def go_f8():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                x, w, None, bscale, gscale, gs)

        y = go_f8()
        torch.xpu.synchronize()
        print("f8_out_shape", tuple(y.shape), "dtype", y.dtype, flush=True)
        for i in range(spin):
            go_f8()
            if (i % 256) == 255:
                torch.xpu.synchronize()
        torch.xpu.synchronize()
        act_s, cur_s, th_s = sample_gt(card)
        print(f"f8_spin_done n={spin} act={act_s} cur={cur_s} throttle={th_s}",
              flush=True)
        act0, cur0, th0 = sample_gt(card)
        print(f"f8_timed_begin act={act0} cur={cur0} throttle={th0}", flush=True)
        us = us_bench(go_f8, 20, 30)
        act1, cur1, th1 = sample_gt(card)
        print(f"f8_timed_end act={act1} cur={cur1} throttle={th1}", flush=True)
        print(f"nvfp4_gemm_w4a16_f8scale,{m},{n},{k},{us:.3f}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
