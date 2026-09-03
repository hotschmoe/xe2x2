#!/usr/bin/env python3
"""oneDNN nvfp4_gemm_w4a16 incumbent, M=1 5120. pytorch-xpu on sycl+l0."""
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


def main() -> int:
    print(
        "CONFIG backend=pytorch-xpu on sycl+l0 op=nvfp4_gemm_w4a16",
        "torch", torch.__version__,
        "ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"),
        flush=True,
    )
    so = os.environ.get("B70_XPU_C_SO")
    print("B70_XPU_C_SO", so, flush=True)
    if so and os.path.exists(so):
        # Do not import the image _xpu_C first: TORCH_LIBRARY(_xpu_C) cannot
        # register twice (stock image lacks nvfp4; v028 so has it).
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
    names = []
    try:
        names = [x for x in dir(torch.ops._xpu_C) if "nvfp4" in x.lower() or "fp4" in x.lower() or "gemm" in x.lower()]
    except Exception as e:
        print("dir _xpu_C", type(e).__name__, e, flush=True)
    print("GEMM_OPS", " ".join(names[:40]), flush=True)
    has = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16")
    has_f8 = hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale")
    print("HAS nvfp4_gemm_w4a16", has, "HAS f8scale", has_f8, flush=True)
    if not has and not has_f8:
        print("NVFP4_OP_ABSENT", flush=True)
        print("DONE", flush=True)
        return 0
    n = k = 5120
    gs = 16
    x = torch.randn(1, k, dtype=torch.bfloat16, device="xpu")
    # Packed NT: shape [K/2, N] with stride(0)==1 (K/2 contiguous).
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="xpu").t()
    print("B_shape", tuple(w.shape), "B_stride", tuple(w.stride()), flush=True)
    if has:
        scale = torch.ones((k // gs, n), dtype=torch.bfloat16, device="xpu") * 0.02
        def go():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16(x, w, None, scale, gs)
        try:
            y = go()
            torch.xpu.synchronize()
            print("out_shape", tuple(y.shape), "dtype", y.dtype, flush=True)
            x64 = torch.randn(64, k, dtype=torch.bfloat16, device="xpu")
            def heat():
                return torch.ops._xpu_C.nvfp4_gemm_w4a16(x64, w, None, scale, gs)
            for _ in range(30):
                heat()
            torch.xpu.synchronize()
            print("heat_M64_done", flush=True)
            us = us_bench(go, 30, 40)
            print(f"nvfp4_gemm_w4a16,1,{n},{k},{us:.3f}", flush=True)
        except Exception as e:
            print("CALL_EXC", type(e).__name__, str(e)[:400], flush=True)
            return 1
    if has_f8:
        bscale = torch.ones((k // gs, n), dtype=torch.float8_e4m3fn, device="xpu")
        gscale = torch.ones(1, dtype=torch.float32, device="xpu") * 0.02
        def go_f8():
            return torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(x, w, None, bscale, gscale, gs)
        try:
            y = go_f8()
            torch.xpu.synchronize()
            print("f8_out_shape", tuple(y.shape), "dtype", y.dtype, flush=True)
            us = us_bench(go_f8, 20, 30)
            print(f"nvfp4_gemm_w4a16_f8scale,1,{n},{k},{us:.3f}", flush=True)
        except Exception as e:
            print("F8_CALL_EXC", type(e).__name__, str(e)[:400], flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
