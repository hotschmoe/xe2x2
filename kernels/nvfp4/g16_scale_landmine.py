#!/usr/bin/env python3
"""CPU landmine: s8 DPAS K=32 spans two NVFP4 group-16 scales."""
from __future__ import annotations

import json
import os

MAG2 = [0, 1, 2, 3, 4, 6, 8, 12]
CFG = os.environ.get(
    "NVFP4_CFG",
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/nvfp4-radixark/hf_quant_config.json",
)


def nibble_to_q(nib: int) -> int:
    q = MAG2[nib & 7]
    return -q if (nib & 8) else q


def main() -> int:
    print("CONFIG hist_landmine cpu-only s8_K32 vs NVFP4_g16", flush=True)
    with open(CFG) as f:
        cfg = json.load(f)
    q = cfg.get("quantization") or {}
    print("quant_algo", q.get("quant_algo"), flush=True)
    layers = q.get("quantized_layers") or {}
    nv = mx = fp8 = other = 0
    g16 = g32 = 0
    for name, info in layers.items():
        algo = (info or {}).get("quant_algo")
        gs = (info or {}).get("group_size")
        if algo == "NVFP4":
            nv += 1
            if gs == 16:
                g16 += 1
            elif gs == 32:
                g32 += 1
        elif algo == "MXFP4":
            mx += 1
        elif algo == "FP8":
            fp8 += 1
        else:
            other += 1
    print(
        "layers NVFP4", nv, "MXFP4", mx, "FP8", fp8, "other", other,
        "nv_g16", g16, "nv_g32", g32,
        flush=True,
    )
    # Tiny numeric: A=1, B q nibble cycle, two group scales.
    k = 32
    a = [i + 1 for i in range(k)]
    b = [nibble_to_q(i & 15) for i in range(k)]
    s0, s1 = 0.5, 2.0
    acc_one = sum(a[i] * b[i] for i in range(k)) * s0  # one scale after dpas K32
    acc_two = sum(a[i] * b[i] * (s0 if i < 16 else s1) for i in range(k))
    print("k32_one_scale", acc_one, "k16_two_scale", acc_two, "diff", acc_two - acc_one, flush=True)
    print("LANDMINE s8_dpas_K32_cannot_isolate_g16", abs(acc_two - acc_one) > 1e-9, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
