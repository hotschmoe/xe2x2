#!/usr/bin/env python3
"""Nibble histogram of a real NVFP4 packed weight. CPU only. ASCII."""
from __future__ import annotations

import json
import os
import struct
import sys

PATH = os.environ.get(
    "NVFP4_ST",
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/nvfp4-radixark/model-00001-of-00003.safetensors",
)

MAG2 = [0, 1, 2, 3, 4, 6, 8, 12]


def nibble_to_q(nib: int) -> int:
    q = MAG2[nib & 7]
    return -q if (nib & 8) else q


def main() -> int:
    print("CONFIG hist_nvfp4 path", PATH, flush=True)
    with open(PATH, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
        data0 = 8 + hlen
        tensors = [(k, v) for k, v in meta.items() if isinstance(v, dict) and "data_offsets" in v]
        packed = []
        for name, info in tensors:
            dtype = str(info.get("dtype", ""))
            shape = info.get("shape") or []
            if "U8" in dtype.upper() or dtype in ("UINT8", "U8"):
                packed.append((name, info, shape))
        print("n_tensors", len(tensors), "u8_tensors", len(packed), flush=True)
        if not packed:
            for name, info, shape in [(k, v, v.get("shape")) for k, v in meta.items() if isinstance(v, dict)][:12]:
                print("sample", name, info.get("dtype"), info.get("shape"), flush=True)
            print("NO_U8_WEIGHT", flush=True)
            return 1
        want = []
        for name, info, shape in packed:
            lname = name.lower()
            if "mlp" in lname and any(x in lname for x in ("down_proj", "up_proj", "gate_proj")):
                if "weight_scale" in lname or "scale" in lname and "weight" not in lname:
                    continue
                want.append((name, info, shape))
        if not want:
            want = packed[:8]
        print("ffn_u8_in_shard", len(want), flush=True)

        def hist_bytes(raw: bytes):
            counts = [0] * 16
            overflow = 0
            n_nib = 0
            for b in raw:
                lo = b & 15
                hi = (b >> 4) & 15
                for nib in (lo, hi):
                    counts[nib] += 1
                    n_nib += 1
                    q = nibble_to_q(nib)
                    if q >= 8 or q <= -8:
                        overflow += 1
            return counts, overflow, n_nib

        shown = 0
        for name, info, shape in want:
            if shown >= 8:
                break
            off0, off1 = info["data_offsets"]
            nbytes = off1 - off0
            f.seek(data0 + off0)
            raw = f.read(nbytes)
            counts, overflow, n_nib = hist_bytes(raw)
            ov_frac = overflow / max(n_nib, 1)
            print(
                "pick", name, "dtype", info.get("dtype"), "shape", shape,
                "bytes", nbytes, "nibbles", n_nib,
                "overflow_pm", round(1000.0 * ov_frac),
                "ov_frac", "%.4f" % ov_frac,
                flush=True,
            )
            print("hist", " ".join(str(c) for c in counts), flush=True)
            print("frac", " ".join("%.4f" % (c / max(n_nib, 1)) for c in counts), flush=True)
            shown += 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
