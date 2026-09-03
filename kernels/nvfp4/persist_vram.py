#!/usr/bin/env python3
"""Idea 11: resident 4-bit vs load-time s8 VRAM envelope. CPU only."""
from __future__ import annotations

import json
import os
import struct

DIR = os.environ.get(
    "NVFP4_DIR",
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/nvfp4-radixark",
)


def main() -> int:
    print("CONFIG persist_vram cpu-only dir", DIR, flush=True)
    sts = sorted(
        os.path.join(DIR, n)
        for n in os.listdir(DIR)
        if n.endswith(".safetensors")
    )
    u8 = 0
    bf16 = 0
    f8 = 0
    other = 0
    n_u8 = n_other = 0
    for path in sts:
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(hlen))
        for name, info in meta.items():
            if not isinstance(info, dict) or "data_offsets" not in info:
                continue
            off0, off1 = info["data_offsets"]
            nbytes = off1 - off0
            dtype = str(info.get("dtype", "")).upper()
            if "U8" in dtype or dtype in ("UINT8", "U8"):
                u8 += nbytes
                n_u8 += 1
            elif "BF16" in dtype or "F16" in dtype:
                bf16 += nbytes
            elif "F8" in dtype or "E4M3" in dtype or "E5M2" in dtype:
                f8 += nbytes
            else:
                other += nbytes
                n_other += 1
    # Packed NVFP4 weights are 2 nibbles/byte. Load-time s8 is 2x those bytes.
    s8_unpack = u8 * 2
    card = 30.3 * (1 << 30)
    print("shards", len(sts), "u8_tensors", n_u8, flush=True)
    print("bytes_u8_packed", u8, "GiB", round(u8 / (1 << 30), 3), flush=True)
    print("bytes_s8_unpack", s8_unpack, "GiB", round(s8_unpack / (1 << 30), 3), flush=True)
    print("bytes_f8", f8, "GiB", round(f8 / (1 << 30), 3), flush=True)
    print("bytes_bf16", bf16, "GiB", round(bf16 / (1 << 30), 3), flush=True)
    print("bytes_other", other, "GiB", round(other / (1 << 30), 3), flush=True)
    resident = u8 + f8 + bf16 + other
    persist = s8_unpack + f8 + bf16 + other
    print("resident_4bit_plus_rest_GiB", round(resident / (1 << 30), 3), flush=True)
    print("persist_s8_plus_rest_GiB", round(persist / (1 << 30), 3), flush=True)
    print("card_GiB", 30.3, "persist_fits", persist < card, "resident_fits", resident < card, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
