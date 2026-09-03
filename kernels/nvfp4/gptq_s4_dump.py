#!/usr/bin/env python3
"""Unpack GPTQ INT4 qweight to s4 [-8,7], hist, dump a tile for ESIMD.
CPU only. Never E2M1 bitcast. ASCII."""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

CKPT = os.environ.get(
    "GPTQ_ST",
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/"
    "gptq-int4-mtp-bf16-9d189a60/model-00002-of-00005.safetensors",
)
OUT = os.environ.get(
    "GPTQ_DUMP",
    "/mnt/vm_8tb/github/xe2x2/results/k6/gptq_s4_down0_256.bin",
)
OUT_SC = os.environ.get(
    "GPTQ_DUMP_SC",
    "/mnt/vm_8tb/github/xe2x2/results/k6/gptq_s4_down0_256_sc.bin",
)
TILE_K = int(os.environ.get("TILE_K", "256"))
TILE_N = int(os.environ.get("TILE_N", "256"))
GS = 128


def load_meta(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
        data0 = 8 + hlen
    tens = {
        k: v
        for k, v in meta.items()
        if isinstance(v, dict) and "data_offsets" in v
    }
    return tens, data0


def read_tensor(path, data0, info):
    off0, off1 = info["data_offsets"]
    dtype = str(info.get("dtype", "")).upper()
    shape = tuple(info.get("shape") or [])
    with open(path, "rb") as f:
        f.seek(data0 + off0)
        raw = f.read(off1 - off0)
    if "I32" in dtype or dtype in ("INT32", "I32"):
        arr = np.frombuffer(raw, dtype="<i4")
    elif "F16" in dtype or dtype in ("FLOAT16", "F16"):
        arr = np.frombuffer(raw, dtype="<f2")
    else:
        raise SystemExit("bad dtype %s %s" % (dtype, info))
    return arr.reshape(shape)


def unpack_i32_nibbles_last(a):
    """LSB-first 8 nibbles along the last axis. a int32 [..., packed]."""
    parts = [((a >> (s * 4)) & 15).astype(np.int16) for s in range(8)]
    stacked = np.stack(parts, axis=-1)
    new_shape = a.shape[:-1] + (a.shape[-1] * 8,)
    return stacked.reshape(new_shape)


def unpack_i32_nibbles_first(a):
    """LSB-first 8 nibbles along axis 0. a int32 [packed, N]."""
    parts = [((a >> (s * 4)) & 15).astype(np.int16) for s in range(8)]
    stacked = np.stack(parts, axis=1)
    return stacked.reshape(a.shape[0] * 8, a.shape[1])


def hist16(x):
    c = np.bincount(x.ravel().astype(np.int32), minlength=16)
    return c[:16]


def main() -> int:
    print("CONFIG gptq_s4_dump path", CKPT, "gs", GS, flush=True)
    tens, data0 = load_meta(CKPT)
    want = []
    for name, info in tens.items():
        if ".mlp." not in name:
            continue
        if name.endswith(".qweight"):
            want.append(name[: -len(".qweight")])
    want = sorted(want)
    print("mlp_qweight_in_shard", len(want), flush=True)

    dumped = False
    shown = 0
    for prefix in want:
        if shown >= 6:
            break
        qw_i = tens[prefix + ".qweight"]
        qz_i = tens[prefix + ".qzeros"]
        sc_i = tens[prefix + ".scales"]
        gi_i = tens.get(prefix + ".g_idx")
        qw = read_tensor(CKPT, data0, qw_i)
        qz = read_tensor(CKPT, data0, qz_i)
        sc = read_tensor(CKPT, data0, sc_i)
        gi = read_tensor(CKPT, data0, gi_i) if gi_i is not None else None
        # qweight [K/8, N], qzeros [K/gs, N/8], scales [K/gs, N]
        k = int(qw.shape[0]) * 8
        n = int(qw.shape[1])
        u = unpack_i32_nibbles_first(qw)  # [K, N] 0..15
        z = unpack_i32_nibbles_last(qz)  # [K/gs, N]
        hc = hist16(u)
        zc = hist16(z)
        z_uniq = np.unique(z)
        s4 = u.astype(np.int16) - 8
        s4_min = int(s4.min())
        s4_max = int(s4.max())
        ov = int(np.count_nonzero((s4 < -8) | (s4 > 7)))
        zp8 = int(np.count_nonzero(z != 8))
        g_ok = -1
        if gi is not None:
            expect = np.arange(gi.shape[0], dtype=gi.dtype) // GS
            g_ok = int(np.array_equal(gi, expect))
        print(
            "pick",
            prefix,
            "qweight",
            list(qw.shape),
            "qzeros",
            list(qz.shape),
            "scales",
            list(sc.shape),
            "K",
            k,
            "N",
            n,
            "nibble_hist",
            ",".join(str(int(x)) for x in hc),
            "z_uniq",
            ",".join(str(int(x)) for x in z_uniq[:16]),
            "z_ne_8",
            zp8,
            "s4_min",
            s4_min,
            "s4_max",
            s4_max,
            "s4_ov",
            ov,
            "g_idx_linear",
            g_ok,
            flush=True,
        )
        shown += 1
        if (not dumped) and prefix.endswith("down_proj"):
            tk = min(TILE_K, k - (k % 64))
            tn = min(TILE_N, n - (n % 16))
            tile = s4[:tk, :tn].astype(np.int8)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "wb") as f:
                f.write(struct.pack("<II", tk, tn))
                f.write(tile.tobytes(order="C"))
            print(
                "DUMP",
                OUT,
                "k",
                tk,
                "n",
                tn,
                "bytes",
                8 + tile.size,
                "s4_min",
                int(tile.min()),
                "s4_max",
                int(tile.max()),
                flush=True,
            )
            ng = tk // GS
            sc_tile = sc[:ng, :tn].astype("<f2")
            with open(OUT_SC, "wb") as f:
                f.write(struct.pack("<III", tk, tn, GS))
                f.write(tile.tobytes(order="C"))
                f.write(sc_tile.tobytes(order="C"))
            print(
                "DUMP_SC",
                OUT_SC,
                "k",
                tk,
                "n",
                tn,
                "gs",
                GS,
                "ng",
                ng,
                "sc_min",
                float(sc_tile.astype(np.float32).min()),
                "sc_max",
                float(sc_tile.astype(np.float32).max()),
                flush=True,
            )
            dumped = True
    if not dumped:
        print("NO_DUMP", flush=True)
        return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
