#!/usr/bin/env python3
"""CPU IGA Xe2 disasm of oneDNN ngen kernel bins. No GPU."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

LIB = (
    "/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/"
    "oneapi-root/opt/intel/oneapi/debugger/2026.1/opt/debugger/lib/libiga64.so"
)
IGA_XE2 = (2 << 24) | 0


class CtxOpts(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_size_t), ("gen", ctypes.c_uint32)]


class DisOpts(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("formatting_opts", ctypes.c_uint32),
        ("_reserved0", ctypes.c_uint32),
        ("_reserved1", ctypes.c_uint32),
        ("decoder_opts", ctypes.c_uint32),
        ("base_pc_offset", ctypes.c_uint32),
    ]


def disasm_file(bin_path: Path, out_path: Path) -> int:
    lib = ctypes.CDLL(LIB)
    lib.iga_status_to_string.restype = ctypes.c_char_p
    lib.iga_context_create.argtypes = [
        ctypes.POINTER(CtxOpts),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.iga_disassemble.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(DisOpts),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    data = bin_path.read_bytes()
    copts = CtxOpts(ctypes.sizeof(CtxOpts), IGA_XE2)
    ctx = ctypes.c_void_p()
    st = lib.iga_context_create(ctypes.byref(copts), ctypes.byref(ctx))
    if st != 0:
        print("iga_context_create", st, lib.iga_status_to_string(st))
        return 2
    dopts = DisOpts(ctypes.sizeof(DisOpts), 0x08 | 0x10, 0, 0, 0, 0)  # PC+bits
    buf = ctypes.create_string_buffer(data)
    text = ctypes.c_char_p()
    st = lib.iga_disassemble(
        ctx,
        ctypes.byref(dopts),
        buf,
        ctypes.c_uint32(len(data)),
        None,
        None,
        ctypes.byref(text),
    )
    if st != 0:
        print("iga_disassemble", bin_path, st, lib.iga_status_to_string(st))
        if not text or not text.value:
            return 3
    out = text.value.decode("ascii", "replace") if text.value else ""
    try:
        out_path.write_text(out)
    except PermissionError:
        alt = Path("/tmp") / (out_path.name)
        alt.write_text(out)
        out_path = alt
    dpas = [ln for ln in out.splitlines() if "dpas" in ln.lower()]
    print(
        f"WROTE {out_path} bytes={len(out)} lines={out.count(chr(10))} "
        f"dpas_lines={len(dpas)}"
    )
    for ln in dpas[:8]:
        print("  ", ln[:160])
    if len(dpas) > 8:
        print(f"  ... {len(dpas) - 8} more dpas lines")
    return 0 if st == 0 else 4


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: iga_disasm.py file.bin [file.bin ...]")
        return 2
    rc = 0
    for a in sys.argv[1:]:
        p = Path(a)
        rc |= disasm_file(p, Path(str(p) + ".xe2.asm"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
