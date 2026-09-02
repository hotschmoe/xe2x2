# K2 ESIMD DPAS IGC / ocloc encoding 2026-09-02r

Backend sycl+l0. Standalone icpx 2026.1.1 AOT intel_gpu_bmg_g31.
Runtime IGC_ShaderDumpEnable on both cards dumped only HardwareCaps
+ SIPKernelDump.bin (AOT zebin already in the ELF; IGC does not
recompile the user kernel). Encoding is from ocloc disasm of the
unbundled `sycl-spir64_gen` zebin (CPU, same binary both cards).

Tiny GPU tiles (8x16xK) during the dump attempt: max_abs=0 on
card0 s8/s2xs8 and card1 s4/s2.

zebin: has_dpas true, grf_count 128, simd_size 1 (ESIMD).
Loads: `send.ugm load_block2d.ugm.d8` (A) + `d8v` (B, Transformed).

| arm | IGA inner loop (one per K-chunk) | src1 B | src2 A |
|---|---|---|---|
| s8xs8 | `dpas.8x8 (16\|M0) r24:d r24:d r15:b r11.0:b` | `:b` | `:b` |
| s4xs4 | `dpas.8x8 (16\|M0) r24:d r24:d r15:s4 r11.0:s4` | `:s4` | `:s4` |
| s2xs2 | `dpas.8x8 (16\|M0) r18:d r18:d r14:s2 r11.0:s2` | `:s2` | `:s2` |
| s2xs8 | `dpas.8x8 (16\|M0) r18:d r18:d r15:s2 r11.0:b` | `:s2` | `:b` |

IGA prints s8 as `:b` (byte), not `:s8`. Acc is `:d` (s32).
s2xs8 matches arXiv 2508.06753 GEN form `rW:s2 rA:b`.
s8 vs s4 is the operand type, not a different opcode family.

B load bytes (comment rd): s8 d8v rd:8, s4 d8v rd:8, s2 d8v rd:4,
s2xs8 d8v rd:2. Packed s2 B is the small send.

Full asm: `results/k2/ocloc_aot/{s8,s4,s2,s2xs8}/dis/`.
