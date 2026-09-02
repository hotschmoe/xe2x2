# K2 card0 ESIMD DPAS ISA (s8xs8 and s2xs8)

Backend sycl+l0. Standalone AOT intel_gpu_bmg_g31. Card 0 only.
Tiny tile 8x16x32. No compile/runtime refusal.

Runtime IGC_ShaderDumpEnable=1 AOT-skipped: each dump dir had only
HardwareCaps.txt + SIPKernelDump.bin (EUCount=256 SubSlice=32
UsDeviceID=0xe223). No .asm/.visaasm from the GPU run. Fallback:
clang-offload-bundler --unbundle --targets=sycl-spir64_gen, then
ocloc disasm -device bmg-g31 of that zebin, then ocloc compile of
the embedded .spv (-vc-codegen -device bmg-g31) with
IGC_ShaderDumpEnable=1 IGC_ForceDumpAll=1 (CPU, not a second GPU
touch). AOT zebin GEN and recompile GEN match.

## s8xs8

LLVM (already known; immediates now read):
  llvm.genx.dpas2.v128i32.v128i32.v128i32.v64i32(..., i32 8, i32 8, i32 8, i32 8, i32 1, i32 1)
  src1_prec=s8=8, src2_prec=s8=8, systolic=8, repeat=8

VISA inner loop (ocloc_recompile visaasm):
    lsc_load_block2d.ugm (M1, 1)  V113:d8.32x8nn  ...
    lsc_load_block2d.ugm (M1, 1)  V117:d8.16x32nt ...
    dpas.s8.s8.8.8 (M1, 16) V120.0 V120.0 V60.0 V59(0,0)

GEN (AOT zebin ocloc disasm; same line on recompile):
        send.ugm ... // wr:1+0, rd:4; load_block2d.ugm.d8.a64
        send.ugm ... // wr:1+0, rd:8; load_block2d.ugm.d8v.a64
        dpas.8x8 (16|M0)         r24:d         r24:d             r15:b             r11.0:b          {Compacted,$3}

IGA does not emit a `dpas.s8.s8` mnemonic. It is `dpas.8x8` with
rW:b rA:b (byte, not :s8) and acc :d.

Tiny run: check max_abs=0, timed max_abs=0, timed 26.354 us.

## s2xs8

LLVM:
  llvm.genx.dpas2.v128i32.v128i32.v32i32.v64i32(..., i32 4, i32 8, i32 8, i32 8, i32 1, i32 1)
  src1_prec=s2=4, src2_prec=s8=8, systolic=8, repeat=8

VISA inner loop:
    lsc_load_block2d.ugm (M1, 1)  V118:d8.32x8nn  ...
    lsc_load_block2d.ugm (M1, 1)  V123:d8.16x8nt  ...
    dpas.s2.s8.8.8 (M1, 16) V127.0 V127.0 V63.0 V62(0,0)

GEN (AOT zebin ocloc disasm; same line on recompile):
        send.ugm ... // wr:1+0, rd:4; load_block2d.ugm.d8.a64
        send.ugm ... // wr:1+0, rd:2; load_block2d.ugm.d8v.a64
        dpas.8x8 (16|M0)         r18:d         r18:d             r15:s2            r11.0:b          {$3}

Native s2 on src1 (weights). IGA type :s2 vs A-src :b. Not a
decompress loop into s8 DPAS. B block is 16x8 nt / rd:2 vs s8
16x32 nt / rd:8 (4x packed along K).

Tiny run: check max_abs=0, timed max_abs=0, timed 26.250 us.

## Verdict

s8xs8: visa `dpas.s8.s8.8.8` / GEN `dpas.8x8 rW:b rA:b`.
s2xs8: visa `dpas.s2.s8.8.8` / GEN `dpas.8x8 rW:s2 rA:b`.
Tiny 8x16x32 us is launch-dominated; not a MAC ranking.

Dump dirs: results/k2/igc_card0_s8/ (28 files) and
results/k2/igc_card0_s2xs8/ (28 files). Runtime IGC dump empty of
ISA (2 files each); ocloc_disasm/ + ocloc_recompile/ hold the
encoding. Card 0 left free.
