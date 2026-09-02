# K2 card1 IGC/GEN encoding -- ESIMD s4xs4 and s2xs2

CONFIG backend=sycl+l0 card=1 AOT=intel_gpu_bmg_g31 ocloc=26.22.1
device=bmg-g31 IGC=2.36.3. Tiny tile 8x16x64. Integer s4/s2 only; not NVFP4 E2M1.

Runtime `IGC_ShaderDumpEnable=1` (and `IGC_ForceDumpAll=1`) dumped only
`HardwareCaps.txt` + `SIPKernelDump.bin`. AOT skip: no visa/GEN from JIT.

ISA came from the AOT fat binary: copy to /tmp, `clang-offload-bundler
--type=o --targets=sycl-spir64_gen --unbundle`, then `ocloc disasm
-device bmg-g31`. VISA from `ocloc compile -spirv_input -device bmg-g31`
of the embedded `.spv` with IGC dumps on (CPU, no extra GPU). GEN line
from that recompile matches the AOT zebin disasm.

## s4xs4 inner loop (one dpas per K=64 chunk)

VISA (`results/k2/igc_card1_s4/DpasS4Name.visaasm:241`):

```
    dpas.s4.s4.8.8 (M1, 16) V127.0 V127.0 V63.0 V62(0,0)                         /// $40
```

GEN/IGA (`results/k2/igc_card1_s4/DpasS4Name.zebin.asm:75`):

```
        dpas.8x8 (16|M0)         r24:d         r24:d             r15:s4            r11.0:s4         {$3}
```

Feeds: `load_block2d.ugm.d8.a64` -> r11 :s4 (A), `load_block2d.ugm.d8v.a64`
-> r15 :s4 (B VNNI). Acc/dst `:d` (s32). LLVM immediate pair is `i32 6, i32 6`
on `llvm.genx.dpas2.v128i32.v128i32.v128i32.v64i32` (precision s4, not a
shared s8 opcode with a different type string).

## s2xs2 inner loop (one dpas per K=64 chunk)

VISA (`results/k2/igc_card1_s2/DpasS2Name.visaasm:241`):

```
    dpas.s2.s2.8.8 (M1, 16) V127.0 V127.0 V63.0 V62(0,0)                         /// $40
```

GEN/IGA (`results/k2/igc_card1_s2/DpasS2Name.zebin.asm:75`):

```
        dpas.8x8 (16|M0)         r18:d         r18:d             r14:s2            r11.0:s2         {$3}
```

Feeds: `load_block2d.ugm.d8.a64` -> r11 :s2 (A, 2 GRF),
`load_block2d.ugm.d8v.a64` -> r14 :s2 (B VNNI, 4 GRF). Acc/dst `:d`.
LLVM immediate pair is `i32 4, i32 4` on
`llvm.genx.dpas2.v128i32.v128i32.v64i32.v32i32`.

## Reading

Both arms are systolic depth 8, repeat 8, exec size 16. Precision is
the VISA suffix (`dpas.s4.s4.8.8` vs `dpas.s2.s2.8.8`) and the IGA
src types (`:s4` vs `:s2`), not a bare `dpas.s4.s4` without .8.8.
`has_dpas: true` in zebin `ze_info`. No `:e2m1` / `:f4` on these
kernels. Tiny-run numeric: max_abs=0 both arms.

Dump file counts after ocloc land: 10 files each in
`results/k2/igc_card1_s4/` and `results/k2/igc_card1_s2/`.
Runtime-only dump was 2 files (empty of ISA).
