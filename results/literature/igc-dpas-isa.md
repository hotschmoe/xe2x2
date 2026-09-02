# IGC DPAS ISA (visa DPAS.md)

Status: LANDED.
Companion to the ESIMD API notes. This is the compiler's instruction spec, not a B70 measurement.

## Source

https://raw.githubusercontent.com/intel/intel-graphics-compiler/master/documentation/visa/instructions/DPAS.md

Opcode 0x83. Text form:

```
DPAS.W.A.SD.RC    (Exec_size) <dst> <src0> <src1> <src2>
```

Examples:

```
DPAS.u4.s8.8.8  (Exec_size) <dst> <src0> <src1> <src2>   // u4 src1, s8 src2
DPAS.bf.bf.8.8  ...
DPAS.hf.hf.8.8  ...
```

GEN dump in 2508.06753 is the same op with GEN-style types: `dpas.8x8 (16|M0) rD:d rAcc:d rW:s2 rA:b`.

## Operand roles

```
D = C + A x B
D (Dst)  : M x N
C (Src0) : M x N
A (Src2) : M x K     // activations
B (Src1) : K x N     // weights
```

M = RepeatCount. N = exec size (8 pre-PVC, 16 PVC+). K = depth * OPS_PER_CHAN.

Note: Src2 is A, Src1 is B. Easy to swap when reading ESIMD `dpas(B, A)`.

## Precision table (integer ranges)

```
u1  [0, 1]        reserved
s1  [-1, 0]       reserved
u2  [0, 3]
s2  [-2, 1]       two's complement 2-bit, NOT a symmetric [-2, 2)
u4  [0, 15]
s4  [-8, 7]
u8  [0, 255]
s8  [-128, 127]
bf, hf, tf32, and later bf8/hf8 for float paths
```

s2 range [-2, 1] is a numeric-oracle landmine. NVFP4 E2M1 is a different 16-code float LUT and is not this integer type.

Integer src1 and src2 precisions may differ. Float src1/src2 must match.

## OPS_PER_CHAN

```
if Src1 bits == 16:            OPC = 2          // bf/hf, and Src2 must be 16
else if Src1 == 8 or Src2 == 8: OPC = 4          // any 8-bit present => dot4
else:                           OPC = 8          // both < 8-bit => dot8
```

XEHP+ only encodes SD in {1,2,4,8} but "only supports a systolic depth of 8".

K with SD=8:

```
tf32:        K = 8
bf16/fp16:   K = 16
any 8-bit:   K = 32     // includes s2 x s8
both <8-bit: K = 64     // s4xs4, s2xs2, s4xs2, u4xu2, ...
```

This is why 2508.06753 says int2xint8 DPAS has the same throughput as int8xint8 DPAS: the 8-bit operand forces OPC=4.

s2xs2 / s4xs4 can do twice the K per instruction vs s8. That is a different arm than the literature mix.

## Layout

Dst, Src0, Src2: row-major in GRF-as-1D.

Src1 (B): neither row- nor column-major. View GRF as 2D, one GRF per row, 16 DW columns on PVC (N=16). Each column of B is packed down its GRF column. For 8-bit, K=32 => 32 int8 per column = 8 DW = 8 GRFs for all 16 columns.

That packed-B layout is hardware VNNI.

Src2 alignment (SD=8):

```
s8/u8: 8 DWORD aligned
s4/u4: 4 DWORD aligned
s2/u2: 2 DWORD aligned
```

Dst, Src0, Src1 are GRF aligned.

## Repeat vs depth

Pseudo-code: for each r in RC, take Src0.R[r], then for d in SD do a per-channel dot4/dot8/dot2 into that accumulator, then write Dst.R[r]. Src1 is reused across repeats; Src2 walks forward.

2508.06753 reuses r96 (int8 A) across 8 DPAS with different s2 weight registers: that is several depth/weight tiles against one quantized activation register, not RC inside a single instruction. Their `dpas.8x8` already has RC=8 inside each op.

## K2 use

Keep the disasm. Match W.A (e.g. s2.s8 vs s2.s2). Record exec size 16 vs 8. Record whether IGC emitted OPC=4 or 8. Host oracle must use the signed ranges above, not E2M1 codes.
