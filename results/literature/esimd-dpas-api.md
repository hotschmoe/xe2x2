# ESIMD DPAS API (`sycl_ext_intel_esimd` / `xmx::dpas`)

Status: LANDED from intel/llvm docs (preferred over blogs).
Not local evidence. Templates describe what IGC is asked to emit, not what BMG-G31 actually runs.

## Source

- Spec: https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md
  (section "Dot Product Accumulate Systolic - DPAS API")
- Example: https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/dpas.md
- Namespace: `sycl::ext::intel::esimd::xmx`
- Kernel attribute: `[[intel::sycl_explicit_simd]]`
- Related ISA: IGC `documentation/visa/instructions/DPAS.md` (see `igc-dpas-isa.md`)

Docs name DG2 (exec size 8) and PVC (exec size 16). Xe2/BMG is PVC-class for exec size 16. Confirm with a local query; do not assume DG2 tables.

## Template parameters

```
enum class dpas_argument_type {
  Invalid = 0,
  u1 = 1,  // reserved, not supported
  s1 = 2,  // reserved, not supported
  u2 = 3,  // unsigned 2 bits
  s2 = 4,  // signed 2 bits
  u4 = 5,  // unsigned 4 bits
  s4 = 6,  // signed 4 bits
  u8 = 7,  // unsigned 8 bits
  s8 = 8,  // signed 8 bits
  bf16 = 9,
  fp16 = 10,
  tf32 = 12,
};

// Result = A x B + C
template <
    int SystolicDepth, int RepeatCount,
    typename T, typename CT, typename BT, typename AT,
    dpas_argument_type BPrecision = ...,
    dpas_argument_type APrecision = ...,
    int N, int BN, int AN>
simd<T, N> dpas(simd<CT, N> C, simd<BT, BN> B, simd<AT, AN> A);

// Result = A x B  (no acc input)
template <
    int SystolicDepth, int RepeatCount, typename T, typename BT, typename AT,
    dpas_argument_type BPrecision = ...,
    dpas_argument_type APrecision = ...,
    int BN, int AN>
auto dpas(simd<BT, BN> B, simd<AT, AN> A);
```

Call convention in the example:

```
c = xmx::dpas<8, RepeatCount, ResType, BPackedType, APackedType, BPrec, APrec>(b, a);
```

Argument order of the call is (B, A) for the no-acc form, i.e. weights/src1 first, activations/src2 second, matching IGC "Src1=B, Src2=A".

### SystolicDepth

Documented as 8 for all known target devices. XEHP+ hardware only supports SD=8 (IGC).

### RepeatCount

Docs example uses 4. Range cited: 1 to 8. Issue 21741 and the 2508.06753 dump both use 8. M = RepeatCount.

### Execution size N

- DG2: N must be 8
- PVC: N must be 16
- Xe2/BMG: treat as 16 unless a local query says otherwise

N is not a template on the short form; it is implied by the simd lengths.

## K formula (this is the mix-width law)

```
K = SystolicDepth * OpsPerChannel
OpsPerChannel = min(32 / MaxBitSizeOfElement(A, B), 8)
```

OpsPerChannel in {1, 2, 4, 8}.

Documented examples:

```
A tf32, B tf32  => K = 8
A fp16, B fp16  => K = 16
A s8,   B s8    => K = 32
A u4,   B u2    => K = 64
A s2,   B s2    => K = 64
```

Derived for K2 mixes, using MaxBitSize:

```
s8 x s8 : max=8, OPC=4, K=32
s4 x s4 : max=4, OPC=8, K=64
s2 x s2 : max=2, OPC=8, K=64
s2 x s8 : max=8, OPC=4, K=32   <-- same K as int8, not 4x
u2 x u8 : max=8, OPC=4, K=32
s4 x s8 : max=8, OPC=4, K=32
s4 x s2 : max=4, OPC=8, K=64
```

s1/u1 are reserved and not supported.

## Legal type combinations

DG2 (N=8):

```
Result              C                   B                         A
float               float               half                      half
float               float               bfloat16                  bfloat16
unsigned int, int   unsigned int, int   u8,s8,u4,s4,u2,s2         u8,s8,u4,s4,u2,s2
```

PVC (N=16):

```
Result              C                   B                         A
float, half         float, half         half                      half
float, bfloat16     float, bfloat16     bfloat16                  bfloat16
float               float               tfloat32                  tfloat32
unsigned int, int   unsigned int, int   u8,s8,u4,s4,u2,s2         u8,s8,u4,s4,u2,s2
```

Integer A and B may mix widths (u4 x u2 is an explicit example). Float A and B must match (no hf x bf). Acc/result for integer is int or unsigned int (s32/u32).

## Packing (must match DPAS, not naive row-major)

A, C, Result: horizontal packing into 32-bit elements.
B: vertical packing, also called VNNI.

For elements >= 8-bit, horizontal packing of A/C is "automatic". For 2-bit and 4-bit, pack into uint8 or uint32.

Example, 4-bit A stored as uint8 (two 4-bit fields per byte):

```
byte = (a_{2i+1} << 4) | a_{2i}
```

B VNNI for 16-bit: two consecutive K rows packed into one uint32 per column:

```
dword = (row1 << 16) | row0
```

For 2-bit B, 16 int2 along K land in one 32-bit channel. That is the VNNI16 the 2508.06753 paper names.

Example dpas.md uses u4 x u4 packed in `unsigned char`, ResType `unsigned int`, SystolicDepth=8, RepeatCount=4, ExecSize=16:

```
OpsPerChannel = min(32 / max(4,4), 8) = 8
M = 4, K = 8*8 = 64, N = 16
```

## Transforms / 2D load

`load_2d` has `Transposed` and `Transformed` template flags. `Transformed=true` is the hardware VNNI transform on the load (LSC). Campaign already records a sibling result: hardware `lsc_load_2d` Transformed was bit-exact on BMG-G31; host-prepacked flat VNNI was not. Re-measure; do not assume.

## joint_matrix vs ESIMD

oneAPI GPU opt guide: XMX via `sycl::ext::oneapi::experimental::matrix::joint_matrix` requires `aspect ext_intel_matrix`. Issue 21741: that aspect was false on B70 with 2025.3.3; Intel says BMG31 joint_matrix needs 2026.0+. ESIMD `xmx::dpas` is the path that actually lights DPAS when joint_matrix is gated off.

joint_matrix is SPMD-subgroup (not ESIMD). Do not mix the two in one kernel.

## K2 checklist from the API

- Compile `dpas<s8,s8>`, `dpas<s4,s4>`, `dpas<s2,s2>`, unsigned, and mixed `s2 x s8` / `u2 x u8`.
- Keep RepeatCount and SystolicDepth=8 as the first tile.
- Pack B VNNI; pack sub-byte A horizontally.
- Disassemble: expect `dpas.<W>.<A>.8.<RC>` with W/A in {s2,u2,s4,u4,s8,u8}.
- Numeric oracle on host with the IGC s2 range [-2, 1], not a naive [-2, 2).
