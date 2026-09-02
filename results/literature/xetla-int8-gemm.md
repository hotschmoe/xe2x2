# XeTLA -- int8 GEMM templates and epilogues

Status: LANDED (public docs v0.3.7, Dec 2023). Repo archived 2024-12-18.
Not local evidence. Public XeTLA is an incumbent floor, not a B70 measurement.

## Source

- Site: https://intel.github.io/xetla
- Git: https://github.com/intel/xetla (archived; "CUTLASS will include all XeTLA features, refer Cutlass-Fork" = intel/sycl-tla)
- Construct a GEMM: https://github.com/intel/xetla/blob/main/media/docs/construct_a_gemm.md
- Functionality: https://github.com/intel/xetla/blob/main/media/docs/functionality.md
- 2508.06753 says their Xe2 int2 kernels were implemented "using the XeTLA framework" plus an autotune search. That autotune is not in the public 0.3.7 docs.

## What public XeTLA actually lists

Matrix-engine (XMX) GEMM dtypes:

```
int8  * int8  => {int8, int32}
bf16  * bf16  => {bf16, fp32}
fp16  * fp16  => {fp16, fp32}
tf32  * tf32  => {tf32, fp32}
```

Vector-engine (FPU): `fp32 * fp32 => fp32`.

Layouts: A {N,T} x B {N,T} => C N (row-major).

No public int4/int2 XMX dtype row. int4 appears only as a dequant policy (weights stored 4-bit, compute still on XMX at a wider type).

## int4 is dequant, not INT4 DPAS

Doxygen GEMM group lists:

```
gemm_t< compute_policy_int4_dequantize_xmx< ... dtype_scale_, dtype_zero_pt_, dequant_s_, gpu_arch::Xe > ... >
dispatch_policy_int4_dequantize_kslicing< ... >
gemm_universal_t< dispatch_policy_int4_dequantize_kslicing<...>, gemm_t_, epilogue_t_ >
```

Name is `int4_dequantize_xmx`: decompress then XMX, matching the campaign prior that oneDNN s4/u4 GEMM is weight decompress, not INT4 DPAS.

Public docs never mention int2. Native int2xint8 is a v2-paper / later-XeTLA thing, not the 0.3.7 feature table.

## Building a GEMM

Three pieces:

```
using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;
```

1) gemm building block (`gemm_selector_t`):

```
using gemm_t = typename xetla::group::gemm_selector_t<
    data_type_a, data_type_b,
    mem_layout::row_major, mem_layout::row_major,
    mem_space::global, mem_space::global,
    8, 8,                    // alignment, elements
    data_type_acc,           // accumulator
    tile_shape,
    sg_tile_k,               // K stride / inner-loop elements
    mma_engine::xmx,         // xmx vs fpu
    gpu_arch::Xe,            // docs also mention Xe2 as arch tag
    stages,                  // prefetch pipe stages
    sync_freq                // periodic sync, inner-loop units
>::gemm;
```

2) dispatch policy: default, kslicing (splitK), stream_k.

3) epilogue: register-level fusion after acc.

Typical tiles in the construct guide (shape-dependent, not a law):

```
wg_tile_m = 256
wg_tile_n = 256
sg_tile_m = 32
sg_tile_n = 64
```

Workgroup grid: (ceil(M/wg_m), ceil(N/wg_n)). Local: (ceil(wg_m/sg_m), ceil(wg_n/sg_n)). One subgroup = one hardware thread.

2508.06753 autotune searches those four tile sizes plus K-split and B-scale fusion.

## SplitK

For tall-K / small-MN (e.g. 256x256x8192) one wg_tile_m/n of 256 yields one workgroup. Split K:

- Workgroup-level: `dispatch_policy_kslicing<num_global_splitk, num_local_splitk, gpu_arch::Xe>`. Partial sums combined with `atomic_add`. Docs say atomic_add is float-only, so output dtype must be float, not fp16/bf16.
- Subgroup-level: reduce in SLM, half still allowed.

2508.06753 uses this for small GEMMs.

## Epilogues (public list)

```
Bias Add
GELU Forward
GELU Backward
RELU
Residual Add
```

Example chain:

```
using tile_op_t = chained_tile_op_t<relu_op_t, bias_op_t>;
using epilogue_t = xetla::group::epilogue_t<
    xetla::group::epilogue_policy_default<gpu_arch::Xe>,
    tile_shape,
    mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;
```

Epilogue also does C store / dtype convert. Fusion is register-level (acc still in GRF). That is the pattern 2508.06753 uses for int32->float * scales -> bf16, plus optional bias/activation.

## Successor

intel/xetla is archived. intel/sycl-tla (CUTLASS SYCL, target `intel_gpu_bmg_g31`) is the living tree. Campaign K1 should dump sycl-tla / oneDNN ISA rather than treat 0.3.7 XeTLA as the current floor.

## K2 / K1 takeaways

- Public XeTLA int8 XMX GEMM is the documented incumbent template: xmx engine, int32 acc, epilogue fusion, optional K-split.
- int4 in this tree is dequant-then-XMX. Do not cite it as native s4 DPAS.
- int2 is not in the public feature list. Native int2xint8 lives in the 2508.06753 XeTLA fork/autotune, which we do not have as a submodule here.
- Tile numbers above are starting guesses, not requirements.
