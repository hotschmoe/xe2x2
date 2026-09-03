# K5 RMSNorm-epilogue quant 2026-09-02s

Backend sycl+l0, standalone icpx 2026.1.1 AOT intel_gpu_bmg_g31.
Naive one-WI-per-row. Contract: RMSNorm gamma=1 eps=1e-6,
symmetric s8 qmax=127, per-row absmax scale. Host oracle in
float; device f16 input so max_abs<=1 is the close.

Path A: 2 launches (rmsnorm f16, then quant).
Path B: 1 launch (fused writes s8+scale).

| M x K | card0 two us | card0 fused us | card1 two us | card1 fused us | max_abs |
|---|---:|---:|---:|---:|---:|
| 1 x 5120 | 1361 | 830 | 1252 | 830 | 1 |
| 1 x 17408 | 3769 | 2820 | 3769 | 2820 | 0-1 |
| 64 x 5120 | 1607 | 1194 | 1607 | 1194 | 1 |
| 64 x 17408 | 5498 | 4086 | 5498 | 4086 | 1 |

Fusion removes 1 launch and ~30-40% us on this kernel.
Absolute us is hundreds to thousands: this is not a serving
epilogue and not a beat of the 45 us W8A8 GEMM. It is the
naive launch-count micro.

## WG=256 bandwidth fused (2026-09-02u)

Same contract, one WG per row, 256 WI, group reduce. Backend
sycl+l0. GT0 cur=2800 both cards. max_abs<=1.

| M x K | card0 first fused | card0 repeat | card1 fused | two-launch card1 |
|---|---:|---:|---:|---:|
| 1 x 5120 | 36.0 | 13.0 | 7.1 | 10.8 |
| 1 x 17408 | 93.9 | 38.3 | 21.1 | 31.0 |
| 64 x 5120 | 38.7 | 15.8 | 8.7 | 12.5 |
| 64 x 17408 | 20.0 | 37.6 | 21.2 | 31.0 |

Short kernels still swing (7-36 us at M=1 5120). All are ~20-100x
the naive 830 us. Fusion still beats two-launch. Do not freeze
one us. This epilogue is now in the same class as 45 us W8A8
GEMM, not 18x larger.

## scalar RMSNorm-quant inside GEMM (2026-09-02at)

One launch, f16 A -> RMSNorm+s8 then 64 dpas.8x4 f16 out.
Scalar math in the k-loop. NT=2 spin=4000. timed 2800.

| shape | card0 event | card1 event | cosine | max_abs |
|---|---:|---:|---:|---:|
| 1 x 5120 | 314.2 | 312.9 | 0.73 | 50 |

~9x GEMM-only 34 us. Not closed. Two-launch K5+GEMM stays.

## vectorized RMSNorm-quant inside GEMM (2026-09-02au)

Same contract, simd convert/reduce/hmax/rnde, f16 2D
load pitch in bytes. NT=2 spin=4000. timed 2800.
ocloc: 64x dpas.8x4, rnde (32|M0), no SLM.

| shape | card0 event | card0 pipe | card1 event | card1 pipe | cosine | max_abs |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 5120 | 71.67 | 72.47 | 71.78 | 72.28 | 1.0 | 0.015625 |
| 4 x 5120 | 72.44 | 72.80 | 72.32 | 72.76 | 1.0 | 0.015625 |

~4.3x scalar 313 us. Closed at 1 f16 ulp. Not a 34 us
GEMM beat. Two-launch WG-256 + GEMM still the decode path.

## WG-256 producer then s8 GEMM (2026-09-02ba)

Producer writes s8 A + scale once; GEMM is the 64
dpas.8x4 f16 tile. In-order queue, no host wait.
NT=2 spin=4000. timed 2800. cosine=1.0 max_abs=0.015625.

| shape | card | prod_us | gemm_us | pair_event | pipe_host |
|---|---|---:|---:|---:|---:|
| 1 x 5120 | 0 | 10.46 | 33.06 | 43.66 | 44.30 |
| 1 x 5120 | 1 | 10.40 | 33.20 | 43.75 | 44.43 |
| 4 x 5120 | 0 | 10.43 | 33.26 | 43.82 | 44.87 |
| 4 x 5120 | 1 | 10.40 | 33.20 | 43.72 | 44.99 |

Beats fusev 72 us. Extra ~10 us over GEMM-only 34.
Keep two-kernel producer+GEMM.

## producer+GEMM M=1 N=17408 both cards (2026-09-03cd)

Same dpas_s8_prod. NT=2 spin=4000. timed
act=cur=2800 throttle=0. cosine=1.0
max_abs=0 (M=1) / 0.0078125 (M=4).

| shape | card | prod_us | gemm_us | pair_event | pipe_host |
|---|---|---:|---:|---:|---:|
| 1 x 17408 x 5120 | 0 | 10.846 | 143.529 | 154.505 | 156.354 |
| 1 x 17408 x 5120 | 1 | 10.818 | 142.625 | 153.581 | 154.033 |
| 4 x 17408 x 5120 | 0 | 10.807 | 143.003 | 153.940 | 155.764 |
| 4 x 17408 x 5120 | 1 | 10.828 | 143.437 | 154.401 | 155.938 |

New floor 155 us both cards. ~3.52x
square 44. Spread ~1.5%. Extra still
~11 us (K=5120). Beats W8A8 158.1.

## producer+GEMM M=1 K=17408 card1 (2026-09-03ce)

Same dpas_s8_prod. NT=2 spin=4000. timed
act=cur=2800 throttle=0. cosine=0.999995
max_abs=0.064 ok=1.

| shape | card | prod_us | gemm_us | pair_event | pipe_host |
|---|---|---:|---:|---:|---:|
| 1 x 5120 x 17408 | 1 | 33.099 | 261.068 | 294.305 | 294.453 |
| 4 x 5120 x 17408 | 1 | 33.870 | 260.320 | 294.320 | 295.423 |

~6.68x square 44. Extra ~33 us
(K-linear). GEMM matches s8 261.6.
Loses to W8A8 155.3 (~1.90x). Napkin
297 hit. One-card. Do not freeze 294 us.

K5 next: sibling producer K=17408 vs
mixed s8xs4 numeric oracle.
