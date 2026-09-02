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
naive launch-count micro. A bandwidth-capable producer
epilogue is still open.
