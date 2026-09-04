# K8 Lightning W8A8 expert-up M=1 2026-09-04ae

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`int8_gemm_w8a8` GEMM-only. Heat M=64 spin=512.
ONE routed expert UP-proj n=1856 k=2688. Rank us.
No serve. Card1 only. Napkin is CONFIG.

## W8A8 routed expert UP-proj M=1 card1 (2026-09-04ae)

cosine=1.000000 max_abs=0.030334 ok=1.
gpu-run 14s.

| card | us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.285 | 112.655 | 1.000000 | 0.030334 | 1 |

Clocks (freq 50 ms): throttle=0 all 160 samples.
GPU-window act=550,900,900,1050,2800,2800
cur=550,867,867,1017,2800,2800.
Two samples act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=2800 throttle=0 D0.

vs square W8A8 M=1 5120 44 us. Launch class,
not N-linear (napkin 44*(1856/5120)~16 CONFIG).
Do not freeze (act not held 2800). One-card
enough (W8A8 family already matched both cards).

Evidence: `results/k8/w8a8_moe_up_m1_card1.txt`,
`results/k8/w8a8_moe_up_m1_card1.freq`.

## ESIMD s8 RC=4 routed expert UP-proj M=1 card0 (2026-09-04ad)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=1
n=1856 k=2688 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as square s8 34. Stock
dpas_s8_sc U=16 refused
k=2688 (inner_k=1024,
2688%1024=640). ONE expert,
not grouped-6. Rank pipe_host.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.622 | 16.060 | 0.6387 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. vs W8A8 44.285 a
beat (~0.36x); vs square s8 34
(~0.47x); vs napkin
34*(1856/5120)~12 (~1.30x).
Numeric closed. Clocks held.
One-card enough (matched s8
RC=4 family).

Evidence: `results/k8/esimd_s8_moe_up_m1_s4000_card0.txt`,
`results/k8/esimd_s8_moe_up_m1_s4000_card0.freq`,
`results/k8/esimd_s8_moe_up_m1_s4000_card0.u16_refuse.txt`.
