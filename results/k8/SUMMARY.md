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

## ESIMD Mamba-2 SSD SSU T=1 card0 (2026-09-04af)

backend sycl+l0, standalone AOT
mamba_ssu_t1. T=1 heads=64
d_head=64 d_state=128 groups=8
VL=16 wi=one_per_head spin=0.
Not GDN. Rank pipe_host.
GDN delta 7.1 is the wrong math.
Sibling SSU B8/W4 is community,
not FINDINGS.

cosine=1.000000 max_abs=4.7684e-07
ok=1. gpu-run 2s.

| card | event_us | pipe_host_us | wait_host_us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 234.510 | 190.028 | 258.467 | 22.18 | 1.000000 | 4.7684e-07 | 1 |

timed_begin act=950 cur=933
throttle=0. timed_end act=cur=2800
throttle=0. start D3hot act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
Do not freeze (spin=0, clocks
not held). Held-clock 04ah
80.064 us. One-card first
light. Sibling pending. New
math: not a both-card floor.

Evidence: `results/k8/mamba_ssu_t1_s0_card0.txt`,
`results/k8/mamba_ssu_t1_s0_card0.freq`.

## ESIMD grouped 6-expert s8 decode M=1 card1 (2026-09-04ag)

backend sycl+l0, standalone AOT
moe_group_s8_m1. experts=6 m=1
n=1856 k=2688 spin=4000. RC=4
NT=2 kstep=64 wg=8x2_alongN
scale 0.02 out f16. Six
launches share A and one
in-order queue. Not U=14.
Rank pipe_host of all 6.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | last_event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164.633 | 33.398 | 165.223 | 0.3623 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done
act=cur=2800 throttle=0.
freq 50 ms throttle=0 all 17
samples. GPU-window act=517,400
then 7x 2800. vs 6*16.060=96.360
(~1.72x); vs 6*44.285=265.710
(~0.62x). mean event 27.4
us/expert vs U=14 16.060.
pipe_host ~ event sum. Numeric
closed. Clocks held. One-card
enough (matched s8 RC=4 family).

Evidence: `results/k8/moe_group_s8_m1_s4000_card1.txt`,
`results/k8/moe_group_s8_m1_s4000_card1.freq`.

## ESIMD Mamba conv1d K=4 C=4096 T=1 card1 (2026-09-04ai)

backend sycl+l0, standalone AOT
gdn_conv1d. T=1 C=4096 k=4 f16
VL=16 wg=16 spin=4000. Existing
generic --c path. Rank
pipe_host. Occupancy may wash
vs GDN C=2048 4.4.

cosine=1.000000 max_abs=0
cosine_st=1.000000 max_abs_st=0
ok=1. gpu-run 2s.

| card | event_us | wait_host_us | pipe_host_us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.896 | 14.891 | 4.355 | 22.57 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done
act=cur=2800 throttle=0.
freq 50 ms throttle=0 all 7
samples. GPU-window act=0 then
550 then 0 (sampler miss,
short kernel). start D3hot
act=0 cur=2800 throttle=0.
end D0 act=0 cur=2800
throttle=0. vs C=2048
4.350/4.500 (wash, not 2x);
vs C=6144 4.799/5.000; vs
fused qkv 4.4. Numeric
closed. One-card enough
(conv family already both-
card at other C).

Evidence: `results/k8/mamba_conv_c4096_s4000_card1.txt`,
`results/k8/mamba_conv_c4096_s4000_card1.freq`.

## ESIMD Mamba-2 SSD SSU T=1 held-clock card0 (2026-09-04ah)

backend sycl+l0, standalone AOT
mamba_ssu_t1. T=1 heads=64
d_head=64 d_state=128 groups=8
VL=16 wi=one_per_head spin=4000.
Not GDN. Rank pipe_host.
GDN delta 7.1 is the wrong math.
Prior 04af 190.028 us spin=0.
Sibling SSU B8/W4 is community,
not FINDINGS.

cosine=1.000000 max_abs=4.7684e-07
ok=1. gpu-run 2s.

| card | event_us | pipe_host_us | wait_host_us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 79.531 | 80.064 | 93.563 | 52.65 | 1.000000 | 4.7684e-07 | 1 |

spin_done act=cur=2800
throttle=0. timed act=cur=2800
throttle=0 both ends. start D0
act=0 cur=2800 throttle=0. end
D0 act=cur=2800 throttle=0.
freq 50 ms throttle=0 all 11
samples. GPU-window act=0,0,0,400
then 4x 2800 then 3x 0. vs 04af
190.028 (~0.42x; 190 was ramp).
Numeric closed. Clocks held.
One-card held-clock. Sibling
04aj 79.923 us, spread 0.176%.

Evidence: `results/k8/mamba_ssu_t1_s4000_card0.txt`,
`results/k8/mamba_ssu_t1_s4000_card0.freq`.

## ESIMD Mamba-2 SSD SSU T=1 held-clock card1 sibling (2026-09-04aj)

backend sycl+l0, standalone AOT
mamba_ssu_t1. T=1 heads=64
d_head=64 d_state=128 groups=8
VL=16 wi=one_per_head spin=4000.
Not GDN. Rank pipe_host.
GDN delta 7.1 is the wrong math.
Prior 04ah card0 80.064 us.
Sibling SSU B8/W4 is community,
not FINDINGS.

cosine=1.000000 max_abs=4.7684e-07
ok=1. gpu-run 2s.

| card | event_us | pipe_host_us | wait_host_us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 79.531 | 80.064 | 93.563 | 52.65 | 1.000000 | 4.7684e-07 | 1 |
| 1 | 79.536 | 79.923 | 93.906 | 52.74 | 1.000000 | 4.7684e-07 | 1 |

spin_done act=cur=2800
throttle=0. timed act=cur=2800
throttle=0 both ends. start
D3hot act=0 cur=2800 throttle=0.
end D0 act=0 cur=2800
throttle=0. freq 50 ms
throttle=0 all 11 samples.
GPU-window act=0,0,0,400 then
4x 2800 then 2x 0. vs 04ah
80.064 spread 0.176% (<5%).
Numeric closed. Clocks held
2800. Both-card FINDINGS floor
80 us pipe_host at 2800.

Evidence: `results/k8/mamba_ssu_t1_s4000_card1.txt`,
`results/k8/mamba_ssu_t1_s4000_card1.freq`.

## NVFP4 nibble LUT routed expert UP-proj M=1 card0 REFUSED (2026-09-04ak)

backend sycl+l0, standalone AOT
nibble_lut_sc. NT=2 U=16 m=1
n=1856 k=2688 spin=4000.
Packed E2M1 B, simd nibble
LUT, VNNI4, RC=4 8x2-N
scale-to-f16. Never bitcast
s4. Stock U=16 inner_k=1024.
No nibble_lut u14. Rank
pipe_host. One-card shape
steal on LUT family.

| card | pipe_host_us | note |
|---:|---|---|
| 0 | REFUSED | U=16 inner_k=1024, 2688%1024=640 |

stderr: nibble_lut_sc: shape
m=1 n=1856 k=2688 nt=2
unroll=16. n=1856%32=0 (N
ok). Check-only 4x32x1024
cosine=1.000000 max_abs=0
ok=1 event 209.688 pipe_host
204.179 at act=400/550 (not
held; not Lightning timed).
start D3hot act=0 cur=2800
throttle=0. gpu-run 2s exit
2. STOP rewrite. 158 us
stays 5120, not this shape.

Evidence: `results/k8/nibble_lut_moe_up_m1_s4000_card0.txt`,
`results/k8/nibble_lut_moe_up_m1_s4000_card0.freq`,
`results/k8/nibble_lut_moe_up_m1_s4000_card0.u16_refuse.txt`.

## W8A8 shared expert M=1 card1 (2026-09-04am)

cosine=1.000000 max_abs=0.030846 ok=1.
gpu-run 16s.

| card | us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.273 | 236.032 | 1.000000 | 0.030846 | 1 |

Clocks (freq 50 ms): throttle=0 all 177 samples.
GPU-window act=2800,900,900,2800,2800
cur=2800,867,867,2800,2800.
Three samples act=cur=2800. start act=0
cur=2800 throttle=0 D0. end act=0
cur=2800 throttle=0 D0.

vs routed-up W8A8 44.285 (~0.95x)
and square M=1 5120 44 us. Launch
class, not N-linear (napkin
44.285*(3712/1856)~88.6 CONFIG).
Do not freeze (act not held 2800).
One-card enough (W8A8 family
already matched both cards).

Evidence: `results/k8/w8a8_moe_shared_m1_card1.txt`,
`results/k8/w8a8_moe_shared_m1_card1.freq`.

## ESIMD s8 packed qkv M=1 card0 (2026-09-04al)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=1
n=4608 k=2688 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as expert-up 16.060. Packed
Q 4096 + K 256 + V 256. Rank
pipe_host. Napkin N-linear
16.060*(4608/1856)~40.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.240 | 16.609 | 1.5254 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 10 samples.
GPU-window act=0,0,0,0,400,550,
2800 then 3x 0. start D3hot
act=0 cur=550 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs expert-up 16.060 (~1.03x);
vs napkin 40 (~0.42x); vs Qwen
packed qkv 74 (~0.22x). Launch
class, not N-linear. Numeric
closed. Clocks held. One-card
enough (matched s8 U=14 RC=4
family). W8A8 packed qkv still
open.

Evidence: `results/k8/esimd_s8_qkv_m1_s4000_card0.txt`,
`results/k8/esimd_s8_qkv_m1_s4000_card0.freq`.
