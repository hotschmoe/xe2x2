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

## E2M1 two-term s4 U=14 routed expert UP-proj M=1 card1 (2026-09-04ao)

backend sycl+l0, standalone AOT
compose_e2m1_sc_u14. NT=2 U=14
m=1 n=1856 k=2688 spin=4000.
A=s4. B=E2M1 split two s4
planes, acc=acc_lo+8*acc_hi.
RC=4 wg=8x2_alongN
dpas_lo_hi=56. Never bitcast.
Rank pipe_host.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.245 | 15.518 | 0.6545 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,400,750,
2800,2800,0. start D3hot
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs s8 16.060 (~0.97x, same
class); vs W8A8 44.285 (~0.35x,
a beat); vs 5120 two-term 28.5
(~0.54x); vs N-linear
28.5*(1856/5120)~10.3 (~1.50x);
vs K-linear 28.5*(2688/5120)~15.0
(~1.04x). Launch class vs s8,
tracks K-linear not N-linear.
Numeric closed. Clocks held.
One-card enough (matched
two-term RC=4 family both-card
at 5120). Never bitcast.

Evidence: `results/k8/e2m1_twoterm_moe_up_u14_s4000_card1.txt`,
`results/k8/e2m1_twoterm_moe_up_u14_s4000_card1.freq`.

## NVFP4 nibble LUT U=14 routed expert UP-proj M=1 card0 (2026-09-04an)

backend sycl+l0, standalone AOT
nibble_lut_sc_u14. NT=2 U=14
inner_k=896 m=1 n=1856
k=2688 spin=4000. Packed
E2M1 B, simd nibble LUT,
VNNI4, RC=4 8x2-N
scale-to-f16. Never bitcast
s4. Stock U=16 REFUSED 04ak.
Rank pipe_host. One-card
U=14 steal on LUT family.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | GBs_packedB | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 83.130 | 83.659 | 0.1200 | 30.007 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 12 samples.
GPU-window act=0,0,0,0,0 then
4x 2800 then 3x 0. start D0
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs s8 16.060 (~5.21x); vs
W8A8 44.285 (~1.89x); vs LUT
5120 158 (~0.529x); vs napkin
K-linear 158*(2688/5120)~83
(~1.01x). K-linear, not
launch-class. LUT tax (30
GB/s). Numeric closed. Clocks
held. One-card enough
(matched LUT family at 5120).

Evidence: `results/k8/nibble_lut_moe_up_u14_s4000_card0.txt`,
`results/k8/nibble_lut_moe_up_u14_s4000_card0.freq`.

## W8A8 packed qkv M=1 card1 (2026-09-04aq)

cosine=1.000000 max_abs=0.030884 ok=1.
gpu-run 15s.

| card | us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.320 | 299.765 | 1.000000 | 0.030884 | 1 |

Clocks (freq 50 ms): throttle=0 all 173 samples.
GPU-window act=400,850,850,1000,2800
cur=400,817,817,967,2800.
One sample act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=2800 throttle=0 D0.

vs s8 packed qkv 16.609 (~2.49x;
s8 wins). vs routed-up W8A8
44.285 (~0.93x) and shared
42.273 (~0.98x) and square
M=1 5120 44 us. Launch class,
not N-linear (napkin
44.285*(4608/1856)~109.95 CONFIG).
Do not freeze (act not held 2800).
One-card enough (W8A8 family
already matched both cards).

Evidence: `results/k8/w8a8_qkv_m1_card1.txt`,
`results/k8/w8a8_qkv_m1_card1.freq`.

## ESIMD s8 o-proj M=1 card0 (2026-09-04ap)

backend sycl+l0, standalone AOT
dpas_s8_sc. NT=2 U=16 m=1
n=2688 k=4096 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as packed qkv 16.609. Stock
U=16 inner_k=1024 divides
4096 (4 K-blocks). Rank
pipe_host.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.776 | 23.115 | 0.9668 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,400,0 then
3x 2800 then 0. start D0
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs packed qkv 16.609 (~1.39x);
vs expert-up 16.060 (~1.44x);
vs packed qkv W8A8 41.320
(~0.56x); vs Qwen NT1 SK 44
(~0.53x); vs Qwen W8A8 47
(~0.49x); vs K-block napkin
16.609*(4/3)~22.1 (~1.04x).
Not launch-class 16. Tracks
extra K-block (4 vs 3).
Numeric closed. Clocks held.
One-card enough (matched s8
RC=4 family). Lightning W8A8
o-proj still open.

Evidence: `results/k8/esimd_s8_oproj_m1_s4000_card0.txt`,
`results/k8/esimd_s8_oproj_m1_s4000_card0.freq`.

## ESIMD s8 shared expert M=1 card0 (2026-09-04ar)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=1
n=3712 k=2688 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as expert-up 16.060. ONE shared
expert, every token. Rank
pipe_host. Napkin N-linear
16.060*(3712/1856)=32.120.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 1s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.213 | 16.541 | 1.2308 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,400,0,2800
then 3x 0. start D3hot
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs W8A8 42.273 (~0.39x, a beat);
vs expert-up 16.060 (~1.03x);
vs packed qkv 16.609 (~1.00x);
vs napkin 32.120 (~0.52x).
Launch class, not N-linear.
Numeric closed. Clocks held.
One-card enough (matched s8
U=14 RC=4 family). nvfp4 /
GPTQ still open.

Evidence: `results/k8/esimd_s8_shared_m1_s4000_card0.txt`,
`results/k8/esimd_s8_shared_m1_s4000_card0.freq`.

## oneDNN nvfp4_gemm_w4a16 routed expert UP-proj M=1 card1 (2026-09-04as)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`
plus v028 `/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so`.
`nvfp4_gemm_w4a16` folded bf16 scale g16.
B packed NT stride(0)=1. A bf16.
ONE routed expert UP-proj n=1856 k=2688.
warmup 10 iters 20. No M=64 heat.
Rank us. No serve. Card1 only.
Napkin is CONFIG. ABSENT/EXC is a RESULT.

HAS nvfp4_gemm_w4a16 True HAS f8scale True.
out bf16 [1,1856]. gpu-run 17s.

| card | us | HAS | f8scale | out |
|---:|---:|---|---|---|
| 1 | 39.255 | True | True | bf16 [1,1856] |

Clocks (freq 50 ms): throttle=0 all 195 samples.
GPU-window act=400,1600,1900,1900
cur=400,1583,1883,1883.
Zero samples act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=1883 throttle=0 D0.

vs unheld 5120 37.169 (~1.06x). vs held
34.7 (~1.13x). vs W8A8 44.285 (~0.89x).
vs s8 16.060 (~2.44x). vs two-term
15.518 (~2.53x). vs LUT 83.659 (~0.47x).
vs napkin 37.169*(1856/5120)~13.47
(~2.91x). Launch class, not N-linear.
No E2M1 cosine this dump. f8scale not
timed. Do not freeze (act not held 2800;
no M=64 heat). One-card enough (w4a16
family already matched both cards at 5120).

Evidence: `results/k8/nvfp4_moe_up_m1_card1.txt`,
`results/k8/nvfp4_moe_up_m1_card1.freq`.

## ESIMD grouped 6-expert s8 DOWN-proj M=1 card0 (2026-09-04at)

backend sycl+l0, standalone AOT
moe_group_s8_m1. experts=6 m=1
n=2688 k=1856 spin=4000. RC=4
NT=2 kstep=64 wg=8x2_alongN
scale 0.02 out f16. Six
launches share A and one
in-order queue. Not U=14
(k=1856). Rank pipe_host of
all 6. Napkin launch-bound
~165 or K-linear
165.223*(1856/2688)=114.087.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | last_event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 119.487 | 23.487 | 120.502 | 0.4968 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done
act=cur=2800 throttle=0.
freq 50 ms throttle=0 all 15
samples. GPU-window act=0,0,0,0,0
then 6x 2800 then 3x 0. vs
grouped-up 165.223 (~0.73x);
vs 6*16.060=96.360 (~1.25x);
vs K-linear 114.087 (~1.06x);
vs 6*44.285=265.710 (~0.45x).
mean event 19.9 us/expert vs
UP 27.4. pipe_host ~ event
sum. K-linear, not launch-
bound. Numeric closed. Clocks
held. One-card enough
(matched s8 RC=4 family).

Evidence: `results/k8/moe_group_s8_down_m1_s4000_card0.txt`,
`results/k8/moe_group_s8_down_m1_s4000_card0.freq`.

## ESIMD s8 RC=4 routed expert UP-proj M=64 card1 (2026-09-04au)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=64
n=1856 k=2688 spin=512. Same
RC=4 8x2-N scale-to-f16 family
as M=1 expert-up 16.060. ONE
expert, not grouped-6. Rank
pipe_host. Napkin N-linear
wrong at M=1; at M=64 expect
leaving launch-class.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.286 | 31.198 | 19.7787 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 12 samples.
GPU-window act=0,0,0,0,0,0,0,0,400
then 3x 0. start D0 act=0
cur=1883 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs M=1 s8 16.060 (~1.94x);
vs W8A8 M=1 44.285 (~0.70x);
vs 5120 M=64 75 (~0.42x);
vs M-linear 16.060*64=1028
(~0.030x); vs N-linear
75*(1856/5120)~27.2 (~1.15x).
Left launch-class 16, not
M-linear. Tracks N-linear from
5120 M=64 75. Numeric closed.
Clocks held. One-card enough
(matched s8 U=14 RC=4 family).

Evidence: `results/k8/esimd_s8_moe_up_m64_s512_card1.txt`,
`results/k8/esimd_s8_moe_up_m64_s512_card1.freq`.

## ESIMD fused conv+SSU T=1 card0 (2026-09-04av)

backend sycl+l0, standalone AOT
mamba_fuse_t1. Sequential-in-
one-go conv K=4 C=4096 THEN
Mamba-2 SSU T=1, two in-order
launches, last event. T=1
heads=64 d_head=64 d_state=128
groups=8 spin=4000. Not GDN.
Rank pipe_host vs 4.355+80.064
=84.419. First fuse: one-card
first light. Sibling not yet.

conv_cosine=1.000000
conv_max_abs=0
ssu_cosine=1.000000
ssu_max_abs=1.9073e-06 ok=1.
gpu-run 2s.

| card | last_event_us | pipe_host_us | wait_host_us | baseline_us | conv_cosine | ssu_cosine | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 79.544 | 81.075 | 98.638 | 84.419 | 1.000000 | 1.000000 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 12 samples.
GPU-window act=0,0,0,0,517 then
4x 2800 then 3x 0. start D3hot
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs napkin 84.419 (~0.96x);
vs SSU 80.064 (~1.01x; +1.011
us). last_event 79.544 vs SSU
event 79.531 (wash). Packed
conv launch, not +4.355.
Numeric closed. Clocks held.
One-card first; sibling not
yet. Not a both-card floor.

Evidence: `results/k8/mamba_fuse_t1_s4000_card0.txt`,
`results/k8/mamba_fuse_t1_s4000_card0.freq`.

## ESIMD Lightning MoE router M=1 card1 (2026-09-04aw)

backend sycl+l0, standalone AOT
moe_router. M=1 hidden=2688
n_experts=128 n_groups=8
topk_group=1 topk=6
selection=global_top6. f16
GEMV plus sigmoid, f32 acc.
spin=4000. Timed kernel is
GEMV+sigmoid; top-6 host-side
of GPU scores. No expert bias.
Rank pipe_host vs eager.
Napkin BW ~1.14 us or launch-
class 16. New numeric.

cosine=1.000000
max_abs=8.9407e-08 set_ok=1
ok=1. gpu-run 1s.

| card | event_us | wait_host_us | pipe_host_us | cosine | max_abs | set_ok | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.932 | 40.508 | 27.333 | 1.000000 | 8.9407e-08 | 1 | 1 |

host_top6=gpu_top6
indices=87,10,64,111,99,71.
timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,0,400,2800
then 3x 0. start D0 act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs eager wait_host 40.508
(~0.67x); vs event 26.932
(~1.01x); vs s8 UP 16.060
(~1.70x); vs o-proj 23.115
(~1.18x); vs grouped-up
165.223 (~0.17x); vs BW 1.14
(~24x). GEMV+sigmoid 27-class,
not XMX launch 16, not BW.
Numeric closed. Clocks held.
One-card first; sibling not
yet. Rank pipe_host.

Evidence: `results/k8/moe_router_s4000_card1.txt`,
`results/k8/moe_router_s4000_card1.freq`.

## NVFP4 closed-form nibble LUT U=14 routed expert UP-proj M=1 card0 (2026-09-04ax)

backend sycl+l0, standalone AOT
nibble_lut_scf_u14. NT=2 U=14
inner_k=896 m=1 n=1856
k=2688 spin=4000. Packed
E2M1 B, closed-form exp/mant
decode, VNNI4, RC=4 8x2-N
scale-to-f16. Never bitcast
s4. Rank pipe_host. One-card
U=14 steal on closed-form
LUT family (5120 both-card
134.8).

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | GBs_packedB | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 71.359 | 71.715 | 0.1398 | 34.956 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 11 samples.
GPU-window act=0,0,0,0,550 then
3x 2800 then 3x 0. start D0
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs merge LUT 83.659 (~0.857x);
vs s8 16.060 (~4.47x); vs
two-term 15.518 (~4.62x); vs
W8A8 44.285 (~1.62x); vs
closed-form 5120 134.8
(~0.532x); vs napkin K-linear
134.8*(2688/5120)~70.8
(~1.01x). K-linear, not
launch-class. LUT tax (35
GB/s). Under STOP 4x W8A8
176. Numeric closed. Clocks
held. One-card enough
(matched closed-form LUT
family at 5120). Rank
pipe_host.

Evidence: `results/k8/nibble_lut_scf_u14_s4000_card0.txt`,
`results/k8/nibble_lut_scf_u14_s4000_card0.freq`.

## ESIMD dequant-to-bf16 GEMV hail-mary M=1 card1 (2026-09-04ay)

backend sycl+l0, standalone AOT
gemv_bf16_m1. hail-mary. M=1
N=1856 K=2688. bf16 A/B, f32
acc, f16 out. XVE FMA VL=16
4-out/WI WG=16, not DPAS.
spin=4000. GEMV-only, not a
dequant launch. Rank pipe_host
vs LUT 83.659 and s8 16.060.
New numeric. STOP if us > 4x
W8A8 ~177.

cosine=1.000000 max_abs=0
ok=1. gpu-run 2s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.609 | 41.037 | 26.962 | 0.3701 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,0,400,2800,
2800,0,0. start D0 act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs LUT 83.659 (~0.32x, a beat);
vs s8 16.060 (~1.68x); vs W8A8
44.285 (~0.61x, a beat); vs
two-term 15.518 (~1.74x); vs
nvfp4 39.255 (~0.69x); vs
router 27.333 (~0.99x); vs
event 26.609 (~1.01x); vs STOP
177 (~0.15x). 27-class like
router, not LUT 84 / XMX 16.
Numeric closed. Clocks held.
One-card first; sibling not
yet. No STOP. Rank pipe_host.

Evidence: `results/k8/gemv_bf16_moe_up_m1_s4000_card1.txt`,
`results/k8/gemv_bf16_moe_up_m1_s4000_card1.freq`.

## ESIMD Mamba-2 SSD prefill T=256 card0 (2026-09-04az)

backend sycl+l0, standalone AOT
mamba_ssd_t256. T=256
chunk_size=128 heads=64
d_head=64 d_state=128 groups=8
spin=0. Serial state carry over
T. Not GDN. Rank pipe_host vs
256*80.064=20496.384 and vs
fused T=1 81.075. First
chunked prefill: one-card first
light. Sibling not yet.

cosine=1.000000
max_abs=6.1035e-05 ok=1.
oracle_s=0.434 score_y=full
score_state=final
samples=1572864. gpu-run 3s.

| card | event_us | wait_host_us | pipe_host_us | napkin_us | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21082.469 | 21114.645 | 21114.404 | 20496.384 | 1.000000 | 6.1035e-05 | 1 |

timed act=cur=2800 throttle=0
both ends. freq 50 ms
throttle=0 all 19 samples.
GPU-window act=0,0,0,0,400 then
5x 0 then 6x 2800 then 3x 0.
start D3hot act=0 cur=2800
throttle=0. end D0 act=0
cur=2800 throttle=0. vs napkin
20496.384 (~1.03x); vs fused
T=1 81.075*256=20755.2
(~1.017x); vs SSU 80.064 per
token 82.478 (~1.03x). pipe ~
event. Serial T, not a beat.
Numeric closed. spin=0 but
timed act=cur=2800 (kernel
~21 ms ramps). Do not freeze
as a spin=4000 both-card
floor. One-card first; sibling
not yet. Not a both-card
floor. SSU is not GDN delta.
Rank pipe_host.

Evidence: `results/k8/mamba_ssd_t256_s0_card0.txt`,
`results/k8/mamba_ssd_t256_s0_card0.freq`.

## ESIMD GPTQ s4 g128 U=14 routed expert UP-proj M=1 card1 (2026-09-04ba)

backend sycl+l0, standalone AOT
dpas_s4_gptq_sc_u14. True INT4
XMX control, not NVFP4. NT=2
U=14 inner_k=896 (three
blocks). m=1 n=1856 k=2688.
gs=128. RC=4 8x2-N
s4xs4_gptq scale-to-f16.
weights=SYNTHETIC (no
Lightning GPTQ ckpt; omit
--b-bin). spin=4000. Never
E2M1. Rank pipe_host vs s8
16.060, two-term 15.518,
nvfp4 39.255, W8A8 44.285,
GPTQ 5120 29.9. New numeric.

cosine=1.000000 max_abs=0
ok=1. gpu-run 2s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.807 | 30.068 | 16.224 | 0.6312 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,0,400,2800,
2800,0,0. start D3hot act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs s8 16.060 (~1.01x); vs
two-term 15.518 (~1.05x); vs
nvfp4 39.255 (~0.41x, a beat);
vs W8A8 44.285 (~0.37x, a
beat); vs GPTQ 5120 29.9
(~0.54x); vs napkin K-linear
29.9*(2688/5120)~15.7
(~1.03x); vs N-linear
29.9*(1856/5120)~10.8
(~1.50x); vs event 15.807
(~1.03x); vs STOP 177
(~0.09x). Launch class, not
N-linear. Scale tax washed.
Numeric closed. Clocks held.
One-card first; sibling not
yet. Never bitcast. Rank
pipe_host.

Evidence: `results/k8/gptq_s4_moe_up_u14_s4000_card1.txt`,
`results/k8/gptq_s4_moe_up_u14_s4000_card1.freq`.

## E2M1 two-term s4 k64 routed expert DOWN-proj M=1 card1 (2026-09-04bc)

backend sycl+l0, standalone AOT
compose_e2m1_k64. NT=2 unroll=1
kstep=64 inner_k=64. m=1 n=2688
k=1856. RC=4 8x2-N two-term
w_lo+8*w_hi scale-to-f16. A=s4.
Never bitcast. spin=4000. New
k64 compose (U=14/16 refuse
k=1856). Rank pipe_host vs
two-term UP 15.518, grouped
down s8 120.502, GPTQ 16.224,
W8A8 44.285.

cosine=1.000000 max_abs=0
ok=1. gpu-run 1s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.656 | 24.378 | 11.008 | 0.9363 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=0,0,0,0,400,750,
0,2800,0. start D3hot act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs two-term UP 15.518 (~0.71x);
vs K-linear from UP
15.518*(1856/2688)~10.715
(~1.03x); vs grouped-down s8/6
20.084 (~0.55x); vs GPTQ
16.224 (~0.68x); vs s8 U=14 UP
16.060 (~0.69x); vs W8A8
44.285 (~0.25x); vs 5120
two-term 28.5 (~0.39x); vs
K-linear 28.5*(1856/5120)~10.33
(~1.07x); vs N-linear from UP
~22.47 (~0.49x); vs event
10.656 (~1.03x); vs STOP 177
(~0.06x). K-linear, not
16-class launch floor. Numeric
closed. Clocks held. One-card
first; sibling not yet. Never
bitcast. Rank pipe_host. No
STOP.

Evidence: `results/k8/e2m1_twoterm_down_k64_s4000_card1.txt`,
`results/k8/e2m1_twoterm_down_k64_s4000_card1.freq`.

## NVFP4 nibble LUT k64 routed expert DOWN-proj M=1 card0 (2026-09-04bb)

backend sycl+l0, standalone AOT
nibble_lut_k64. NT=2 unroll=1
kstep=64 (29 blocks). m=1
n=2688 k=1856 spin=4000.
Packed E2M1 B, simd nibble
LUT, VNNI4, RC=4 8x2-N
scale-to-f16. Never bitcast
s4. Stock U=14/16 cannot
divide k=1856. Rank
pipe_host. One-card k64
steal on LUT family.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | pipe_host_us | TOPS | GBs_packedB | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37.961 | 38.351 | 0.2628 | 65.711 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 10 samples.
GPU-window act=0,0,0,0,400 then
2x 2800 then 3x 0. start D3hot
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs LUT up 83.659 (~0.458x); vs
s8 16.060 (~2.39x); vs
grouped-down/6 20.084 (~1.91x);
vs W8A8 44.285 (~0.866x, a
beat); vs napkin K-linear
83.659*(1856/2688)~57.765
(~0.664x); vs same packed
bytes as up 83.659 (~0.458x);
vs STOP 177 (~0.217x). Faster
than K-linear, not
launch-class 16. LUT tax (66
GB/s vs up 30). Numeric
closed. Clocks held. Never
bitcast. One-card first;
sibling not yet. Rank
pipe_host.

Evidence: `results/k8/nibble_lut_down_k64_s4000_card0.txt`,
`results/k8/nibble_lut_down_k64_s4000_card0.freq`.

## hail-mary 16-entry iselect nibble LUT U=14 routed expert UP-proj M=1 card1 (2026-09-04bd)

backend sycl+l0, standalone AOT
nibble_lut_sct_u14. hail-mary.
NT=2 U=14 inner_k=896 m=1
n=1856 k=2688 spin=4000.
Packed E2M1 B, 16-entry GRF
table + iselect, VNNI4, RC=4
8x2-N scale-to-f16. Never
bitcast s4. Lost ~6.46x at
5120. One fire then STOP if
still a loss vs merge LUT
83.659. Rank pipe_host.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 4s.

| card | event_us | pipe_host_us | TOPS | GBs_packedB | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 536.547 | 537.031 | 0.0186 | 4.649 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 33 samples.
GPU-window act=0,0,0,0 then
25x 2800 then 4x 0. start D3hot
act=0 cur=2800 throttle=0. end
D0 act=0 cur=2800 throttle=0.
vs merge LUT 83.659 (~6.42x);
vs closed-form 71.715 (~7.49x);
vs s8 16.060 (~33.4x); vs
two-term 15.518 (~34.6x); vs
W8A8 44.285 (~12.1x); vs 5120
iselect 1022 (~0.526x); vs
napkin K-linear ~536.4
(~1.00x); vs N-linear ~370.5
(~1.45x); vs STOP 177 (~3.03x).
K-linear, not launch-class.
LUT tax (4.6 GB/s). Numeric
closed. Clocks held. Never
bitcast. STOP rewrite. One-card
enough (iselect family already
both-card at 5120). Rank
pipe_host.

Evidence: `results/k8/nibble_lut_sct_u14_s4000_card1.txt`,
`results/k8/nibble_lut_sct_u14_s4000_card1.freq`.

## oneDNN nvfp4_gemm_w4a16 routed expert DOWN-proj M=1 card0 (2026-09-04be)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`
plus v028 `/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so`.
`nvfp4_gemm_w4a16` folded bf16 scale g16.
B packed NT stride(0)=1. A bf16.
ONE routed expert DOWN-proj n=2688 k=1856.
warmup 10 iters 20. No M=64 heat.
Rank us. No serve. Card0 only.
Napkin is CONFIG. ABSENT/EXC is a RESULT.

HAS nvfp4_gemm_w4a16 True HAS f8scale True.
out bf16 [1,2688]. B (928,2688) stride (1,928).
gpu-run 13s.

| card | us | HAS | f8scale | out |
|---:|---:|---|---|---|
| 0 | 39.224 | True | True | bf16 [1,2688] |

Clocks (freq 50 ms): throttle=0 all 152 samples.
GPU-window act=1600,1450,1350,1200,1200
cur=1567,1450,1500,1183,1183.
Zero samples act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=1183 throttle=0 D0.

vs UP 39.255 (~1.00x). vs two-term down
11.008 (~3.56x). vs LUT down 38.351
(~1.02x). vs W8A8 44.285 (~0.89x).
vs s8 16.060 (~2.44x). vs GPTQ 16.224
(~2.42x). vs grouped-down s8/6 20.084
(~1.95x). vs unheld 5120 37.169
(~1.06x). vs held 34.7 (~1.13x).
vs napkin N-linear 39.255*(2688/1856)
~56.85 (~0.69x). vs napkin K-linear
39.255*(1856/2688)~27.10 (~1.45x).
Launch class, not N-linear. Same
packed B bytes and FLOPs as UP.
No E2M1 cosine this dump. f8scale not
timed. Do not freeze (act not held 2800;
no M=64 heat). One-card enough (w4a16
family already matched both cards at 5120).

Evidence: `results/k8/nvfp4_moe_down_m1_card0.txt`,
`results/k8/nvfp4_moe_down_m1_card0.freq`.

## ESIMD s8xs4 U=14 routed expert UP-proj M=1 card0 (2026-09-04bf)

backend sycl+l0, standalone AOT
dpas_s8xs4_sc_u14. Integer mix
control: A=s8 B=s4 packed 2/byte.
Not E2M1 bitcast. Not E2M1-as-s8
LUT (nibble_lut 83.659 already).
NT=2 U=14 inner_k=448 (six
blocks). m=1 n=1856 k=2688.
RC=4 8x2-N s8xs4 scale-to-f16.
dpas=28. spin=4000. Never
E2M1. Rank pipe_host vs s8
16.060, GPTQ 16.224, two-term
15.518, W8A8 44.285, s8xs4
5120 21.961. New u14 mix.

cosine=1.000000 max_abs=0
ok=1. gpu-run 1s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.958 | 27.291 | 13.458 | 0.7700 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 9 samples.
GPU-window act=1200,0,0,0,400,
2800,2800,0,0. start D3hot act=0
cur=1183 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs s8 16.060 (~0.84x, a beat);
vs GPTQ 16.224 (~0.83x, a beat);
vs two-term 15.518 (~0.87x, a
beat); vs W8A8 44.285 (~0.30x,
a beat); vs nvfp4 39.255
(~0.34x, a beat); vs LUT 83.659
(~0.16x); vs s8xs4 5120 21.961
(~0.61x); vs napkin K-linear
21.961*(2688/5120)~11.53
(~1.17x); vs N-linear
21.961*(1856/5120)~7.96
(~1.69x); vs s2xs8 14.1
(~0.95x); vs event 12.958
(~1.04x); vs STOP 177
(~0.08x). 13.5-class, below
the 16-class launch floor.
Numeric closed. Clocks held.
Never bitcast. One-card first;
sibling not yet. Rank
pipe_host.

Evidence: `results/k8/s8xs4_moe_up_u14_s4000_card0.txt`,
`results/k8/s8xs4_moe_up_u14_s4000_card0.freq`.

## hail-mary bitcast E2M1 onto s4 DPAS routed expert UP-proj M=1 card1 (2026-09-04bg)

backend sycl+l0, standalone
bitcast_e2m1_s4. hail-mary.
m=1 n=1856 k=2688. RC=8
padM=RC (M=1 not aligned).
E2M1 nibbles as s4 two's
complement into dpas<s4,s4>.
Oracle E2M1 dequant.
warmup=20 iters=20. No spin.
Never a floor. Rank cosine.

cosine=0.663320 max_abs=14784
ok=0. gpu-run 2s.

| card | event_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.104 | 1.7697 | 0.663320 | 14784 | 0 |

M=1 unaligned rc=2. padM=8
check 8x16x64 max_abs=352
ok=0 (matches K6). GPU
max_abs matches host oracle.
freq 50 ms throttle=0 all 14
samples. GPU-window act=
0,0,0,0,400 then 9x 0. Zero
samples act=cur=2800. start
D3hot act=0 cur=2800
throttle=0. end D0 act=0
cur=600 throttle=0. vs
two-term 15.518 (~2.91x); vs
s8 16.060 (~2.81x); vs W8A8
44.285 (~1.02x); vs STOP 177
(~0.25x). Cosine death.
Clocks not held. Do not
freeze us. Never a floor.
Explicit negative on this
histogram. One-card enough
(K6 already both-card). Rank
cosine.

Evidence: `results/k8/bitcast_e2m1_s4_moe_up_m1_card1.txt`,
`results/k8/bitcast_e2m1_s4_moe_up_m1_card1.freq`.

## E2M1 two-term s4 U=14 routed expert UP-proj M=64 card0 (2026-09-04bh)

backend sycl+l0, standalone AOT
compose_e2m1_sc_u14. NT=2 U=14
m=64 n=1856 k=2688 spin=512.
A=s4. B=E2M1 split two s4
planes, acc=acc_lo+8*acc_hi.
RC=4 wg=8x2_alongN
dpas_lo_hi=56. Never bitcast.
Rank pipe_host vs M=1 two-term
15.518, s8 M=64 31.198, W8A8
M=1 44.285. Prior: large-M
loses vs W8A8 at 5120.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 3s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28.833 | 43.655 | 29.637 | 22.1474 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 12 samples.
GPU-window act=0,0,0,0,400,0,0,0,
400,0,0,0. start D3hot act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
vs M=1 two-term 15.518 (~1.91x);
vs s8 M=64 31.198 (~0.95x, a
beat); vs W8A8 M=1 44.285
(~0.67x, a beat); vs 5120
8x2-N 217.92 (~0.14x); vs
napkin N-linear
217.92*(1856/5120)~79 (~0.38x);
vs K-linear
217.92*(2688/5120)~114 (~0.26x);
vs M-linear 15.518*64~993
(~0.030x); vs 4x8 N-linear
68.7*(1856/5120)~24.9 (~1.19x);
vs event 28.833 (~1.03x); vs
STOP 177 (~0.17x). Left
launch-class vs M=1, same as
s8 M=64 ~1.94x. CONFIG prior
large-M loses vs W8A8 at 5120
does not hold here. Numeric
closed. Clocks held. Never
bitcast. One-card enough
(matched two-term RC=4 family).
Rank pipe_host. M=256 still
open.

Evidence: `results/k8/e2m1_twoterm_moe_up_m64_s512_card0.txt`,
`results/k8/e2m1_twoterm_moe_up_m64_s512_card0.freq`.

## oneDNN nvfp4_gemm_w4a16 shared expert M=1 card1 (2026-09-04bi)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`
plus v028 `/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so`.
`nvfp4_gemm_w4a16` folded bf16 scale g16.
B packed NT stride(0)=1. A bf16.
ONE shared expert n=3712 k=2688.
warmup 10 iters 20. No M=64 heat.
Rank us. No serve. Card1 only.
Napkin is CONFIG. ABSENT/EXC is a RESULT.

HAS nvfp4_gemm_w4a16 True HAS f8scale True.
out bf16 [1,3712]. B (1344,3712) stride (1,1344).
gpu-run 14s.

| card | us | HAS | f8scale | out |
|---:|---:|---|---|---|
| 1 | 38.266 | True | True | bf16 [1,3712] |

Clocks (freq 50 ms): throttle=0 all 165 samples.
GPU-window act=400,900,900,900,1350,1350
cur=400,900,900,883,1333,1333.
Zero samples act=cur=2800. start act=0
cur=600 throttle=0 D3hot. end act=0
cur=1333 throttle=0 D0.

vs UP 39.255 (~0.97x). vs DOWN 39.224
(~0.98x). vs shared s8 16.541 (~2.31x).
vs shared W8A8 42.273 (~0.91x). vs s8
16.060 (~2.38x). vs two-term 15.518
(~2.47x). vs GPTQ 16.224 (~2.36x).
vs LUT 83.659 (~0.46x). vs unheld 5120
37.169 (~1.03x). vs held 34.7 (~1.10x).
vs napkin N-linear 39.255*(3712/1856)
~78.51 (~0.49x). Launch class, not
N-linear, despite 2x N / 2x packed B
vs UP. No E2M1 cosine this dump.
f8scale not timed. Do not freeze
(act not held 2800; no M=64 heat).
One-card enough (w4a16 family already
matched both cards at 5120).

Evidence: `results/k8/nvfp4_moe_shared_m1_card1.txt`,
`results/k8/nvfp4_moe_shared_m1_card1.freq`.

## ESIMD s8 packed qkv M=64 card0 (2026-09-04bj)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=64
n=4608 k=2688 spin=512. Same
RC=4 8x2-N scale-to-f16 family
as M=1 qkv 16.609. Packed Q
4096 + K 256 + V 256. Rank
pipe_host vs M=1 qkv 16.609,
s8 expert M=64 31.198,
two-term M=64 29.637, W8A8
qkv M=1 41.320. Napkin
N-linear from expert M=64
31.198*(4608/1856)~77.5.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 5s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 109.099 | 124.061 | 106.287 | 14.5322 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 35 samples.
GPU-window act=0,0,0,0,400 then
26x 0, 2800 then 3x 0. start
D0 act=0 cur=2800 throttle=0.
end D0 act=0 cur=2800
throttle=0. vs M=1 qkv 16.609
(~6.40x); vs s8 expert M=64
31.198 (~3.41x); vs two-term
M=64 29.637 (~3.59x); vs W8A8
qkv M=1 41.320 (~2.57x); vs
napkin N-linear ~77.5 (~1.37x);
vs M-linear 16.609*64~1063
(~0.100x); vs event 109.099
(~0.97x); vs STOP 165 (~0.64x).
Left launch-class vs M=1, more
than expert M=64 ~1.94x. Fat
N=4608 is 2.48x expert N but
3.41x us. Do not claim a
W8A8 beat (M=64 W8A8 open).
Numeric closed. Clocks held.
One-card enough (matched s8
U=14 RC=4 family). Rank
pipe_host. M=256 still open.

Evidence: `results/k8/esimd_s8_qkv_m64_s512_card0.txt`,
`results/k8/esimd_s8_qkv_m64_s512_card0.freq`.

## W8A8 o-proj / mamba out_proj M=1 card1 (2026-09-04bk)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`int8_gemm_w8a8` GEMM-only. Heat M=64 spin=512.
Lightning attn o-proj / mamba out_proj
n=2688 k=4096. Same MNK. Rank us.
No serve. Card1 only. Napkin is CONFIG.

cosine=1.000000 max_abs=0.031166 ok=1.
gpu-run 13s.

| card | us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.081 | 249.768 | 1.000000 | 0.031166 | 1 |

Clocks (freq 50 ms): throttle=0 all 149 samples.
GPU-window act=1350,550,1050,0,0,1350
cur=1333,1017,1017,1000,1150,1500.
Zero samples act=cur=2800. start act=0
cur=1333 throttle=0 D3hot. end act=0
cur=2800 throttle=0 D0.

vs s8 o-proj 23.115 (~1.91x; s8 wins).
vs packed qkv W8A8 41.320 (~1.07x).
vs expert-up 44.285 (~1.00x). vs shared
42.273 (~1.04x). vs nvfp4 UP 39.255
(~1.12x). vs napkin N-linear
44.285*(2688/1856)~64.14 (~0.69x).
vs K-linear 44.285*(4096/2688)~67.48
(~0.65x). vs square M=1 5120 44 us.
vs Qwen o-proj W8A8 47 (~0.94x).
Launch class, not N-linear / K-linear
despite k=4096. Do not freeze (act
not held 2800). One-card enough
(W8A8 family already matched both
cards). Same MNK covers mamba
out_proj W8A8.

Evidence: `results/k8/w8a8_oproj_m1_card1.txt`,
`results/k8/w8a8_oproj_m1_card1.freq`.

## ESIMD s8 mamba in_proj M=1 card0 (2026-09-04bl)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=1
n=10304 k=2688 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as packed qkv 16.609. N napkin
z/x 4096+4096 + B/C 1024+1024
+ dt 64. n%32=0. Official PTQ
is FP8; s8 is the beat-me
control. Rank pipe_host vs
packed qkv 16.609, expert-up
16.060, shared 16.541, o-proj
23.115. Napkin N-linear
16.060*(10304/1856)~89.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 2s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.922 | 36.842 | 23.504 | 2.4167 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 11 samples.
GPU-window act=0,0,0,0,400,0,600
then 2x 2800 then 2x 0. start
D3hot act=0 cur=2800 throttle=0.
end D0 act=0 cur=2800
throttle=0. vs packed qkv
16.609 (~1.42x); vs expert-up
16.060 (~1.46x); vs shared
16.541 (~1.42x); vs o-proj
23.115 (~1.02x); vs napkin
N-linear ~89 (~0.264x); vs
event 22.922 (~1.03x); vs
W8A8 qkv 41.320 (~0.57x); vs
W8A8 expert 44.285 (~0.53x).
n=10304 accepted (n%32=0).
Left 16-class vs packed qkv
16.609. Fat N=10304 is 5.55x
expert N but 1.46x us; 2.24x
qkv N but 1.42x us. Not
N-linear ~89. 23-class wash
vs o-proj 23.115. Official
PTQ is FP8; do not claim an
FP8 beat. Numeric closed.
Clocks held. One-card enough
(matched s8 U=14 RC=4 family).
Rank pipe_host. M=64 and FP8
still open.

Evidence: `results/k8/esimd_s8_mamba_in_m1_s4000_card0.txt`,
`results/k8/esimd_s8_mamba_in_m1_s4000_card0.freq`.

## FP8 W8A16 mamba in_proj M=1 card1 (2026-09-04bm)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`fp8_gemm_w8a16` GEMM-only. Lightning
mamba in_proj n=10304 k=2688. Official
PTQ is FP8, not NVFP4. A bf16, B
e4m3fn. warmup 10 iters 20. Rank us
vs s8 23.504 (04bl). No serve.
Card1 only. Napkin is CONFIG.
ABSENT/EXC is a RESULT.

HAS fp8_gemm_w8a16 True. out bf16
[1,10304]. gpu-run 14s.

| card | us | GBs | HAS | out |
|---:|---:|---:|---|---|
| 1 | 51.036 | 542.700 | True | bf16 [1,10304] |

Clocks (freq 50 ms): throttle=0 all 167 samples.
GPU-window act=400,1600,0,0,0,0,0,0,0,1500
cur=400,1583,1550,1550,1550,1500,1500,1500,1500,1500.
Zero samples act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=1500 throttle=0 D0.

vs s8 in_proj 23.504 (~2.17x; s8 wins).
vs packed qkv W8A8 41.320 (~1.24x).
vs expert-up 44.285 (~1.15x). vs o-proj
W8A8 44.081 (~1.16x). vs nvfp4 UP
39.255 (~1.30x). vs square fp8 70.340
(~0.73x). vs napkin N-linear
70.340*(10304/5120)~141.6 (~0.36x).
vs W8A8 N-linear 44.285*(10304/1856)
~245.9 (~0.21x). Fat N left 44-class
(~1.15x), not N-linear. 543 GB/s of
608 (~0.89x roof) on 27.7 MB B.
Official PTQ incumbent, not an E2M1
spoof. No host cosine this dump.
Do not freeze (act not held 2800).
One-card enough (fp8_gemm_w8a16
family already both-card at 5120).
Rank us. out_proj still open.

Evidence: `results/k8/fp8_mamba_in_m1_card1.txt`,
`results/k8/fp8_mamba_in_m1_card1.freq`.

## pytorch-xpu SDPA Lightning GQA 32/2 T_kv=1 card0 (2026-09-04bn)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`F.scaled_dot_product_attention`
enable_gqa bf16. q 32 heads kv 2
d=128 T_kv=1. warmup 20 iters 40
host sync. Not flash-attn. Six
layers. Rank us vs SSU 80.064
(04ah). Card0 only. Napkin is
CONFIG. ABSENT/EXC is a RESULT.

out (1, 32, 1, 128) bf16.
gpu-run 16s.

| card | us | vs_SSU |
|---:|---:|---:|
| 0 | 54.944 | 0.686x |

Clocks (freq 50 ms): throttle=0 all 183 samples.
GPU-window act=1500,1400 then 517,400
cur=1483,1400,1550,900,400.
Zero samples act=cur=2800. start act=0
cur=2800 throttle=0 D3hot. end act=0
cur=400 throttle=0 D0.

vs SSU 80.064 (~0.686x). Faster
not <<. 6-layer napkin 329.664 vs
23*SSU 1841.472 (~0.179x). No host
cosine this dump. Do not freeze
(act not held 2800; still beats
held SSU). Not flash-attn. Six
layers. One-card first; sibling
not yet. Rank us. T=256 still
open.

Evidence: `results/k8/gqa_t1_card0.txt`,
`results/k8/gqa_t1_card0.freq`.

## FP8 W8A16 mamba out_proj M=1 card1 (2026-09-04bo)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`fp8_gemm_w8a16` GEMM-only. Lightning
mamba out_proj n=2688 k=4096. Same MNK
as attn o-proj. Official PTQ is FP8,
not NVFP4. A bf16, B e4m3fn. warmup 10
iters 20. Rank us vs s8 o-proj 23.115
(04ap), W8A8 o-proj 44.081 (04bk), FP8
in_proj 51.036 (04bm). No serve.
Card1 only. Napkin is CONFIG.
ABSENT/EXC is a RESULT.

HAS fp8_gemm_w8a16 True. out bf16
[1,2688]. gpu-run 14s.

| card | us | GBs | HAS | out |
|---:|---:|---:|---|---|
| 1 | 43.694 | 251.982 | True | bf16 [1,2688] |

Clocks (freq 50 ms): throttle=0 all 159 samples.
GPU-window act=400,1550 then 0
cur=400,1517,1450,2400.
Zero samples act=cur=2800. start act=0
cur=1500 throttle=0 D3hot. end act=0
cur=2400 throttle=0 D0.

vs s8 o-proj 23.115 (~1.89x; s8 wins).
vs W8A8 o-proj 44.081 (~0.991x).
vs FP8 in_proj 51.036 (~0.856x).
vs expert-up 44.285 (~0.987x). vs packed
qkv W8A8 41.320 (~1.057x). vs nvfp4 UP
39.255 (~1.113x). vs square fp8 70.340
(~0.621x). vs napkin K-linear
70.340*(4096/5120)~56.3 (~0.776x). vs
N-linear 70.340*(2688/5120)~36.9
(~1.183x). vs byte-linear vs in_proj
~20.3 (~2.15x). Same MNK left 44-class
(~0.99x), not byte-linear / N-linear.
252 GB/s of 608 (~0.41x roof) on 11.01
MB B. Official PTQ incumbent, not an
E2M1 spoof. No host cosine this dump.
Do not freeze (act not held 2800).
One-card enough (fp8_gemm_w8a16
family already both-card at 5120).
Rank us. s8 out_proj still open.

Evidence: `results/k8/fp8_mamba_out_m1_card1.txt`,
`results/k8/fp8_mamba_out_m1_card1.freq`.

## pytorch-xpu SDPA Lightning GQA 32/2 T_kv=256 card0 (2026-09-04bp)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`F.scaled_dot_product_attention`
enable_gqa bf16. q 32 heads kv 2
d=128 T_kv=256. warmup 20 iters 40
host sync. Not flash-attn. Six
layers. Rank us vs SSU 80.064
(04ah), GQA T=1 54.944 (04bn),
SSD T=256 21114.404 (04az).
Card0 only. Napkin is CONFIG.
ABSENT/EXC is a RESULT.

out (1, 32, 1, 128) bf16.
gpu-run 17s.

| card | us | vs_SSU | vs_T1 | vs_SSD |
|---:|---:|---:|---:|---:|
| 0 | 48.560 | 0.607x | 0.884x | 0.0023x |

Clocks (freq 50 ms): throttle=0 all 195 samples.
GPU-window act=400,1000 then 517,400
cur=400,983 then 1117,400.
Zero samples act=cur=2800. start act=400
cur=400 throttle=0 D3hot. end act=0
cur=400 throttle=0 D0.

vs SSU 80.064 (~0.607x). Faster
not << vs decode SSU. vs GQA T=1
54.944 (~0.884x), not T-linear.
vs SSD 21114.404 (~0.0023x, 435x)
IS <<. 6-layer napkin 291.360 vs
23*SSU 1841.472 (~0.158x). No host
cosine this dump. Do not freeze
(act not held 2800; still beats
held SSU). Not flash-attn. Six
layers. One-card first; sibling
not yet. Rank us.

Evidence: `results/k8/gqa_t256_card0.txt`,
`results/k8/gqa_t256_card0.freq`.

## oneDNN nvfp4_gemm_w4a16 routed expert UP-proj M=64 card1 (2026-09-04bq)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`
plus v028 `/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so`.
`nvfp4_gemm_w4a16` folded bf16 scale g16.
B packed NT stride(0)=1. A bf16.
ONE routed expert UP-proj n=1856 k=2688.
warmup 10 iters 20. No extra M=64 heat.
Rank us vs M=1 39.255, s8 M=64 31.198,
two-term M=64 29.637. No serve. Card1
only. Napkin is CONFIG. ABSENT/EXC is a
RESULT. Clocks may not hold. P2P off.

HAS nvfp4_gemm_w4a16 True HAS f8scale True.
out bf16 [64,1856]. B (1344,1856) stride
(1,1344). gpu-run 15s.

| card | us | HAS | f8scale | out |
|---:|---:|---|---|---|
| 1 | 40.184 | True | True | bf16 [64,1856] |

Clocks (freq 50 ms): throttle=0 all 164 samples.
GPU-window act=400,1600,1550,1650,1800
cur=400,1583,1550,1617,1767.
Zero samples act=cur=2800. start act=0
cur=2400 throttle=0 D3hot. end act=0
cur=1767 throttle=0 D0.

vs M=1 39.255 (~1.02x). vs s8 M=64
31.198 (~1.29x). vs two-term M=64
29.637 (~1.36x). vs W8A8 44.285
(~0.91x). vs LUT 83.659 (~0.48x). vs
K6 M=64 37.1 (~1.08x). vs unheld 5120
37.169 (~1.08x). vs held 34.7 (~1.16x).
vs napkin K6 1.07x M=1 ~42.0 (~0.96x).
vs N-linear 37.1*(1856/5120)~13.45
(~2.99x). 40-class launch, not M-linear.
No E2M1 cosine this dump. f8scale not
timed. Do not freeze (act not held 2800).
One-card enough (w4a16 family already
matched both cards at 5120). Rank us.
No STOP. M=256 still open.

Evidence: `results/k8/nvfp4_moe_up_m64_card1.txt`,
`results/k8/nvfp4_moe_up_m64_card1.freq`.

## ESIMD s8 lm_head M=1 card0 (2026-09-04br)

backend sycl+l0, standalone AOT
dpas_s8_sc_u14. NT=2 U=14 m=1
n=131072 k=2688 spin=4000. Same
RC=4 8x2-N scale-to-f16 family
as packed qkv 16.609. Wide-N
leftover. n%32=0. Rank
pipe_host vs packed qkv 16.609,
expert-up 16.060. Napkin
N-linear 16.060*(131072/1856)~1134.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 10s.

| card | event_us | wait_host_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1027.229 | 1043.496 | 1012.237 | 0.6860 | 1.000000 | 0 | 1 |

timed act=cur=2800 throttle=0
both ends. spin_done act=cur=2800
throttle=0. freq 50 ms
throttle=0 all 99 samples.
GPU-window act=400,0,400,0 then
47x 2800 then 3x 0. cur=400,550,
400,2800. 47 samples
act=cur=2800. start D3hot act=400
cur=400 throttle=0. end D0 act=0
cur=2800 throttle=0. vs packed
qkv 16.609 (~60.9x); vs expert-up
16.060 (~63.0x); vs shared 16.541
(~61.2x); vs o-proj 23.115
(~43.8x); vs mamba in 23.504
(~43.1x); vs napkin N-linear
~1134 (~0.892x); vs event
1027.229 (~0.985x). n=131072
accepted (n%32=0). Host cosine
closed, not heavy. 336 MiB B.
348 GB/s of 608 (~0.57x roof).
Wide-N leftover IS N-linear, not
launch-class. Numeric closed.
Clocks held. One-card enough
(matched s8 U=14 RC=4 family).
Rank pipe_host. MTP-M still open.

Evidence: `results/k8/esimd_s8_lmhead_m1_s4000_card0.txt`,
`results/k8/esimd_s8_lmhead_m1_s4000_card0.freq`.

## W8A8 routed expert UP-proj M=64 card1 (2026-09-04bt)

Backend pytorch-xpu on sycl+l0. Image
`b70-sglang-xpu-int8-runtime:20260826-mtp6`.
`int8_gemm_w8a8` GEMM-only. Heat M=64 spin=512.
ONE routed expert UP-proj n=1856 k=2688. Rank us
vs s8 M=64 31.198, two-term 29.637, nvfp4 M=64
40.184, W8A8 M=1 44.285. No serve. Card1 only.
Napkin is CONFIG. Clocks may not hold. P2P off.

cosine=0.999998 max_abs=0.031235 ok=1.
gpu-run 12s.

| card | us | GBs | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.907 | 125.014 | 0.999998 | 0.031235 | 1 |

Clocks (freq 50 ms): throttle=0 all 136 samples.
GPU-window act=400,1000,1000,2800,2633,2800,2800
cur=400,983,967,2800,2800,2800,2800.
Three samples act=cur=2800. start act=0
cur=1767 throttle=0 D3hot. end act=0
cur=2800 throttle=0 D0.

vs s8 M=64 31.198 (~1.28x). vs two-term
M=64 29.637 (~1.35x). vs nvfp4 M=64
40.184 (~0.99x, wash). vs W8A8 M=1
44.285 (~0.90x). vs LUT 83.659 (~0.48x).
vs M-linear 44.285*64=2834 (~0.014x).
40-class launch, not M-linear. Matched-M
W8A8 does not lose to nvfp4 (04bq 0.91x
was vs M=1 44.285). Numeric closed. Do
not freeze (act not held 2800). One-card
enough (W8A8 family already matched both
cards). Rank us. No STOP. Grouped M=64
still open. in_proj W8A8 still open.

Evidence: `results/k8/w8a8_moe_up_m64_card1.txt`,
`results/k8/w8a8_moe_up_m64_card1.freq`.

## ESIMD grouped 6-expert s8 UP M=64 card0 (2026-09-04bs)

backend sycl+l0, standalone AOT
moe_group_s8_m1. experts=6 m=64
n=1856 k=2688 spin=512. RC=4
NT=2 kstep=64 wg=8x2_alongN
scale 0.02 out f16. Six
launches share A and one
in-order queue. Not U=14.
Not 64 one-expert launches.
Rank pipe_host of all 6.

cosine=1.000000 max_abs=0 ok=1.
gpu-run 3s.

| card | event_us | last_event_us | pipe_host_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 230.343 | 41.250 | 231.179 | 16.5737 | 1.000000 | 0 | 1 |

timed act=2783 cur=2800
throttle=1 both ends.
spin_done act=2783 cur=2800
throttle=1. freq 50 ms
throttle=1 in 1 of 25 samples.
GPU-window act=0,0,400 then 16x
0 then 2800,2783,2800 then 2x 0.
cur=2800,2800,2800,400 then 16x
400 then 5x 2800. 2 samples
act=cur=2800. start D3hot act=0
cur=2800 throttle=0. end D0
act=0 cur=2800 throttle=0.
median_sum 227.916 min 222.917
max 241.043. vs grouped M=1
165.223 (~1.40x); vs
6*31.198=187.188 (~1.23x); vs
6*39.907=239.442 (~0.97x); vs
6*44.285=265.710 (~0.87x); vs
6*16.060=96.360 (~2.40x); vs
M-linear 165.223*64=10574
(~0.022x); vs 6*29.637=177.822
(~1.30x); vs 6*40.184=241.104
(~0.96x). mean event 38.4
us/expert vs U=14 31.198.
pipe_host ~ event sum. Host
cosine closed, not heavy.
Left 165-class vs M=1 grouped,
not M-linear. k64-loop not
U=14. Numeric closed.
throttle=1 timed act=2783. Do
not freeze as held 2800.
One-card first; sibling later
(throttle=1, event spread ~8%).
Rank pipe_host. M=256 grouped
still open.

Evidence: `results/k8/moe_group_s8_up_m64_s512_card0.txt`,
`results/k8/moe_group_s8_up_m64_s512_card0.freq`.

## hail-mary dyadic s2/s4 planes routed expert UP-proj M=1 card1 (2026-09-04bu)

backend sycl+l0, standalone
dyadic_s2. hail-mary.
m=1 n=1856 k=2688. RC=8
K=64 N=16 OPC=8. padM=RC
(M=1 not aligned). s2xs2
one-plane control. 4-plane
E2M1 unfused sequential 4x
this (K6). residual
{1.5,3,6} pairwise sums of
{0.5,1,2,4}. s2xs4
COMPILE_REFUSED (K6). IGC
s2 range [-2,1], not E2M1.
warmup=20 iters=20. No spin.
Never a floor. Rank event
us. P2P off.

cosine=1.000000 max_abs=0
ok=1. gpu-run 2s.

| card | event_us | TOPS | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.615 | 1.9181 | 1.000000 | 0 | 1 |

M=1 unaligned rc=2. padM=8
check 8x16x64 event 23.443
max_abs=0 ok=1. four_plane
napkin 166.460 is CONFIG
(4x one-plane), not a fused
RESULT. freq 50 ms
throttle=0 all 10 samples.
GPU-window act=0,0,0,0,400,
550,600,0,0,0. cur=2800 x4
then 400,550 then 4x 600.
Zero samples act=cur=2800.
start D3hot act=0 cur=2800
throttle=0. end D0 act=0
cur=600 throttle=0. vs
two-term 15.518 (~2.68x /
~10.7x); vs s8 16.060
(~2.59x / ~10.4x); vs
s8xs4 13.458 (~3.09x /
~12.4x); vs W8A8 44.285
(~0.94x / ~3.76x); vs LUT
83.659 (~0.50x / ~1.99x);
vs bitcast padM=8 45.104
(~0.92x); vs STOP 177
(~0.235x / ~0.94x).
41-class launch like W8A8
44 / bitcast 45, not
16-class. No pipe_host
(event only). Clocks not
held. Do not freeze us.
Never a floor. One-card
enough (s2xs2 family
already numeric-closed at
K6). Rank event us. No
STOP (166 < 177).

Evidence: `results/k8/dyadic_s2_moe_up_m1_card1.txt`,
`results/k8/dyadic_s2_moe_up_m1_card1.freq`.

## hail-mary 16-code product LUT GEMV M=1 n=1856 k=2688 card0 (2026-09-04bv)

backend sycl+l0, standalone AOT
prod_lut_gemv_n1856. hail-mary.
W4A4 table16x16 m=1 n=1856
k=2688 spin=0. A/B E2M1 nibble
codes, 256-entry product
table, per-column scalar XVE
k-loop. Never bitcast s4.
Lost at 5120 (697/1106 us).
One fire then STOP if still a
loss vs LUT 83.659 / s8 16.060
/ GEMV bf16 26.962 or us > 4x
W8A8 ~177. Rank pipe_host.

cosine=0.000000 max_abs=0 ok=0.
gpu-run 2s exit 1.

| card | event_us | pipe_host_us | TOPS | GBs_unpackedB | cosine | max_abs | ok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 447.888 | 448.180 | 0.0223 | 11.13 | 0.000000 | 0 | 0 |

timed act=cur=2800 throttle=0
both ends. spin=0 (no
spin_done). freq 50 ms
throttle=0 all 6 samples.
GPU-window act=0,0,0,0,400,2800
cur=2800,2800,2800,2800,400,2800.
start D3hot act=0 cur=2800
throttle=0. clocks end skipped
(ok=0 tripped set -e). vs LUT
83.659 (~5.36x); vs s8 16.060
(~27.9x); vs GEMV bf16 26.962
(~16.6x); vs W8A8 44.285
(~10.1x); vs LUT down 38.351
(~11.7x); vs 5120 697 (~0.643x)
/ 1106 (~0.405x); vs napkin
K-linear ~366 (~1.23x); vs
N-linear ~253 (~1.77x); vs
FLOP ~133 (~3.38x); vs event
447.888 (~1.00x); vs STOP 177
(~2.53x). Over 4x W8A8. Naive
XVE k-loop, not launch-class.
LUT tax (11 GB/s). All-zero C:
N%16==0 plus (i*17)&15 /
(i*13)&15 makes B constant
along K and sum q(A)=0; 5120
max_abs=0 was vacuous. Never
bitcast. STOP rewrite. One-card
enough (product LUT family
already both-card at 5120).
Rank pipe_host.

Evidence: `results/k8/prod_lut_gemv_down_m1_s0_card0.txt`,
`results/k8/prod_lut_gemv_down_m1_s0_card0.freq`.
