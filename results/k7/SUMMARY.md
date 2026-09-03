# K7 GDN inventory 2026-09-03ep/gj

Qwen3.8-27B text_config. Backend pytorch-xpu on sycl+l0.
No serve. Rank us. Short kernels, cur 550-2800, throttle=0.

## Config inventory

H=5120, FFN 17408, 64 layers, full_attention_interval=4
(48 GDN + 16 full attn). conv_k=4. key_dim=2048 (16*128).
value_dim=6144 (48*128). Recurrent S bf16 1.5 MiB/layer,
72 MiB all GDN layers.

## depthwise conv1d K=4 both cards (2026-09-03ep/er)

Causal pad K-1. ok=1.

| arm | T | card0 us | card1 us |
|---|---:|---:|---:|
| q C=2048 | 1 | 125.283 | 117.114 |
| q C=2048 | 64 | 113.816 | 115.534 |
| q C=2048 | 256 | 114.778 | 115.485 |
| k C=2048 | 1 | 114.668 | 115.776 |
| v C=6144 | 1 | 116.826 | 119.369 |
| v C=6144 | 256 | 115.435 | 114.514 |

~115 us class all shapes. T=1 q spread ~7% (clocks).
Launch-bound (GB/s 0.2 at T=1 vs 55 at v T=256). vs s2
decode 11.5 vs W8A8 44.

## delta recurrent T=1 both cards (2026-09-03eq/es)

48 heads, S 128x128 bf16. ok=1.

| card | us | GBs |
|---|---:|---:|
| 0 | 306.575 | 10.38 |
| 1 | 309.042 | 10.30 |

~308 us both cards. Spread ~0.8%. ~7x W8A8 44.
State 1.5 MiB/layer is not the roof; this eager bmm path is.

## GDN q/v proj W8A8 M=1 (2026-09-03et/ew)

int8_gemm_w8a8. k=5120. heat M=64
spin=512. cosine=1 ok=1.

| arm | n | card0 us | card1 us |
|---|---:|---:|---:|
| q | 2048 | 45.344 | 58.429 |
| v | 6144 | 46.080 | 46.306 |

v-proj 46 us both (spread 0.5%).
q-proj 45-58, clocks. Same class
as square 44, not N-linear. Under
conv 115 and delta 308.

## ESIMD conv1d K=4 both cards (2026-09-03ex/fa)

backend sycl+l0, AOT gdn_conv1d.
T=1 f16, VL=16 wg=16, spin=4000.
cosine=1 max_abs=0 ok=1.

| C | card0 us (MHz) | card1 us (MHz) |
|---|---:|---:|
| 2048 | 4.350 (1700) | 4.500 (1400) |
| 6144 | 4.799 (2250) | 5.000 (2167) |

~4.4 us class both cards. ~26x
eager 115. Spread ~3.4%. Clocks
not 2800. Do not freeze 4.4.

## ESIMD delta 48x128x128 both cards (2026-09-03ey/ez)

backend sycl+l0, AOT gdn_delta.
spin=4000. cosine=1 max_abs=0.015625
(1 ulp f16) ok=1. act=cur=2800
throttle=0.

| card | pipe_host us | event us | GBs |
|---|---:|---:|---:|
| 0 | 7.028 | 7.825 | 454.61 |
| 1 | 7.093 | 8.432 | 450.48 |

7.1 us both cards at 2800. Spread
~0.9%. ~43x eager 308. Near copy
550. Mixer 4.4+7.1 ~11.5 us under
W8A8 46.

K7 next: o-proj W8A8 vs fused
qkv conv.

## GDN o-proj W8A8 both cards (2026-09-03fb/fe)

backend pytorch-xpu on sycl+l0.
M=1 n=5120 k=6144. heat M=64
spin=512. cosine=1 ok=1.

| card | us |
|---|---:|
| 0 | 46.293 |
| 1 | 47.133 |

46-47 us both. Spread ~1.8%.
Same class as v-proj 46.

## ESIMD fused qkv conv1d both cards (2026-09-03fc/fd)

backend sycl+l0, AOT
gdn_conv1d_qkv. C=10240 T=1
spin=4000. cosine=1 max_abs=0
ok=1.

| arm | card0 us (MHz) | card1 us (MHz) |
|---|---:|---:|
| fused | 4.856 (1183) | 4.437 (2800) |
| trio | 14.233 (2800) | 13.449 (2800) |

Fused 4.4-4.9 class. Spread ~9%
(clocks). Do not freeze 4.44 as
2800. Trio ~13.8 at 2800.

K7 next: packed qkv W8A8 vs
fuse conv+delta.

## Packed qkv W8A8 both cards (2026-09-03ff/fi)

backend pytorch-xpu on sycl+l0.
M=1 n=10240 k=5120. heat M=64
spin=512. cosine=1 ok=1.

| card | us |
|---|---:|
| 0 | 95.783 |
| 1 | 95.481 |

96 us both. Spread ~0.3%. ~1.44x
3 sequential 138.

## ESIMD mixer conv+delta both cards (2026-09-03fg/fh)

backend sycl+l0, AOT gdn_mixer.
spin=4000. cosine=1 max_abs=
1.22e-4 cosine_o=1 ok=1.
act=cur=2800 throttle=0.

| card | pipe_host us | event us |
|---|---:|---:|
| 0 | 8.746 | 8.898 |
| 1 | 8.229 | 9.758 |

8.2-8.7 us both at 2800. Spread
~6.3%. Do not freeze 8.23.

K7 next: packed qkv M=64 vs
conv T=64.

## Packed qkv W8A8 M=64 both cards (2026-09-03fj/fm)

backend pytorch-xpu on sycl+l0.
M=64 n=10240 k=5120. heat M=64
spin=512. cosine=1 ok=1.

| card | us |
|---|---:|
| 0 | 142.053 |
| 1 | 138.079 |

138-142 us both. Spread ~2.9%.
Wash vs 3x 46.

## ESIMD conv1d T=64 both cards (2026-09-03fk/fl)

backend sycl+l0, AOT gdn_conv1d_t.
C=2048 T=64 spin=4000. cosine=1
max_abs=0 ok=1. act=cur=2800
throttle=0.

| card | pipe_host us | event us |
|---|---:|---:|
| 0 | 10.130 | 9.768 |
| 1 | 10.161 | 9.771 |

10.1 us both at 2800. Spread
~0.3%. ~11x eager 115.

## ESIMD conv1d T=256 both cards (2026-09-03fn/fq)

backend sycl+l0, AOT gdn_conv1d_t.
C=2048 T=256 spin=4000. cosine=1
max_abs=0 ok=1. act=cur=2800
throttle=0.

| card | pipe_host us | event us |
|---|---:|---:|
| 0 | 37.607 | 37.253 |
| 1 | 37.811 | 37.229 |

37.7 us both at 2800. Spread
~0.5%. ~3.72x T=64.

## Packed qkv W8A8 M=256 both cards (2026-09-03fo/fp)

backend pytorch-xpu on sycl+l0.
M=256 n=10240 k=5120. heat M=64
spin=512. cosine=1 ok=1.

| card | us |
|---|---:|
| 0 | 163.739 |
| 1 | 163.539 |

164 us both. Spread ~0.12%.
~1.17x M=64.

## ESIMD conv1d T=256 C=6144 both cards (2026-09-03fr/fu)

backend sycl+l0, same
gdn_conv1d_t. C=6144 T=256
spin=4000. cosine=1 max_abs=0
ok=1. timed act=cur=2800
throttle=0.

| card | pipe_host us | event us |
|---|---:|---:|
| 0 | 38.010 | 37.630 |
| 1 | 38.032 | 37.747 |

38.0 us both at 2800. Spread
~0.06%. Wash vs C=2048 37.7
not 3x. Occupancy.

## ESIMD delta T=64 both cards (2026-09-03fs/ft)

backend sycl+l0, AOT gdn_delta_t.
T=64 nv=48 dv=128 dk=128.
spin=4000. cosine=1 max_abs=
1.5e-5 cosine_o=1 max_abs_o=
2.4e-4 ok=1. cur=2800 throttle=1.

| card | pipe_host us | event us | act |
|---|---:|---:|---:|
| 0 | 271.249 | 271.950 | 2583 |
| 1 | 264.906 | 266.430 | 2633 |

265-271 us both. Spread ~2.4%.
~37x decode 7.1 not 64x. Do not
freeze 265 as 2800.

## ESIMD delta T=256 both cards (2026-09-03fv/fw)

backend sycl+l0, same
gdn_delta_t. T=256 spin=4000.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. cur=2800 throttle=1.

| card | pipe_host us | event us | act |
|---|---:|---:|---:|
| 0 | 1109.372 | 1112.104 | 2617 |
| 1 | 1099.419 | 1101.310 | 2650 |

1100-1109 us both. Spread ~0.9%.
~4.1x T=64. Prefill leftover.
Do not freeze 1100 as 2800.

## ESIMD conv1d T=64 C=6144 both cards (2026-09-03fx/ga)

backend sycl+l0, same
gdn_conv1d_t. C=6144 T=64
spin=4000. cosine=1 max_abs=0
ok=1. timed act=cur=2800
throttle=0.

| card | pipe_host us | event us | GBs |
|---|---:|---:|---:|
| 0 | 10.449 | 9.880 | 155 |
| 1 | 10.231 | 9.872 | 159 |

10.2-10.4 us both. Spread ~2.1%.
Wash vs C=2048 10.1 not 3x.
Occupancy.

## ESIMD mixer T=64 both cards (2026-09-03fy/fz)

backend sycl+l0, AOT gdn_mixer_t.
T=64 C=10240 nv=48 spin=4000.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. cur=2800 throttle=1.

| card | pipe_host us | event us | act |
|---|---:|---:|---:|
| 0 | 399.311 | 400.487 | 2633 |
| 1 | 394.542 | 395.102 | 2667 |

395-399 us both. Spread ~1.2%.
~1.45x sequential ~275. Stop
two-kernel packed mixer at
prefill. Do not freeze 395 as
2800.

## ESIMD conv1d T=64 C=10240 both cards (2026-09-03gb/gc)

backend sycl+l0, same
gdn_conv1d_t. C=10240 T=64
spin=4000. cosine=1 max_abs=0
ok=1. timed act=cur=2800
throttle=0.

| card | pipe_host us | event us | GBs |
|---|---:|---:|---:|
| 0 | 10.667 | 10.128 | 253 |
| 1 | 10.535 | 10.107 | 257 |

10.5-10.7 us both. Spread ~1.3%.
Wash vs C=2048 10.1 not 5x.
Occupancy.

## ESIMD conv1d T=256 C=10240 both cards (2026-09-03gd/ge)

backend sycl+l0, same
gdn_conv1d_t. C=10240 T=256
spin=4000. cosine=1 max_abs=0
ok=1. timed act=cur=2800
throttle=0.

| card | pipe_host us | event us | GBs |
|---|---:|---:|---:|
| 0 | 40.797 | 40.344 | 259 |
| 1 | 40.742 | 40.393 | 259 |

40.7-40.8 us both. Spread ~0.13%.
~1.07x C=6144 38.0 not 5x.
Occupancy.

## ESIMD chunk/WY delta T=256 C=16 card1 (2026-09-03gf)

backend sycl+l0, AOT
gdn_delta_chunk. T=256 C=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 3210.272
event 3199.857. 4.91 GB/s.
~2.92x fused 1100. Stop C=16
vs fused. One-card.

## ESIMD fused delta T=256 hold retry card1 (2026-09-03gg)

backend sycl+l0, same
gdn_delta_t. T=256 spin=4000.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 1085.686 event
1088.060. act=2683 cur=2800
throttle=1. vs fw 1099. Cannot
hold 2800. Do not freeze 1086
as 2800.

## ESIMD chunk/WY delta T=256 C=64 card0 (2026-09-03gh)

backend sycl+l0, AOT
gdn_delta_chunk64. T=256 C=64
spin=0. cosine=1 max_abs=3.1e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 95419.883
event 95413.974. 0.17 GB/s.
~88x fused 1086, ~30x C=16.
Stop C=64. Stop this WY path.

## ESIMD fused delta T=256 SLM-K card0 (2026-09-03gi)

backend sycl+l0, AOT
gdn_delta_slmk. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 847.280 event
840.529. 18.6 GB/s. act 2783-
2767 throttle=1 at end. ~1.28x
fused 1086. Do not freeze 847.
Sibling before promote.

## ESIMD fused delta T=256 row-block rb=4 card1 (2026-09-03gj)

backend sycl+l0, AOT
gdn_delta_rowb. T=256 rb=4
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 1034.092
event 1034.237. 15.3 GB/s.
~1.05x fused 1086, loses to
SLM-K 847.

K7 next: sibling SLM-K vs
row-block rb=8.
