# K7 GDN inventory 2026-09-03ep/gt

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

## ESIMD fused delta T=256 SLM-K both cards (2026-09-03gi/gk)

backend sycl+l0, AOT
gdn_delta_slmk. T=256 blk=16.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. throttle=1 both.

| card | spin | pipe_host us | event us | act |
|---|---:|---:|---:|---:|
| 0 | 0 | 847.280 | 840.529 | 2767 |
| 1 | 4000 | 858.215 | 859.891 | 2700 |

847-858 us both. Spread ~1.3%.
~1.27x fused 1086. New leftover
class. Do not freeze 847 as 2800.

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

## ESIMD fused delta T=256 row-block rb=8 card0 (2026-09-03gl)

backend sycl+l0, AOT
gdn_delta_rowb8. T=256 rb=8
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 2060.439
event 2058.990. 7.7 GB/s.
~2x rb=4 1034, ~2.4x SLM-K 847.
Stop rb=8.

## ESIMD fused delta T=256 SLM-K+rb=4 card0 (2026-09-03gm)

backend sycl+l0, AOT
gdn_delta_slmk_rb4. T=256 blk=16
rb=4 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=2.4e-4 ok=1. timed
act=cur=2800 throttle=0.
pipe_host 998.817 event 998.307.
15.8 GB/s. ~1.18x SLM-K 847.
Stop combine vs SLM-K.

## ESIMD fused delta T=256 SLM-K blk=32 both cards (2026-09-03gn/go)

backend sycl+l0, AOT
gdn_delta_slmk32. T=256 blk=32.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. throttle=1 both.

| card | spin | pipe_host us | event us | act |
|---|---:|---:|---:|---:|
| 1 | 0 | 831.953 | 824.901 | 2750 |
| 0 | 4000 | 862.027 | 863.659 | 2650 |

832-862 us both. Spread ~3.6%.
Wash vs blk=16 847-858. Do not
freeze 832 as 2800.

## ESIMD fused delta T=256 SLM-K blk=64 card1 (2026-09-03gp)

backend sycl+l0, AOT
gdn_delta_slmk64. T=256 blk=64
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 834.985 event
827.867. 18.9 GB/s. act 2800-
2733 throttle=1 at end. Wash vs
blk=32 832. Stop larger blk.

## ESIMD fused delta T=64 SLM-K both cards (2026-09-03gq/gs)

backend sycl+l0, same
gdn_delta_slmk. T=64 blk=16.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1.

| card | spin | pipe_host us | event us | act | thr |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 214.083 | 213.732 | 2800 | 0 |
| 1 | 4000 | 217.843 | 219.055 | 2750 | 1 |

214-218 us both. Spread ~1.8%.
~1.23x fused 265. T-linear vs
847. Do not freeze 214 as 2800.

## ESIMD fused delta T=256 SLM a/b card1 (2026-09-03gr)

backend sycl+l0, AOT
gdn_delta_slmab. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 853.663 event
848.302. 18.5 GB/s. act 2783-
2750 throttle=1. Wash vs SLM-K
847-858. Stop a/b SLM.

## ESIMD fused delta T=256 v-prefetch card0 (2026-09-03gt)

backend sycl+l0, AOT
gdn_delta_slmv. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 873.078 event
866.299. 18.1 GB/s. act 2783-
2750 throttle=1. ~1.03x SLM-K
847. Stop v-prefetch vs SLM-K.

## ESIMD fused delta T=1 SLM-K blk=1 card0 (2026-09-03gu)

backend sycl+l0, AOT
gdn_delta_slmk1. T=1 blk=1
spin=4000. cosine=1 max_abs=1.2e-4
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 8.149
event 7.836. 392 GB/s. ~1.16x
fused 7.1. Event wash vs fused
7.825. Stop SLM-K vs fused at
decode.

## ESIMD fused delta T=256 SLM-K inner unroll card1 (2026-09-03gv)

backend sycl+l0, AOT
gdn_delta_slmku. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 856.296 event
853.003. 18.4 GB/s. act 2800-
2783 throttle=1. Wash vs SLM-K
847-858. Stop inner unroll.

## ESIMD fused delta T=256 SLM f32 k/q card0 (2026-09-03gw)

backend sycl+l0, AOT
gdn_delta_slmf32. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 867.995 event
860.399. 18.2 GB/s. act 2800-
2767 throttle=1. ~1.02x SLM-K
847. Stop f32 SLM vs half.

## ESIMD fused delta T=256 SLM double-buffer card1 (2026-09-03gx)

backend sycl+l0, AOT
gdn_delta_slmdb. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 842.973 event
833.992. 18.7 GB/s. act 2783-
2733 throttle=1. Wash vs SLM-K
847-858. Do not freeze 843 as
2800. Stop double-buffer.

## ESIMD fused delta T=256 SLM-K tree hsum card0 (2026-09-03gy)

backend sycl+l0, AOT
gdn_delta_slmh. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 425.689 event
416.516. 37.1 GB/s. act 2800-
2700 throttle=1. ~1.99x SLM-K
847. New leftover class. Do
not freeze 426 as 2800. Sibling
before promote.

## ESIMD fused delta T=16 SLM-K both cards (2026-09-03gz/ha)

backend sycl+l0, same
gdn_delta_slmk. T=16 blk=16.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1.

| card | spin | pipe_host us | event us | act | thr |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 58.130 | 200.466 | 550-2800 | 0 |
| 0 | 4000 | 58.600 | 58.286 | 2767 | 1 |

58 us both. Spread ~0.8%. Near
T-linear 53. Do not freeze 58
as 2800.

## ESIMD fused delta T=256 tree hsum sibling card1 (2026-09-03hb)

backend sycl+l0, same
gdn_delta_slmh. T=256 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 477.332 event
478.950 vs card0 425.689.
Spread ~12%. act=2417 cur=2800
throttle=1. Clock spread. 426-
477 us both. Do not freeze 426
as 2800.

## ESIMD fused delta T=64 tree hsum card0 (2026-09-03hc)

backend sycl+l0, same
gdn_delta_slmh. T=64 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. timed act=cur=2800
throttle=0. pipe_host 108.823
event 109.112. 57.9 GB/s.
~1.97x SLM-K 214. Napkin 107.
Sibling before promote.

## ESIMD fused delta T=1 tree hsum card1 (2026-09-03hd)

backend sycl+l0, AOT
gdn_delta_h. T=1 spin=4000.
cosine=1 max_abs=0.0625
cosine_o=1 max_abs_o=2 ok=1.
timed act=cur=2800 throttle=0.
pipe_host 6.085 event 7.237.
525 GB/s. ~1.16x fused 7.1.
max_abs_o=2 vs fused 0. Do not
replace fused 7.1.

## ESIMD fused delta T=16 tree hsum card0 (2026-09-03he)

backend sycl+l0, same
gdn_delta_slmh. T=16 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 33.801 event
34.380. 116 GB/s. act=2583
cur=2800 throttle=1. ~1.72x
SLM-K 58. Napkin 29. Do not
freeze 34 as 2800. Sibling
before promote.

## ESIMD fused delta T=64 tree hsum sibling card1 (2026-09-03hf)

backend sycl+l0, same
gdn_delta_slmh. T=64 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 124.803 event
124.950 vs card0 108.823.
Spread ~15%. act=2450 cur=2800
throttle=1. Clock spread. 109-
125 us both. Do not freeze 109
as 2800.

## ESIMD fused delta T=256 tile-fused reduce card0 (2026-09-03hg)

backend sycl+l0, AOT
gdn_delta_slmht. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 260.132 event
263.562. 60.7 GB/s. act=2600
cur=2800 throttle=0. ~1.64x
tree hsum 426. New leftover
class. Do not freeze 260 as
2800. Sibling before promote.

## ESIMD fused delta T=16 tree hsum sibling card1 (2026-09-03hh)

backend sycl+l0, same
gdn_delta_slmh. T=16 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 33.683 event
34.336 vs card0 33.801. Spread
~0.3%. act=2633 cur=2800
throttle=1. 34 us both. Do not
freeze 34 as 2800.

## ESIMD fused delta T=64 tile-fused card0 (2026-09-03hi)

backend sycl+l0, same
gdn_delta_slmht. T=64 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 66.704 event
154.266 (ramp). timed_begin
act=cur=550. timed_end act=2700
throttle=0. event min 64.
~1.63x tree hsum 109. Napkin
66. Do not freeze 67 as 2800.
Hold retry.

## ESIMD fused delta T=256 tile-fused sibling card1 (2026-09-03hj)

backend sycl+l0, same
gdn_delta_slmht. T=256 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 294.043 event
296.854 vs card0 260.132.
Spread ~13%. act 2283-2300
cur=2800 throttle=1. Clock
spread. 260-294 us both. Do
not freeze 260 as 2800.

## ESIMD fused delta T=16 tile-fused card0 (2026-09-03hk)

backend sycl+l0, same
gdn_delta_slmht. T=16 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 21.712 event
22.143. 181 GB/s. act 2617-
2600 cur=2800 throttle=1.
~1.56x tree hsum 34. Napkin 21.
Do not freeze 22 as 2800.
Sibling before promote.

## ESIMD fused delta T=64 tile-fused hold card1 (2026-09-03hl)

backend sycl+l0, same
gdn_delta_slmht. T=64 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 76.650 event
77.055 vs card0 66.704. Spread
~15%. act=2367 cur=2800
throttle=1. Clock spread. 67-77
us both. Do not freeze 67 as
2800.

## ESIMD fused delta T=256 slmht tt unroll card0 (2026-09-03hm)

backend sycl+l0, AOT
gdn_delta_slmhtu. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 275.567 event
272.823. 57.3 GB/s. act 2800-
2767 throttle=1. ~1.06x slmht
260. Stop inner unroll vs
slmht.

## ESIMD fused delta T=16 tile-fused sibling card1 (2026-09-03hn)

backend sycl+l0, same
gdn_delta_slmht. T=16 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 21.286 event
23.773 vs card0 21.712. Spread
~2%. act=2633 cur=2800
throttle=1. 22 us both. Do not
freeze 22 as 2800.

## ESIMD fused delta T=256 slmht pack a/b/v card0 (2026-09-03ho)

backend sycl+l0, AOT
gdn_delta_slmhtp. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 266.427 event
267.690. 59.2 GB/s. act 2600-
2650 cur=2800 throttle=0.
~1.02x slmht 260. Stop pack
a/b/v vs slmht.

## ESIMD fused delta T=1 tile-fused card1 (2026-09-03hp)

backend sycl+l0, AOT
gdn_delta_ht. T=1 spin=4000.
cosine=1 max_abs=0.03125
cosine_o=1 max_abs_o=2 ok=1.
pipe_host 5.542 event 6.437.
577 GB/s. act=cur=2800
throttle=0. ~1.28x fused 7.1.
Do not replace fused 7.1.

## ESIMD fused delta T=256 slmht blk=8 card0 (2026-09-03hq)

backend sycl+l0, AOT
gdn_delta_slmht8. T=256 blk=8
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 269.210 event
274.242. 58.6 GB/s. act 2700-
2667 cur=2800 throttle=0.
~1.04x slmht 260. Stop blk=8
vs slmht.

## ESIMD fused delta T=1 tile-fused scalar hsum card1 (2026-09-03hr)

backend sycl+l0, AOT
gdn_delta_hts. T=1 spin=4000.
cosine=1 max_abs=0.0625
cosine_o=1 max_abs_o=2 ok=1.
pipe_host 6.088 event 6.630.
525 GB/s. act=cur=2800
throttle=0. Wash vs tree hsum
6.09. max_abs_o=2. Stop scalar
hsum vs reduce.

## ESIMD fused delta T=256 slmht blk=32 card0 (2026-09-03hs)

backend sycl+l0, AOT
gdn_delta_slmht32. T=256 blk=32
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 252.173 event
256.893. 62.6 GB/s. act=2600
cur=2800 throttle=0. ~1.03x
slmht 260 at 2600. Possible
cut. Do not freeze 252 as 2800.
Sibling before promote.

## ESIMD fused delta T=256 slmht packed-o card1 (2026-09-03ht)

backend sycl+l0, AOT
gdn_delta_slmhto. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 247.158 event
251.826. 63.8 GB/s. act=2700
cur=2800 throttle=0. ~1.05x
slmht 260 at 2600. Clock 2700
vs 2600. Do not freeze 247 as
2800. Sibling before promote.

## ESIMD fused delta T=256 packed-o sibling card0 (2026-09-03hu)

backend sycl+l0, same
gdn_delta_slmhto. T=256 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 296.793 event
298.659 vs card1 247.158.
Spread ~20%. act=2233 cur=2800
throttle=1. Clock-linear. Stop
packed-o vs slmht. Do not
freeze 247 as 2800.

## ESIMD fused delta T=256 slmht blk=32 sibling card1 (2026-09-03hv)

backend sycl+l0, same
gdn_delta_slmht32. T=256 blk=32
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 292.134 event
294.375 vs card0 252.173.
Spread ~16%. act 2233-2250
cur=2800 throttle=1. Clock-
linear. Do not freeze 252 as
2800.

## ESIMD fused delta T=256 slmht 2-row card0 (2026-09-03hw)

backend sycl+l0, AOT
gdn_delta_slmht2. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 327.459 event
327.638. 48.2 GB/s. act=cur=2800
throttle=0. ~1.26x slmht 260.
Stop 2-row vs slmht.

## ESIMD fused delta T=64 slmht blk=32 card1 (2026-09-03hx)

backend sycl+l0, same
gdn_delta_slmht32. T=64 blk=32
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 66.502 event
155.943 (ramp). timed_begin
act=cur=550. timed_end act=2700
throttle=0. event min 64. Wash
vs slmht T=64 67. Stop blk=32
at T=64. Do not freeze 67 as
2800.

## ESIMD fused delta T=256 slmht SLM-db card0 (2026-09-03hy)

backend sycl+l0, AOT
gdn_delta_slmhtdb. T=256 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 268.173 event
268.232. 58.8 GB/s. act 2600-
2550 cur=2800 throttle=0.
~1.03x slmht 260. Stop SLM-db
vs slmht.

## ESIMD fused delta T=8 tile-fused card1 (2026-09-03hz)

backend sycl+l0, same
gdn_delta_slmht8. T=8 blk=8
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 12.526 event
14.859. 283 GB/s. act=2750
cur=2800 throttle=1. ~1.76x
fused 7.1. Near half T=16 22.
Do not freeze 13 as 2800.
Sibling before citing the map.

## ESIMD fused delta T=8 tile-fused sibling card0 (2026-09-03ia)

backend sycl+l0, same
gdn_delta_slmht8. T=8 blk=8
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 12.393 event
13.482 vs card1 12.526. Spread
~1%. act=2733 cur=2800
throttle=1. 13 us both. Do not
freeze 13 as 2800.

## ESIMD fused delta T=32 tile-fused card1 (2026-09-03ib)

backend sycl+l0, same
gdn_delta_slmht. T=32 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 38.968 event
40.987. 121 GB/s. act 2483-
2467 cur=2800 throttle=1.
Napkin 40. Do not freeze 39 as
2800. Sibling before citing
the map.

## ESIMD fused delta T=32 tile-fused sibling card0 (2026-09-03ic)

backend sycl+l0, same
gdn_delta_slmht. T=32 blk=16
spin=4000. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 39.398 event
40.727 vs card1 38.968. Spread
~1%. act=2450 cur=2800
throttle=1. 39 us both. Do not
freeze 39 as 2800.

## ESIMD fused delta T=128 slmht32 card1 (2026-09-03id)

backend sycl+l0, same
gdn_delta_slmht32. T=128 blk=32
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 124.610 event
126.724. 75.9 GB/s. act=cur=2700
throttle=0. Napkin 130. Do not
freeze 125 as 2800. Sibling
before citing the map.

## ESIMD fused delta T=128 slmht32 sibling card0 (2026-09-03ie)

backend sycl+l0, same
gdn_delta_slmht32. T=128 blk=32
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 129.673 event
130.378 vs card1 124.610.
Spread ~4%. act=2600 cur=2800
throttle=0. 125-130 us both.
Do not freeze 125 as 2800.

## ESIMD fused delta T=128 tile-fused card1 (2026-09-03if)

backend sycl+l0, same
gdn_delta_slmht. T=128 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 126.655 event
129.042. 74.7 GB/s. act=cur=2700
throttle=0. Wash vs slmht32
125. Stop blk=32 at T=128. Do
not freeze 127 as 2800. Sibling
before citing the map.

## ESIMD fused delta T=128 tile-fused sibling card0 (2026-09-03ig)

backend sycl+l0, same
gdn_delta_slmht. T=128 blk=16
spin=0. cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 131.440 event
132.412 vs card1 126.655.
Spread ~4%. act=2600 cur=2800
throttle=0. 127-131 us both.
Do not freeze 127 as 2800.
T-map closed.

## ESIMD mixer T=256 card1 (2026-09-03ih)

backend sycl+l0, same
gdn_mixer_t. T=256 C=10240
spin=0. cosine=1 max_abs=7.6e-6
cosine_o=1 max_abs_o=2.4e-4
ok=1. pipe_host 1557.055 event
1539.995. act 2750-2667
cur=2800 throttle=1. ~5.2x seq
298. Stop packed mixer at
T=256. Do not freeze 1557 as
2800.

## ESIMD mixer-slmht T=256 card0 (2026-09-03ii)

backend sycl+l0, standalone
gdn_mixer_slmht. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
470.656 event 470.435. act
2800-2783 cur=2800 throttle=0.
~3.31x packed 1557, ~1.58x seq
298. First fuse. Sibling before
promote.

## ESIMD skip-hi T=256 card1 (2026-09-03ij)

backend sycl+l0, standalone
gdn_delta_skiphi. T=256 blk=16
skip_frac=0.5 even_t_b=0 spin=0.
cosine=1 max_abs=3.1e-5
cosine_o=1 max_abs_o=1.2e-4
ok=1. pipe_host 329.899 event
329.021. 47.8 GB/s. act=cur=2800
throttle=0. ~1.27x slmht 260.
Stop skip-hi vs slmht leftover.

## ESIMD mixer-slmht T=64 card0 (2026-09-03ik)

backend sycl+l0, same
gdn_mixer_slmht. T=64 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
117.467 event 117.503. act=2683
cur=2800 throttle=1. ~3.36x
packed 395, ~1.53x seq 77,
T-linear vs 471. Do not freeze
117 as 2800. Sibling before
citing the map.

## ESIMD mixer-slmht T=256 sibling card1 (2026-09-03il)

backend sycl+l0, same
gdn_mixer_slmht. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
470.966 event 471.367 vs card0
470.656. Spread ~0.07%.
act=cur=2800 throttle=0. 471 us
both at 2800. Promote.

## ESIMD mixer-slmht T=128 card0 (2026-09-03im)

backend sycl+l0, same
gdn_mixer_slmht. T=128 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
232.321 event 261.862. act
1600-2800 cur 1583-2800
throttle=0. Napkin 235.
T-linear vs 117 and 471. Do
not freeze 232 as 2800. Sibling
before citing the map.

## ESIMD mixer-slmht T=64 sibling card1 (2026-09-03in)

backend sycl+l0, same
gdn_mixer_slmht. T=64 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
116.017 event 116.143 vs card0
117.467. Spread ~1.2%. act=2717
cur=2800 throttle=1. 116-117 us
both. Do not freeze 117 as 2800.

## ESIMD mixer-slmht T=32 card0 (2026-09-03io)

backend sycl+l0, same
gdn_mixer_slmht. T=32 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
59.779 event 59.057. act=2683
cur=2800 throttle=1. Napkin 58.
T-linear vs 117. Do not freeze
60 as 2800. Sibling before
citing the map.

## ESIMD mixer-slmht T=128 sibling card1 (2026-09-03ip)

backend sycl+l0, same
gdn_mixer_slmht. T=128 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
232.288 event 232.526 vs card0
232.321. Spread ~0.01%.
act=cur=2800 throttle=0. 232 us
both, card1 at 2800.

## ESIMD mixer-slmht T=16 card0 (2026-09-03iq)

backend sycl+l0, same
gdn_mixer_slmht. T=16 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
31.295 event 31.188. act=2733
cur=2800 throttle=1. Napkin 29.
~1.92x T=32 60, ~1.42x slmht
22. Do not freeze 31 as 2800.
Sibling before citing the map.

## ESIMD mixer-slmht T=32 sibling card1 (2026-09-03ir)

backend sycl+l0, same
gdn_mixer_slmht. T=32 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
59.233 event 59.713 vs card0
59.779. Spread ~0.9%. act=2717
cur=2800 throttle=1. 59-60 us
both. Do not freeze 60 as 2800.

## ESIMD conv T=16 C=10240 card0 (2026-09-03is)

backend sycl+l0, same
gdn_conv1d_t. T=16 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 5.710 event
5.242. 129 GB/s. act=1650
cur=1617 throttle=0. T-linear
vs T=64 10.5. seq ~28 vs mixer
31. Do not freeze 5.7 as 2800.
Sibling hold.

## ESIMD mixer-slmht T=16 sibling card1 (2026-09-03it)

backend sycl+l0, same
gdn_mixer_slmht. T=16 C=10240
nv=48 blk=16 spin=4000. cosine=1
max_abs=3.1e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
31.184 event 32.039 vs card0
31.295. Spread ~0.4%. act=2750
cur=2800 throttle=1. 31 us
both. T-map blk=16 closed. Do
not freeze 31 as 2800.

## ESIMD conv T=32 C=10240 card0 (2026-09-03iu)

backend sycl+l0, same
gdn_conv1d_t. T=32 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 5.937 event
5.401. 235 GB/s. act=cur=2800
throttle=0. seq ~45 vs mixer
60. Sibling before citing the
map.

## ESIMD conv T=16 C=10240 hold card1 (2026-09-03iv)

backend sycl+l0, same
gdn_conv1d_t. T=16 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 4.833 event
3.065. 153 GB/s. act=cur=2800
throttle=0. vs card0 5.710 at
1650. Spread ~18% clock. 4.8 us
at 2800. seq ~27 vs mixer 31.

## ESIMD conv T=128 C=10240 card0 (2026-09-03iw)

backend sycl+l0, same
gdn_conv1d_t. T=128 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 19.946 event
19.542. 267 GB/s. act=cur=2800
throttle=0. Napkin 20. T-linear
vs T=64 10.5 and T=256 40.7.
seq ~147 vs mixer 232. Sibling
before citing the map.

## ESIMD conv T=32 C=10240 sibling card1 (2026-09-03ix)

backend sycl+l0, same
gdn_conv1d_t. T=32 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 5.771 event
5.406 vs card0 5.937. Spread
~2.8%. act=cur=2800 throttle=0.
5.8-5.9 us both at 2800. seq
~45 vs mixer 60.

## ESIMD mixer L2-out T=256 card0 (2026-09-03iy)

backend sycl+l0, standalone
gdn_mixer_l2out. T=256 C=10240
nv=48 blk=16 host-L2 spin=0.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=9.8e-4
ok=1. pipe_host 270.767 event
271.320. act 2700-2650 cur=2800
throttle=0. Wash vs slmht 260.
Packed tax ~4%. Device L2 is
the mixer leftover vs 471. Do
not freeze 271 as 2800. Sibling
before promote.

## ESIMD conv T=128 C=10240 sibling card1 (2026-09-03iz)

backend sycl+l0, same
gdn_conv1d_t. T=128 C=10240 k=4
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 19.929 event
19.539 vs card0 19.946. Spread
~0.09%. act=cur=2800 throttle=0.
20 us both at 2800. T-map
C=10240 closed.

## ESIMD mixer L2-once T=256 card0 (2026-09-03ja)

backend sycl+l0, standalone
gdn_mixer_l2once. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
326.779 event 319.096. act
2700-2667 cur=2800 throttle=0.
Beats mixer 471, loses to seq
298. Extra launch. Do not
freeze 327 as 2800. Sibling
before promote.

## ESIMD mixer L2-out T=256 sibling card1 (2026-09-03jb)

backend sycl+l0, same
gdn_mixer_l2out. T=256 C=10240
nv=48 blk=16 host-L2 spin=0.
cosine=1 max_abs=1.5e-5
cosine_o=1 max_abs_o=9.8e-4
ok=1. pipe_host 266.844 event
271.094 vs card0 270.767.
Spread ~1.5%. act 2700-2767
cur=2800 throttle=1. 267-271 us
both. Do not freeze 271 as 2800.

## ESIMD mixer L2-once T=256 sibling card1 (2026-09-03jc)

backend sycl+l0, same
gdn_mixer_l2once. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
312.274 event 317.284 vs card0
326.779. Spread ~4.4%. act
2700-2767 cur=2800 throttle=1.
312-327 us both. 327-class.
Beats mixer 471, loses to seq
298. Extra launch. Do not freeze
327 as 2800.

K7 next: mixer conv-L2 fuse
T=256.

## ESIMD mixer conv-L2 fuse T=256 card0 (2026-09-03jd)

backend sycl+l0, standalone
gdn_mixer_convl2. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
358.198 event 357.836. act=2700
cur=2800 throttle=0. vs L2-once
327 (~1.10x) vs seq 298. Per-t
SLM barriers. Stop this fuse.
Do not freeze 358 as 2800.

K7 next: packed qkv ESIMD s8
vs W8A8 96/140/164.

## ESIMD packed qkv s8 M=1 card0 (2026-09-03je)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=10240 k=5120
NT=2 U=16 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
73.945 event 73.078. timed
act=cur=2800 throttle=0. vs
oneDNN packed qkv W8A8 96
(~0.77x, a beat) vs square s8
34 (~2.17x, N=2x napkin 68).
Held 2800. One-card.

K7 next: packed qkv ESIMD s8
M=64 vs W8A8 140.

## ESIMD packed qkv s8 M=64 card1 (2026-09-03jf)

backend sycl+l0, standalone
dpas_s8_sc8db48. m=64 n=10240
k=5120 NT=2 U=16 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 214.369 event
212.604. timed act=cur=2800
throttle=0. vs W8A8 138-142
(~1.53x) vs square 75 (~2.86x)
vs napkin 150. Loss vs oneDNN.
Do not freeze 214 until sibling.

K7 next: packed qkv ESIMD s8
M=256 vs W8A8 164, and sibling
M=64 card0.

## ESIMD packed qkv s8 M=1 sibling card1 (2026-09-03jg)

backend sycl+l0, same dpas_s8_sc.
M=1 n=10240 k=5120 NT=2 U=16
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 73.782 event
73.214 vs card0 73.945. Spread
~0.2%. timed act=cur=2800
throttle=0. 74-class both at
2800. Beats W8A8 96. Held 2800.

K7 next: packed qkv ESIMD s8
M=256 vs W8A8 164, and sibling
M=64 card0.

## ESIMD packed qkv s8 M=256 card0 (2026-09-03jh)

backend sycl+l0, standalone
dpas_s8_sc8w48m4. m=256 n=10240
k=5120 NT=2 U=8 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 274.205 event
272.672. timed act 2733-2717
cur=2800 throttle=1. vs W8A8
164 (~1.67x) vs square 128
(~2.14x) vs napkin 256. Loss
vs oneDNN. Stop this tile vs
164. Do not freeze 274 as
2800. One-card. Do not promote.

K7 next: packed qkv M=64/M=256
still oneDNN. Do not sibling
the 214/274 losses. o-proj
still 47. Seq 298 is the T=256
mixer leftover.

## ESIMD mixer conv-L2-register T=256 card0 (2026-09-03ji)

backend sycl+l0, standalone
gdn_mixer_convl2r. T=256 C=10240
nv=48 blk=16 spin=0. cosine=1
max_abs=1.5e-5 cosine_o=1
max_abs_o=9.8e-4 ok=1. pipe_host
530.777 event 522.625. act
2700-2600 cur=2800 throttle=0.
vs L2-once 327 (~1.62x) vs seq
298 (~1.78x) vs conv-L2 358
(~1.48x) vs mixer 471 (~1.13x).
Stop this mapping. One-card.
Do not freeze 531 as 2800.
Do not promote.

K7 next: packed qkv M=64/M=256
still oneDNN. Do not sibling
the 214/274 losses. o-proj
still 47. Seq 298 is the T=256
mixer leftover.

## ESIMD o-proj s8 M=1 card1 (2026-09-03jj)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=5120 k=6144
NT=2 U=16 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
62.285 event 61.719. timed
act=cur=2800 throttle=0. vs
oneDNN o-proj W8A8 46-47
(~1.33x, a loss) vs square s8
34 (~1.83x, K=1.2x napkin 41).
Held 2800. One-card. Stop this
tile vs 47. Do not promote.

K7 next: packed qkv M=64/M=256
still oneDNN. Do not sibling
the 214/274 losses. o-proj
still 47. Seq 298 is the T=256
mixer leftover.

## ESIMD q-proj s8 M=1 card0 (2026-09-03jk)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=2048 k=5120
NT=2 U=16 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
27.714 event 27.234. timed
act=cur=2800 throttle=0. vs
oneDNN q W8A8 45-58 (~0.61x
of 45, a beat) vs packed qkv
s8 74 (~0.375x) vs square s8
34 (~0.82x) vs napkin 14
(~2.0x). Held 2800. One-card.
Sibling before promote.

K7 next: sibling q-proj s8
card1 before promote. v-proj
s8 vs 46. packed qkv M=64/M=256
still oneDNN. o-proj still 47.
Seq 298 is the T=256 mixer
leftover.

## ESIMD v-proj s8 M=1 card1 (2026-09-03jl)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=6144 k=5120
NT=2 U=16 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
53.226 event 52.609. timed
act=cur=2800 throttle=0. vs
oneDNN v W8A8 46 (~1.16x, a
loss) vs packed qkv s8 74
(~0.72x) vs square s8 34
(~1.57x, N=1.2x napkin 41).
Held 2800. One-card. Stop this
tile vs 46. Do not promote.

K7 next: sibling q-proj s8
card1 before promote. packed
qkv M=64/M=256 still oneDNN.
o-proj still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD o-proj s8 NT=4 M=1 card0 (2026-09-03jm)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=5120 k=6144
NT=4 U=8 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
103.086 event 102.187. timed
act=cur=2800 throttle=0. vs
oneDNN o-proj W8A8 46-47
(~2.19x, a loss) vs NT=2 62
(~1.66x) vs square s8 34
(~3.03x, K=1.2x napkin 41).
Held 2800. One-card. Stop this
steal vs 47. Do not sibling.
Do not promote.

K7 next: sibling q-proj s8
card1 before promote. packed
qkv M=64/M=256 still oneDNN.
o-proj still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD q-proj s8 M=1 sibling card1 (2026-09-03jn)

backend sycl+l0, same dpas_s8_sc.
M=1 n=2048 k=5120 NT=2 U=16
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 27.666 event
27.258 vs card0 27.714. Spread
~0.2%. timed act=cur=2800
throttle=0. 28-class both at
2800. Beats oneDNN q 45-58.
q 27.7, k same shape 27.7,
v 53.2 (jl), sum ~108.6 vs
packed qkv s8 74. Packed 74
still beats sequential q+k+v
~109. Held 2800.

K7 next: packed qkv s8 74 stays
the M=1 qkv leftover vs split
~109. packed qkv M=64/M=256
still oneDNN. o-proj still 47.
v-proj still 46. Seq 298 is
the T=256 mixer leftover.

## ESIMD packed qkv s8 NT=4 M=1 card0 (2026-09-03jo)

backend sycl+l0, standalone
dpas_s8_sc. M=1 n=10240 k=5120
NT=4 U=8 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
105.746 event 105.193. timed
act=cur=2800 throttle=0. vs
NT=2 packed qkv s8 74 (~1.43x,
a loss) vs W8A8 96 (~1.10x).
NT=4 o-proj was 103 (loss);
this N is 2x wider. Held 2800.
One-card. Stop this steal vs
74. Do not sibling. Do not
promote.

K7 next: packed qkv s8 74 stays
the M=1 qkv leftover vs split
~109. packed qkv M=64/M=256
still oneDNN. o-proj still 47.
v-proj still 46. Seq 298 is
the T=256 mixer leftover.

## ESIMD packed qkv A=s4 M=1 card1 (2026-09-03jp)

backend sycl+l0, standalone
dpas_s4_sc. A=s4 M=1 n=10240
k=5120 NT=2 U=16 spin=4000.
A=s4 cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 16.637 event
16.250. A=s4 timed act=cur=2800
throttle=0. A=s4 vs napkin 33
(~0.50x, occupancy) vs packed
s8 74 (~0.23x wall) vs W8A8 96
(~0.17x wall, not a W8A8 beat)
vs square A=s4 16.5 (wash).
A=s4 held 2800. A=s4 one-card.
Sibling before FINDINGS.

K7 next: sibling A=s4 packed
qkv M=1 card0 before promote.
packed qkv s8 74 stays the s8
M=1 leftover. packed qkv
M=64/M=256 still oneDNN.
o-proj still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD packed qkv A=s4 M=1 sibling card0 (2026-09-03jq)

backend sycl+l0, same dpas_s4_sc.
A=s4 M=1 n=10240 k=5120 NT=2
U=16 spin=4000. A=s4 cosine=1
max_abs=0 ok=1. A=s4 pipe_host
16.607 event 16.229 vs card1
16.637. Spread ~0.2%. A=s4
timed act=cur=2800 throttle=0.
A=s4 17-class both at 2800.
A=s4 wash vs square 16.5.
Occupancy not N-linear. A=s4
not a W8A8-contract beat. vs
packed s8 74 is wall-time
only. Held 2800.

K7 next: packed qkv s8 74 stays
the s8 M=1 leftover vs split
~109. packed qkv M=64/M=256
still oneDNN. o-proj still 47.
v-proj still 46. Seq 298 is
the T=256 mixer leftover.

## ESIMD packed qkv A=s4 M=64 card1 (2026-09-03jr)

backend sycl+l0, standalone
dpas_s4_db48. A=s4 M=64 n=10240
k=5120 NT=2 U=16 wg=4x8 A-db
spin=512. A=s4 cosine=1
max_abs=0 ok=1. A=s4 pipe_host
63.452 event 63.130. A=s4
timed act=cur=2800 throttle=0.
A=s4 vs napkin 67 (~0.95x,
near N-linear) vs W8A8 140
(~0.45x wall, not a W8A8
beat) vs s8 214 (~0.30x wall)
vs square A=s4 33.6 (~1.89x).
A=s4 held 2800. A=s4 one-card.
New s4 packed-qkv M=64 floor.
Sibling before FINDINGS.

K7 next: sibling A=s4 packed
qkv M=64 card0 before promote.
packed qkv s8 74 stays the s8
M=1 leftover vs split ~109.
packed qkv M=64/M=256 still
oneDNN on the W8A8 contract.
o-proj still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD packed qkv A=s4 M=64 sibling card0 (2026-09-03js)

backend sycl+l0, same
dpas_s4_db48. A=s4 M=64
n=10240 k=5120 NT=2 U=16
wg=4x8 A-db spin=512. A=s4
cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 63.290 event
62.750 vs card1 63.452.
Spread ~0.3%. A=s4 timed
act=cur=2800 throttle=0.
A=s4 63-class both at 2800.
A=s4 near N-linear vs square
33.6. A=s4 not a
W8A8-contract beat. vs W8A8
140 is wall-time only. Held
2800.

K7 next: packed qkv s8 74 stays
the s8 M=1 leftover vs split
~109. packed qkv M=256 still
oneDNN on the W8A8 contract.
o-proj still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD packed qkv A=s4 M=256 card1 (2026-09-03jt)

backend sycl+l0, standalone
dpas_s4_w48m4. A=s4 M=256
n=10240 k=5120 NT=2 U=8
wg=4x8 4-acc spin=512. A=s4
cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 95.262 event
95.198. A=s4 timed
act=cur=2800 throttle=0.
A=s4 vs napkin 97 (~0.98x,
near N-linear) vs W8A8 164
(~0.58x wall, not a W8A8
beat) vs s8 274 (~0.35x wall)
vs square A=s4 48.6 (~1.96x).
A=s4 held 2800. A=s4 one-card.
New s4 packed-qkv M=256 floor.
Sibling before FINDINGS.

K7 next: sibling A=s4 packed
qkv M=256 card0 before promote.
packed qkv s8 74 stays the s8
M=1 leftover vs split ~109.
packed qkv M=256 still oneDNN
on the W8A8 contract. o-proj
still 47. v-proj still 46.
Seq 298 is the T=256 mixer
leftover.

## ESIMD o-proj A=s4 M=1 card1 (2026-09-03jv)

backend sycl+l0, standalone
dpas_s4_sc. A=s4 M=1 n=5120
k=6144 NT=2 U=16 spin=4000.
A=s4 cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 19.394 event
19.039. A=s4 timed act=cur=2800
throttle=0. A=s4 vs napkin 20
(~0.97x, near K-linear) vs s8
62 (~0.31x wall) vs W8A8 47
(~0.41x wall, not a W8A8 beat)
vs square A=s4 16.5 (~1.18x).
A=s4 held 2800. A=s4 one-card.
New s4 o-proj floor. Sibling
before FINDINGS.

K7 next: sibling A=s4 o-proj
M=1 card0 before promote.
packed qkv s8 74 stays the s8
M=1 leftover vs split ~109.
packed qkv M=256 still oneDNN
on the W8A8 contract. o-proj
W8A8 still 47. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD packed qkv A=s4 M=256 sibling card0 (2026-09-03ju)

backend sycl+l0, same
dpas_s4_w48m4. A=s4 M=256
n=10240 k=5120 NT=2 U=8
wg=4x8 4-acc spin=512. A=s4
cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 93.706 event
93.984 vs card1 95.262.
Spread ~1.7%. A=s4 timed
act=cur=2800 throttle=0.
A=s4 95-class both at 2800.
A=s4 near N-linear vs square
48.6. A=s4 not a
W8A8-contract beat. vs W8A8
164 is wall-time only. Qwen
packed-qkv s4 map M=1/64/256
closed (16.6 / 63 / 95). Held
2800.

K7 next: packed qkv s8 74 stays
the s8 M=1 leftover vs split
~109. packed qkv M=256 still
oneDNN on the W8A8 contract.
o-proj sibling A=s4 M=1 card0
before promote. v-proj still
46. Seq 298 is the T=256 mixer
leftover.

## ESIMD o-proj A=s4 M=1 sibling card0 (2026-09-03jw)

backend sycl+l0, same
dpas_s4_sc. A=s4 M=1 n=5120
k=6144 NT=2 U=16 spin=4000.
A=s4 cosine=1 max_abs=0 ok=1.
A=s4 pipe_host 19.381 event
19.005 vs card1 19.394.
Spread ~0.07%. A=s4 timed
act=cur=2800 throttle=0.
A=s4 19-class both at 2800.
A=s4 near K-linear vs square
16.5. A=s4 not a
W8A8-contract beat. vs W8A8
47 is wall-time only. Held
2800.

K7 next: packed qkv s8 74 stays
the s8 M=1 leftover vs split
~109. packed qkv M=256 still
oneDNN on the W8A8 contract.
o-proj A=s4 19-class both; vs
W8A8 47 is wall-time only.
v-proj still 46. Seq 298 is
the T=256 mixer leftover.

## ESIMD packed qkv A=s2 M=1 card1 (2026-09-03jx)

backend sycl+l0, standalone
dpas_s2_sc. A=s2 M=1 n=10240
k=5120 NT=2 U=16 spin=4000.
A=s2 cosine=1 max_abs=0 ok=1.
A=s2 pipe_host 11.675 event
11.276. A=s2 timed act=cur=2800
throttle=0. A=s2 vs square 11.5
(wash, occupancy). A=s2 vs s4
packed 16.6 (~0.70x wall).
A=s2 vs packed s8 74 (~0.16x
wall). A=s2 not a
W8A8-contract beat. A=s2 held
2800. A=s2 one-card. Sibling
before FINDINGS.

K7 next: sibling A=s2 packed
qkv M=1 card0 before promote.
packed qkv s8 74 stays the s8
M=1 leftover vs split ~109.
packed qkv M=256 still oneDNN
on the W8A8 contract. o-proj
A=s4 19-class both; vs W8A8
47 is wall-time only. v-proj
still 46. Seq 298 is the T=256
mixer leftover.

## ESIMD packed qkv A=s2 M=1 sibling card0 (2026-09-03jy)

backend sycl+l0, same
dpas_s2_sc. A=s2 M=1 n=10240
k=5120 NT=2 U=16 spin=4000.
A=s2 cosine=1 max_abs=0 ok=1.
A=s2 pipe_host 11.644 event
11.299 vs card1 11.675.
Spread ~0.27%. A=s2 timed
act=cur=2800 throttle=0.
A=s2 12-class both at 2800.
A=s2 wash vs square 11.5.
Occupancy not N-linear. A=s2
not a W8A8-contract beat. vs
s4 packed 16.6 and packed s8
74 is wall-time only. Held
2800.

K7 next: packed qkv s8 74 stays
the s8 M=1 leftover vs split
~109. packed qkv M=256 still
oneDNN on the W8A8 contract.
A=s2 packed qkv M=1 12-class
both; wash vs square 11.5.
o-proj A=s4 19-class both; vs
W8A8 47 is wall-time only.
v-proj still 46. Seq 298 is
the T=256 mixer leftover.

## ESIMD o-proj A=s2 M=1 card1 (2026-09-03jz)

backend sycl+l0, standalone
dpas_s2_sc. A=s2 M=1 n=5120
k=6144 NT=2 U=16 spin=4000.
A=s2 cosine=1 max_abs=0 ok=1.
A=s2 pipe_host 13.545 event
13.159. A=s2 timed act=cur=2800
throttle=0. A=s2 vs napkin 14
(~0.97x, near K-linear) vs s4
19 (~0.71x wall) vs W8A8 47
(~0.29x wall, not a W8A8 beat)
vs square A=s2 11.5 (~1.18x).
A=s2 held 2800. A=s2 one-card.
New s2 o-proj floor. Sibling
before FINDINGS.

## ESIMD o-proj A=s2 M=1 sibling card0 (2026-09-03ka)

backend sycl+l0, same dpas_s2_sc.
A=s2 M=1 n=5120 k=6144 NT=2
U=16 spin=4000. cosine=1
max_abs=0 ok=1. pipe_host
13.519 event 13.167 vs card1
13.545. Spread ~0.2%. timed
act=cur=2800 throttle=0.
14-class both at 2800. Near
K-linear vs square 11.5. Not
a W8A8-contract beat.

K7 next: o-proj W8A8 47 is
beaten by NT=1 SK=2 44 both.
Prefill packed 140/164 remains.
Decode s8 packed 74 stands.
Seq 298 is the T=256 mixer
leftover. Integer s2/s4 decode
maps for packed qkv and o-proj
are closed. STOP B-pipeline
decode, wg 8x4 prefill,
persist 4-acc, NT=1 wg 4x2,
NT=1 prefetch.

## ESIMD o-proj s8 NT=1 M=1 card0 (2026-09-04a)

backend sycl+l0, arm
dpas_s8_sc_nt1. NT=1 m=1
n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 55.016 event 54.443.
timed act=cur=2800 throttle=0.
vs W8A8 47 a loss; steal vs
NT=2 62. One-card. Rank
pipe_host.

## ESIMD o-proj s8 B-pipeline hail-mary persist/mainloop M=1 card1 (2026-09-04b)

backend sycl+l0, arm
dpas_s8_sc_bp. hail-mary
persist/mainloop NT=2 m=1
n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 66.892 event 66.250.
timed act=cur=2800 throttle=0.
Loss vs sc 62 and W8A8 47.
STOP this B-pipeline on
decode o-proj. One-card.

## ESIMD o-proj s8 NT=1 M=1 sibling card1 (2026-09-04c)

backend sycl+l0, same
dpas_s8_sc_nt1. NT=1 m=1
n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 55.323 event 54.784
vs card0 55.016. Spread ~0.56%.
timed act=cur=2800 throttle=0.
55-class both at 2800. Beats
sc NT=2 62. Loses to W8A8 47.
Not a leftover close. Rank
pipe_host.

## ESIMD s8 4-acc wg 8x4 k128 packed qkv M=64 leftover steal card0 (2026-09-04d)

backend sycl+l0, arm
dpas_s8_sc8w84m4. NT=2 m=64
n=10240 k=5120 wg=8x4 4acc
k128 spin=512. cosine=1
max_abs=0 ok=1. pipe_host
154.074 event 153.906. timed
act=cur=2800 throttle=0.
vs W8A8 138-142 a loss; vs
4x8 A-db 214 a win; vs square
4-acc 75 ~2.05x. One-card.
STOP this wg 8x4 at M=64.
Not a W8A8-contract beat.
Rank pipe_host.

## ESIMD s8 4-acc wg 8x4 k128 packed qkv M=256 leftover steal card0 (2026-09-04e)

backend sycl+l0, arm
dpas_s8_sc8w84m4. NT=2 m=256
n=10240 k=5120 wg=8x4 4acc
k128 spin=512. cosine=1
max_abs=0 ok=1. pipe_host
278.725 event 281.568. timed
act=2717 cur=2800 throttle=1.
vs W8A8 164 a loss; vs 4-acc
4x8 274 same class; vs square
4-acc 128 ~2.18x. One-card.
Do not freeze (not 2800).
STOP this wg 8x4 at M=256.
Not a W8A8-contract beat.
Rank pipe_host.

## ESIMD s8 4-acc B-pipeline hail-mary persist/mainloop packed qkv M=256 card1 (2026-09-04f)

backend sycl+l0, arm
dpas_s8_sc8w48m4bp. hail-mary
persist/mainloop. NT=2 m=256
n=10240 k=5120 wg=4x8 4acc
B-pipeline spin=512. cosine=1
max_abs=0 ok=1. pipe_host
343.995 event 344.135. timed
act=2700 cur=2800 throttle=1.
vs W8A8 164 (~2.10x) a loss;
vs 4-acc 4x8 274 (~1.25x) a
loss. One-card. Do not freeze
(not 2800). STOP this persist
B-pipeline on packed prefill.
Not a W8A8-contract beat.
Rank pipe_host.

## ESIMD o-proj s8 NT=1 wg 4x2 M=1 card0 (2026-09-04g)

backend sycl+l0, arm
dpas_s8_sc_nt1w42. NT=1 wg=4x2
m=1 n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 74.278 event 74.010.
timed act=cur=2800 throttle=0.
vs NT=1 8x2 55 a loss; vs
W8A8 47 a loss; vs sc NT=2 62
a loss. One-card. STOP smaller
WG on o-proj. Do not sibling.
Not a W8A8-contract beat.
Rank pipe_host.

## ESIMD o-proj s8 NT=1 wg 4x2 M=1 sibling card1 (2026-09-04h)

backend sycl+l0, arm
dpas_s8_sc_nt1w42. NT=1 wg=4x2
m=1 n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 74.636 event 72.901
vs card0 74.278. Spread ~0.48%.
timed act=cur=2800 throttle=0.
vs NT=1 8x2 55 a loss; vs
W8A8 47 a loss; vs sc NT=2 62
a loss. Both-card 74-class at
2800. STOP this wg 4x2. Do not
promote. Not a W8A8-contract
beat. Rank pipe_host.

## ESIMD o-proj s8 NT=1 B-pipeline leftover steal M=1 card0 (2026-09-04k)

backend sycl+l0, arm
dpas_s8_sc_nt1bp. NT=1
B-pipeline m=1 n=5120 k=6144
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 60.244 event
59.672. timed act=cur=2800
throttle=0. vs NT=1 55 a loss;
vs W8A8 47 a loss. Faster than
NT=2 B-pipeline 67 and slight
vs sc 62. One-card. STOP NT=1
B-pipeline on o-proj. Do not
sibling. Not a W8A8-contract
beat. Rank pipe_host.

## ESIMD s8 4-acc split-K=2 packed qkv M=64 leftover steal card1 (2026-09-04l)

backend sycl+l0, arm
dpas_s8_sc8w48m4sk. NT=2 m=64
n=10240 k=5120 splitK=2 wg=4x8
4acc k128 unroll=5 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 115.081 (gemm+reduce).
event 7.766 reduce-only.
timed act=2783 cur=2800
throttle=1. vs W8A8 138-142 a
win; vs 4x8 A-db 214 a win; vs
wg 8x4 154 a win; vs square
4-acc 75 ~1.53x. One-card.
Do not freeze (not 2800).
W8A8-contract beat of 140 on
this card. Still fire M=256.
Sibling before FINDINGS.
Rank pipe_host.

## ESIMD s8 4-acc split-K=2 packed qkv M=64 sibling card0 (2026-09-04m)

backend sycl+l0, arm
dpas_s8_sc8w48m4sk. NT=2 m=64
n=10240 k=5120 splitK=2 wg=4x8
4acc k128 unroll=5 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 117.264 vs card1
115.081. Spread ~1.90%.
event 6.531 reduce-only.
timed act=2783 cur=2800
throttle=1. vs W8A8 138-142 a
win both cards. 115-class.
Do not freeze (not 2800).
Not a decode leftover.
Rank pipe_host.

## ESIMD s8 4-acc split-K=2 packed qkv M=256 leftover steal card1 (2026-09-04n)

backend sycl+l0, arm
dpas_s8_sc8w48m4sk. NT=2 m=256
n=10240 k=5120 splitK=2 wg=4x8
4acc k128 unroll=5 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 294.564 (gemm+reduce).
event 48.417 reduce-only.
timed act=2650/2633 cur=2800
throttle=1. vs W8A8 164 a
loss (~1.80x); vs 4-acc 4x8
274 a loss (~1.07x). One-card.
Do not freeze (not 2800).
Not a W8A8-contract beat.
STOP split-K at M=256. Do not
sibling. Rank pipe_host.

## ESIMD o-proj s8 NT=1 lsc_prefetch_2d M=1 card0 (2026-09-04p)

backend sycl+l0, arm
dpas_s8_sc_nt1ff.
prefetch=lsc_prefetch_2d
m=1 n=5120 k=6144 spin=4000.
cosine=1 max_abs=0 ok=1.
pipe_host 54.797 event 54.211.
timed act=cur=2800 throttle=0.
vs NT=1 55 55-class, not a
steal; vs W8A8 47 a loss.
One-card. STOP prefetch on
NT=1 o-proj. Do not sibling.
Not a W8A8-contract beat.
Rank pipe_host.

## ESIMD s8 4-acc split-K=5 unroll=8 packed qkv M=256 leftover steal card1 (2026-09-04q)

backend sycl+l0, arm
dpas_s8_sc8w48m4sk5. NT=2 m=256
n=10240 k=5120 splitK=5 wg=4x8
4acc k128 unroll=8 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 392.811 (gemm+reduce).
event 129.562 reduce-only.
timed act=2650 cur=2800
throttle=1. vs W8A8 164 a
loss (~2.40x); vs 4-acc 4x8
274 a loss (~1.43x); vs SK=2
295 a loss (~1.33x). One-card.
Do not freeze (not 2800).
Not a W8A8-contract beat.
STOP SK=5 at M=256. Do not
sibling. Rank pipe_host.

## ESIMD o-proj s8 NT=1 split-K=2 M=1 card0 (2026-09-04r)

backend sycl+l0, arm
dpas_s8_sc_nt1sk. NT=1
splitK=2 m=1 n=5120 k=6144
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 44.348
(gemm+reduce). event 1.474
reduce-only. timed
act=cur=2800 throttle=0.
vs NT=1 55 a win (~0.81x);
vs W8A8 47 a win (~0.94x).
One-card. W8A8-contract beat
of 47 on this card. Sibling
before FINDINGS. Rank
pipe_host.

## ESIMD o-proj s8 NT=1 split-K=2 M=1 sibling card1 (2026-09-04s)

backend sycl+l0, arm
dpas_s8_sc_nt1sk. NT=1
splitK=2 m=1 n=5120 k=6144
spin=4000. cosine=1 max_abs=0
ok=1. pipe_host 44.114 vs
card0 44.348. Spread ~0.53%.
event 1.474 reduce-only.
timed act=cur=2800 throttle=0.
vs NT=1 55 a win (~0.80x);
vs W8A8 47 a win (~0.94x).
44-class both at 2800.
W8A8-contract beat of 47
both cards. Rank pipe_host.

## ESIMD s8 2-acc packed qkv M=64 leftover steal card1 (2026-09-04v)

backend sycl+l0, arm
dpas_s8_sc8w48m2. NT=2 m=64
n=10240 k=5120 wg=4x8 2acc
k128 unroll=8 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 141.647 event
140.521. timed act=cur=2800
throttle=0. vs SK=2 115 a
loss (~1.23x); vs W8A8 140 a
loss (~1.01x); vs 4x8 A-db
214 a win; vs wg 8x4 154 a
win; vs square 4-acc 75
~1.89x. One-card. Held 2800.
STOP 2-acc at M=64. Do not
sibling. Not a W8A8-contract
beat. Rank pipe_host.

## ESIMD s8 2-acc wg 4x8 k128 packed qkv M=256 leftover steal card0 (2026-09-04u)

backend sycl+l0, arm
dpas_s8_sc8w48m2. NT=2 m=256
n=10240 k=5120 wg=4x8 2acc
k128 unroll=8 spin=512.
cosine=1 max_abs=0 ok=1.
pipe_host 327.053 event
325.896. timed act=2667
cur=2800 throttle=1. vs W8A8
164 a loss (~1.99x); vs
4-acc 4x8 274 a loss (~1.19x);
vs SK=2 295 a loss (~1.11x);
vs SK=5 393 (~0.83x). One-card.
Do not freeze (not 2800).
Not a W8A8-contract beat.
STOP 2-acc at M=256. Do not
sibling. Rank pipe_host.
