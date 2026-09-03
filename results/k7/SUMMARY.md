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

K7 next: pack a/b/v T=256 vs
tile-fused T=1.
