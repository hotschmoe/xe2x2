# K7 GDN inventory 2026-09-03ep/ey

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

## ESIMD conv1d K=4 card0 (2026-09-03ex)

backend sycl+l0, AOT gdn_conv1d.
T=1 f16, VL=16 wg=16, spin=4000.
cosine=1 max_abs=0 ok=1.

| C | pipe_host us | event us | cur MHz |
|---|---:|---:|---:|
| 2048 | 4.350 | 1.456 | 1700 |
| 6144 | 4.799 | 1.083 | 2250 |

~26x eager 115. Clocks not 2800.
Do not freeze 4.35. One-card.

## ESIMD delta 48x128x128 card1 (2026-09-03ey)

backend sycl+l0, AOT gdn_delta.
spin=4000. cosine=1 max_abs=0.015625
(1 ulp f16) ok=1. act=cur=2800
throttle=0.

| card | pipe_host us | event us | GBs |
|---|---:|---:|---:|
| 1 | 7.093 | 8.432 | 450.48 |

~43x eager 308. Near copy 550.
One-card. Do not freeze 7.09.

K7 next: sibling swap. card0
delta, card1 conv1d.
