# K0 roofline 2026-09-02h

Backend `sycl+l0`, AOT `intel_gpu_bmg_g31`, icpx 2026.1.1, NEO
1.15.38646+4, GT0 cur=2800 MHz, throttle 0. Numeric copies
byte-exact. GEMM host oracle max_abs=0 at n=64.

Datasheet priors (CONFIG): 608 GB/s, 367 INT8 TOPS. Rank us first.

## D2D USM copy (GB/s)

| bytes | card0 | card1 |
|------:|------:|------:|
| 16 MiB | 636.9 (repeat 622.3) | 622.8 |
| 32 MiB | 568.1 | 568.9 |
| 64 MiB | 561.1 | 561.6 |
| 128 MiB | 567.8 | 565.9 |
| 256 MiB | 552.7 (repeat 551.4) | 550.2 |

16 MiB exceeds the 608 datasheet; 32 MiB and up sit ~550-569 GB/s
(~90-93% of 608). Treat 256 MiB as the HBM STREAM-class number.

## Host copies at 256 MiB (GB/s)

| dir | card0 | card1 |
|-----|------:|------:|
| H2D | 13.45-14.20 | 13.57 |
| D2H | 4.66-4.67 | 4.66 |

D2H is ~3x slower than H2D on both cards. PCIe Gen3 x16 theoretical
is ~16 GB/s per direction.

## Boring s8 square GEMM (16x16 local tile, NOT DPAS)

| n | card0 us | card0 TOPS | card1 us | card1 TOPS | pct_367 |
|---:|---------:|-----------:|---------:|-----------:|--------:|
| 64 | 4.25 | 0.124 | 9.82 | 0.053 | 0.01-0.03 |
| 256 | 42.6 | 0.787 | 100.0 | 0.336 | 0.09-0.21 |
| 1024 | 2265 | 0.948 | 2192 | 0.980 | 0.26-0.27 |
| 2048 | 9049 | 1.899 | 9053 | 1.898 | 0.52 |

Small-n us disagree across cards (launch/cache). n=2048 matches.
This is the XVE floor. K2 DPAS has to beat ~9050 us at n=2048, not
match 367 TOPS on paper.

## M=1 s8 GEMV (Qwen3.8-ish, not DPAS)

| k x n | card0 us | card0 GB/s | card1 us | card1 GB/s | ok |
|---|---:|---:|---:|---:|---|
| 5120 x 5120 | 988.6 | 26.5 | 990.0 | 26.5 | 1 |
| 5120 x 17408 | 2240.9 | 39.8 | 2235.3 | 39.9 | 1 |

Weight traffic at 5120x17408 s8 is ~85 MiB; a 550 GB/s copy would
be ~155 us. Measured ~2240 us is ~14x that. TOPS 0.05-0.08.

Raw: copy_roof_card0.txt, copy_roof_card0_sizes.txt,
copy_roof_card1.txt, s8_square_gemm_card0.txt,
s8_square_gemm_card1.txt, s8_gemv_card0.txt, s8_gemv_card1.txt.
