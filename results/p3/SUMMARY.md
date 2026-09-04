# P3 host-staged PP=2 handoff 2026-09-04j

Backend pytorch-xpu on sycl+l0. fabric host_staged_pp2.
p2p=0. gpu-run both cards. One process, both XPUs.
Identity of the tensor. Rank us. No clock spin.
Do not freeze 77 as 2800.

| name | bytes | us | samecard_us | bubble | GBs | ok |
|---|---:|---:|---:|---:|---:|---:|
| T1_H5120 | 10240 | 76.848 | 22.047 | 0.713 | 0.133 | 1 |
| T4_H5120 | 40960 | 76.204 | 13.262 | 0.826 | 0.538 | 1 |
| T64_H5120 | 655360 | 303.123 | 11.865 | 0.961 | 2.162 | 1 |
| T256_H5120 | 2621440 | 1026.181 | 14.012 | 0.986 | 2.555 | 1 |
| T1_H6144 | 12288 | 50.307 | 13.135 | 0.739 | 0.244 | 1 |
| 1MiB | 1048576 | 467.278 | 13.376 | 0.971 | 2.244 | 1 |

ok_all=1. Decode T=1 hidden 5120 is 77 us class,
~71% bubble vs same-card copy 22 us. Bubble
dominates. First xe2x2 PP=2 synthetic.
Device P2P handoff not measured. P4 blocked
on P2 bulk hang.
