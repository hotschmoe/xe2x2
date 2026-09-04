# P2 synthetic XCCL P2P-off 2026-09-04i

Backend pytorch-xpu on sycl+l0. fabric xccl. p2p=0.
gpu-run both cards. No serve. Rank us.

Pre-health: card0 HEALTHY, card1 HEALTHY,
COLLECTIVE_HEALTH_OK 4x5120 p2p=0.

## Fire 1 (hang137)

Decode through 64-token ok=1, then hang.

| name | op | us | GBs | ok |
|---|---|---:|---:|---:|
| decode_h 10KiB | all_reduce | 137.119 | 0.075 | 1 |
| decode_h | all_gather | 210.358 | 0.097 | 1 |
| decode_h | sendrecv | 848.166 | 0.012 | 1 |
| health_4h 40KiB | all_reduce | 125.922 | 0.325 | 1 |
| health_4h | all_gather | 172.312 | 0.475 | 1 |
| health_4h | sendrecv | 649.646 | 0.063 | 1 |
| prefill_64h 640KiB | all_reduce | 535.338 | 1.224 | 1 |
| prefill_64h | all_gather | 544.275 | 2.408 | 1 |
| prefill_64h | sendrecv | 948.873 | 0.691 | 1 |

Hang ~18 min after 64h, no 256h RESULT, act=2800.
docker rm unruffled_wing.

## Fire 2 (instrumented retry)

| name | op | us | GBs | ok |
|---|---|---:|---:|---:|
| decode_h | all_reduce | 98.846 | 0.104 | 1 |
| decode_h | all_gather | 127.938 | 0.160 | 1 |
| decode_h | sendrecv | 538.775 | 0.019 | 1 |
| health_4h | all_reduce | 88.563 | 0.462 | 1 |
| health_4h | all_gather | 129.711 | 0.632 | 1 |
| health_4h | sendrecv | 524.313 | 0.078 | 1 |
| prefill_64h | all_reduce | 562.496 | 1.165 | 1 |
| prefill_64h | all_gather | 563.377 | 2.327 | 1 |
| prefill_64h | sendrecv | 890.072 | 0.736 | 1 |
| prefill_256h 2.5MiB | all_reduce | 2081.428 | 1.259 | 1 |

Then hang on all_gather 2.5 MiB. docker rm zealous_blackburn.

## Teardown

Post-health after both kills: per-card HEALTHY,
COLLECTIVE_HEALTH_OK 4x5120 p2p=0.

STOP XCCL all_gather >= 2.5 MiB P2P-off until a
new arm with a timeout. Not a full P2 exit.
P4 stays blocked. Do not enable P2P.
