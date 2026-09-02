# scripts

This lab does not own the GPU lease.

Use the host lease and health tools from the serving tree:

- `/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run`
- `/mnt/vm_8tb/github/b70_ai_things/bin/xpu-health`
- `/mnt/vm_8tb/github/b70_ai_things/bin/xpu-collective-health`

Put xe2x2-only harnesses here when they exist (kernel runners, TP=2 /
PP=2 launchers). They must call gpu-run. Do not add a second lease.

xe2x2 helpers:

- `oneapi-env.sh` -- source for host icpx 2026.1.1 (relocated oneAPI
  tree). Live adapter is L0 V2; do not force immediate-lists off.
- `clocks.sh` -- sysfs GT freq / xe hwmon. No GPU lease.
- Host has no `g++`. AOT compile of SYCL kernels uses a CPU docker
  with g++ plus the host oneAPI bind-mount (see kernels/roofline).
