# scripts

This lab does not own the GPU lease.

Use the host lease and health tools from the serving tree:

- `/mnt/vm_8tb/github/b70_ai_things/bin/gpu-run`
- `/mnt/vm_8tb/github/b70_ai_things/bin/xpu-health`
- `/mnt/vm_8tb/github/b70_ai_things/bin/xpu-collective-health`

Put xe2x2-only harnesses here when they exist (kernel runners, TP=2 /
PP=2 launchers). They must call gpu-run. Do not add a second lease.
