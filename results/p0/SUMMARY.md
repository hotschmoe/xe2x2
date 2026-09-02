# P0 freeze 2026-09-02g

Tracked summary. Raw dumps: identities.txt (clinfo/packages),
card0_gpu_run.log / card1_gpu_run.log / collective_health.log
(gitignored `*.log`).

## Identities

- kernel 7.1.0-070100-generic, KMD xe
- firmware GuC 70.58.0, HuC 8.2.10, DMC 2.6, both cards
- NEO / Compute Runtime 26.22.38646.4
- IGC 2.36.3
- Level Zero loader 1.28.2-2, GPU UMD 1.15.38646
- live SYCL adapter: Unified Runtime over Level-Zero V2
- architecture intel_gpu_bmg_g31
- host icpx 2026.1.1.20260724
- GT0 max 2800 MHz, power cap 230 W, idle D3hot
- no display, no live serve

## Health

- gpu-run --card 0 xpu-health --card 0 --img vllm-xpu-env:int8g-v0251 -> HEALTHY
- gpu-run --card 1 xpu-health --card 1 --img vllm-xpu-env:int8g-v0251 -> HEALTHY
- gpu-run xpu-collective-health --p2p 0 (sglang int8 runtime mtp6,
  sha256:adc915d266ea...) -> COLLECTIVE_HEALTH_OK p2p=0

## sycl-ls

card0: [level_zero:gpu] Level-Zero V2, B70, UUID ...0b00..., driver 1.15.38646+4
card1: [level_zero:gpu] Level-Zero V2, B70, UUID ...4400..., driver 1.15.38646+4
OpenCL control: Intel OpenCL Graphics NEO 26.22.38646.4, same architecture.
