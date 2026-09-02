# Host inventory -- b70s4dayz

Snapshot taken 2026-09-02 at xe2x2 repo creation. Re-measure before any
driver, kernel, or runtime change.

## Machine

| Field        | Value                                      |
|--------------|--------------------------------------------|
| Hostname     | `b70s4dayz`                                |
| OS           | Ubuntu (generic kernel)                    |
| Kernel       | `7.1.0-070100-generic`                     |
| Arch         | x86_64                                     |
| CPU          | AMD Ryzen Threadripper 1950X, 16c / 32t    |
| RAM          | 121 GiB                                    |
| Lab path     | `/mnt/vm_8tb/github/xe2x2`                 |
| Serving lab  | `/mnt/vm_8tb/github/b70_ai_things`         |
| Runtime root | `/mnt/vm_8tb/b70`                          |

The host is headless. Neither B70 is display-held.

## GPUs

Two Intel Arc Pro B70 (Battlemage G31, Xe2). KMD is `xe`.

| Slot | DRM     | Render      | PCI           | Device              | EU  | Memory    |
|------|---------|-------------|---------------|---------------------|-----|-----------|
| 0    | card0   | renderD128  | `0000:0b:00.0`| `8086:E223` B70     | 256 | 30.3 GiB  |
| 1    | card1   | renderD129  | `0000:44:00.0`| `8086:E223` B70     | 256 | 30.3 GiB  |

OpenCL platform at snapshot: Intel OpenCL Graphics, NEO `26.22.38646.4`.
Intel device ID: `57891`. Max work group: 1024.

## PCI topology

The cards are not siblings on one switch. Each hangs off its own Intel
PCI bridge (`Device e2ff`):

```
09:00.0 PCI bridge: Intel Device e2ff
  0a:01.0 PCI bridge: Intel Device e2f0
    0b:00.0 VGA: Intel Battlemage G31 [Arc Pro B70]
  0a:02.0 PCI bridge: Intel Device e2f1
    0c:00.0 Audio: Intel Device e2f7

42:00.0 PCI bridge: Intel Device e2ff
  43:01.0 PCI bridge: Intel Device e2f0
    44:00.0 VGA: Intel Battlemage G31 [Arc Pro B70]
  43:02.0 PCI bridge: Intel Device e2f1
    45:00.0 Audio: Intel Device e2f7
```

This is the hardware reason TP=2 and PP=2 are a lab, not a checkbox.
Cross-card copies, P2P claims, and oneCCL topology all have to be
measured against this tree.

## Why TP=2 and PP=2

Two devices, two 30.3 GiB memories. Natural maps:

- TP=2: shard each layer across both cards.
- PP=2: one pipeline stage per card.
- Mixed 2x2: only after both single-axis maps are healthy.

Do not skip per-card and two-rank collective health to get to mixed 2x2.

Four B70s are an evidence-gated expansion (operator will buy 3rd/4th
if xe2x2 shows they pay). This board is x16/x8/x16/x8 in four-card
mode. Do not treat 4x as the live map. See docs/KERNEL_CAMPAIGN.md.

## P0 freeze -- 2026-09-02g

Re-measured on host. Raw dump: `results/p0/identities.txt` plus
`results/p0/SUMMARY.md`. Do not treat the creation snapshot above as
stale; this section extends it.

| Layer | Value |
|-------|-------|
| Kernel | `7.1.0-070100-generic` #202606141628 |
| KMD | `xe` |
| GuC | `xe/bmg_guc_70.bin` 70.58.0 (GT0 and GT1, both cards) |
| HuC | `xe/bmg_huc.bin` 8.2.10 (GT1, both cards) |
| DMC | `i915/bmg_dmc.bin` v2.6 (both cards) |
| Compute Runtime / NEO | `26.22.38646.4` (`intel-opencl-icd`, `intel-ocloc`, `libze-intel-gpu1`) |
| IGC | `intel-igc-core-2` / `intel-igc-opencl-2` 2.36.3 |
| Level Zero loader | `libze1` 1.28.2-2 |
| L0 GPU UMD | `libze_intel_gpu.so.1.15.38646` |
| SYCL adapter | Unified Runtime over Level-Zero **V2** (live) |
| AOT name | `intel_gpu_bmg_g31` |
| Host icpx | Intel oneAPI DPC++ 2026.1.1 (2026.1.1.20260724) at `.../steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi` |
| OpenCL | Intel OpenCL Graphics, both B70s, driver 26.22.38646.4, 256 CU, 2800 MHz, 30.3 GiB |
| GT0 clocks | idle D3hot act/cur 400, min 400, max/rp0 2800 MHz, throttle 0 |
| GT1 clocks | idle act 0 cur 400, max/rp0 1500 MHz (media) |
| Power cap | 230 W (`power1_cap`), crit 460 W, both xe hwmon |
| Display | every DP/HDMI connector `disconnected` on both cards |
| Live serve | none (grafana / prometheus / open-webui only) |

`sycl-ls` with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and
`ZE_AFFINITY_MASK=N`:

- card0 UUID `868023e2-0000-0000-0b00-000000000000` (PCI 0b:00.0)
- card1 UUID `868023e2-0000-0000-4400-000000000000` (PCI 44:00.0)
- Aspects include `ext_intel_esimd` and `ext_intel_matrix`
- OpenCL selector is the labeled control (`Intel OpenCL Graphics`,
  same NEO 26.22.38646.4)

Health images:

- Per-card: `vllm-xpu-env:int8g-v0251` via `bin/xpu-health --card N`
- Two-rank: `b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
  (`:20260826-mtp6`) via `bin/xpu-collective-health --p2p 0`

Both per-card probes HEALTHY. Two-rank compiled XCCL all-reduce
HEALTHY (`COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120
compiled_iterations=10 p2p=0`). P2P stays off.

Host note: no system `g++`. Standalone `icpx` AOT needs a container
g++ or an explicit gcc toolchain. Compile is CPU; run is `gpu-run`.
