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
