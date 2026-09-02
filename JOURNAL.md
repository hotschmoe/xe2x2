# JOURNAL

Append new entries at the bottom. Format: CONTEXT, CONFIG, COMMAND,
RESULT, VERDICT. ASCII only.

Active window starts at 2026-09-02a.

### 2026-09-02a - lab created

CONTEXT -> Operator asked for a lab and research arena for Intel GPUs,
  kernels, and TP=2 / PP=2 on this machine. Named xe2x2. Sibling
  serving tree remains b70_ai_things.

CONFIG -> Host b70s4dayz, kernel 7.1.0-070100-generic, KMD xe,
  two Arc Pro B70 (8086:E223) at 0000:0b:00.0 and 0000:44:00.0,
  NEO 26.22.38646.4, 256 EU and 30.3 GiB per card. Default branch
  master. Visibility public.

COMMAND ->
  ```
  gh repo create xe2x2 --public --description "Xe2 dual-B70 kernel and 2x2 parallelism lab (TP=2, PP=2)"
  ```

RESULT -> https://github.com/hotschmoe/xe2x2 created. Local path
  /mnt/vm_8tb/github/xe2x2. Host inventory written to docs/HOST.md.
  PCI tree shows the two cards on separate Intel e2ff bridges.

VERDICT -> Lab open. Next work is RESEARCH_TODO P0: freeze identities
  and run per-card plus two-rank collective health. Do not start mixed
  2x2 before both single-axis maps have a passing run.
