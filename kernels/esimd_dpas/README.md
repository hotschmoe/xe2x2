# K2 -- ESIMD DPAS microkernels

Question: can we light s8, s4, and s2 DPAS on these two B70s from
hand SYCL ESIMD, and what MAC rate / occupancy do we get?

Open. Tile sizes, prefetch, and load path are the experiment, not
the spec. This is also how INT2 silicon gets exercised even if no
production format wants INT2 yet.

## Why

Portable APIs floor at INT8. Sibling lab: `dpas<s4,s4>` compiled,
disassembled as `dpas.s4.s4`, ~2x s8 MAC rate. INT2 matrix modes are
claimed in silicon; no xe2x2 measurement. Flashnext s8 tile notes
(RC=8, K=32, N=16) are a starting guess, not a requirement.
Host-prepacked VNNI was a BMG-G31 landmine; transformed 2D loads
were bit-exact. Re-test, do not assume.

## Suggested arms

- `dpas<s8,s8>`, `dpas<s4,s4>`, `dpas<s2,s2>`, unsigned variants.
- Mixed, literature-first: `s2 x s8` / `u2 x u8` (arXiv 2508.06753
  Xe2 native int2xint8 -> int32, VNNI16). Do not skip this for s2xs2.
- Other mixes: s8x s4, s4x s2, acc s32 vs wider if the ISA allows.
- Load path A/B: transformed LSC 2D vs flat / prepacked VNNI.
- Blocking sweeps: repeat count, K-depth, N-tiles, prefetch on/off.
- Occupancy: one Xe-core vs full 32, register pressure.

Card0 || card1: split dtypes first (s8 vs s4, then s2 vs mixed),
swap so every dtype has both-card numbers.

## Record

Compile yes/no, IGC encoding name, numeric mismatch vs host oracle
(N cases), MAC/s, percent of 367 INT8 TOPS (and the analogous s4/s2
napkin if a rate multiplier appears). Keep disasm.

## Exit

At least one passing numeric microbench per dtype that compiled, or
a documented compile/runtime refusal. FINDINGS for "s2 exists / does
not exist on this IGC" once both cards agree.

## First binaries (2026-09-02, compile only)

Standalone `icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31`. CPU docker
AOT (`compile_in_docker.sh`), no `--device`, no GPU run this pass.
icpx 2026.1.1. Backend named `sycl+l0` in each source header.

All four TUs compiled. No API refusal.

| source | dpas | K/dpas | binary | AOT |
|--------|------|--------|--------|-----|
| `dpas_s8.cpp` | s8 x s8 -> s32 | 32 | `bin/dpas_s8` | OK |
| `dpas_s4.cpp` | s4 x s4 -> s32 | 64 | `bin/dpas_s4` | OK |
| `dpas_s2xs8.cpp` | s2 x s8 -> s32 | 32 | `bin/dpas_s2xs8` | OK |
| `dpas_s2.cpp` | s2 x s2 -> s32 | 64 | `bin/dpas_s2` | OK |

B feed is `lsc_load_2d` `Transformed=true` (s8 from row-major int8;
s4/s2 from K-packed uint8). Host s32 oracle is in the binary.

Measured 2026-09-02k on both cards (gpu-run, 1024^3): all four
numeric-closed. us at matched ~583 MHz: s8 374, s4 250, s2xs8 278,
s2xs2 223 (card1; card0 faster when clocks were higher). Details:
`results/k2/SUMMARY.md`. Do not quote TOPS% of 367 until clocks hold.

ocloc/IGA (2026-09-02r): `dpas.8x8 (16|M0)` acc `:d`; s8 `:b/:b`,
s4 `:s4/:s4`, s2 `:s2/:s2`, s2xs8 `:s2/:b`. See `results/k2/igc_isa.md`.
