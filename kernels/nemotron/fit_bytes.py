#!/usr/bin/env python3
"""K8 5d: napkin bytes for Lightning expert pool vs 30.3 GiB. CONFIG not RESULT."""
from __future__ import annotations

H = 2688
UP = 1856
SHARE = 3712
N_R = 128
N_S = 1
LAYERS = 23
CARD = 30.3 * (1 << 30)


def gib(n: int) -> float:
    return n / (1 << 30)


def main() -> None:
    routed = LAYERS * N_R * 2 * H * UP  # up+down elements
    shared = LAYERS * N_S * 2 * H * SHARE
    print("CONFIG napkin Lightning expert pool bytes. No checkpoint.")
    print("routed_elems", routed, "shared_elems", shared)
    for name, bpe in (
        ("bf16", 2),
        ("s8_repack", 1),
        ("nvfp4_nibble", 0.5),
        ("gptq_s4", 0.5),
        ("two_term_s4_planes", 1.0),
    ):
        rb = int(routed * bpe)
        sb = int(shared * bpe)
        tot = rb + sb
        print(
            f"{name} routed_GiB={gib(rb):.2f} shared_GiB={gib(sb):.2f} "
            f"experts_GiB={gib(tot):.2f} fit_30.3={tot < CARD}",
            flush=True,
        )
    kv = 6 * 1024  # bytes/token bf16 napkin
    for t in (32768, 120000, 256144):
        print(f"kv_bf16_T={t} GiB={gib(kv * t):.3f}", flush=True)
    print("mamba_state_f32_GiB", gib(23 * 64 * 64 * 128 * 4), flush=True)
    print("VERDICT s8_repack_routed ~27.4 GiB likely no-fit with KV. nvfp4 ~14 GiB fits napkin.")


if __name__ == "__main__":
    main()
