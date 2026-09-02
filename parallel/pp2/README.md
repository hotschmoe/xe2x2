# P3 -- pipeline parallel = 2

Question: can one pipeline stage per B70 hand off activations
correctly, and what is the bubble / copy cost on this fabric?

Open. Host bounce vs device P2P vs mapped peer memory are arms.
Tiny-model PP=2 only after the synthetic handoff is correct.

## Why

PP=2 is the other single axis. It stresses a different pattern than
TP=2: fewer, fatter, stage-boundary transfers and idle bubbles, not
per-layer allreduce. Same PCI tree, same health rules. Do not mix
PP=2 into a kernel microbench.

## Suggested arms

- Synthetic activation tensor stage0 -> stage1 -> (optional) back.
- Copy path A/B/C: host staging, oneCCL send/recv, Level Zero peer.
- Overlap: compute on card0 while copy to card1, and the reverse.
- Stage-memory split: who holds KV, who holds weights.
- Bubble vs payload size.

Both cards, one job. Pause one-card kernel work.

## Record

Bytes, us, bubble fraction, path name, pre/post health, numeric
identity of the handed-off tensor.

## Exit

A correct synthetic handoff with teardown health. FINDINGS for which
copy path survived and whether bubble dominates decode-sized
payloads. No mixed 2x2 until this and `../tp2/` both pass.
