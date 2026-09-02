# K5 -- kill the activation-quant launches

Question: how do we get INT8 GEMMs without ~160 standalone per-token
quant kernels per decode step?

Open. Several independent answers may all be valid. Do not start from
a vLLM Linear wrap.

## Why

W8A8 serving lost to FP8 W8A16 locally even after a native s8 GEMM
was exact. The extra work was absmax + s8 write + launch in front of
every linear. Decode is bandwidth- and launch-bound. Prefill can
afford more math.

Ideas, not a sequence:

- Do not quantize activations (W8A16). Same launch count as FP8
  W8A16. Cross-link K4.
- Dedup: q/k/v share one RMSNorm output; gate/up share one. ~3
  quants/layer, not ~7.
- Producer epilogue: RMSNorm writes s8+scale; SiLU x mul writes
  s8+scale. Zero extra launches. Residual stays f16.
- oneDNN post-ops / fusedq: quant+GEMM as one graph node. Work
  remains, Inductor/XPUGraph may care.
- Static scales (SmoothQuant / PrefixQuant class): no absmax. Only
  cheap if fused into the producer.
- Mega-kernel: RMSNorm + quant + GEMM + residual in one ESIMD
  launch. Research, not a server.

## Suggested arms

- Count launches and HBM bytes for: naive per-linear quant, shared
  residual quant, RMSNorm-epilogue quant, fusedq, W8A16 (zero quant).
- Microbench RMSNorm-epilogue vs RMSNorm then separate quant at
  K=5120 and K=17408, M=1 and M=64.
- Numeric: symmetric s8, qmax 127, vs the existing
  `int8_quant_common.hpp` oracle if you reuse that contract.

Card0 || card1: split fusion styles, swap.

## Record

Launch count, us, extra bytes, cosine vs f16 RMSNorm-then-linear.
Say which contract (dynamic per-token vs static) you implemented.

## Exit

A verdict of the form "this fusion removes N launches at cost X us
and error Y" for at least one producer-epilogue attempt, plus the
W8A16 baseline. FINDINGS if an epilogue is bit-closed against the
two-kernel path.
