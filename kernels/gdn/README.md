# K7 -- GDN / hybrid linear-attention kernels

Question: Qwen3.8-27B and the 35B-A3B-class MoEs are GDN hybrids,
not plain transformers. What actually runs in the Gated DeltaNet
path on these two B70s, and is it XMX, XVE, or launch-bound?

Nemotron 3.5 Lightning is Mamba-2 + MoE, not GDN. That is K8
(`kernels/nemotron/`). Do not reuse these leftover tiles as SSU.

Open. We under-weighted this versus GEMM. A "faster INT8 GEMM" that
leaves GDN in eager bf16 can lose the model even if K4 looks great.

## Why

Sibling serving spent real time on GDN qkvz, conv1d, recurrent
state, and quant reuse. Decode-shaped GDN is a skinny recurrent
update plus a short conv, not a fat DPAS tile. Prefill GDN is closer
to a batched linear kernel. MTP changes the row count.

## Suggested arms

- Inventory: which GDN ops fire in a one-layer Qwen3.8 forward
  (names, dtypes, shapes, backend).
- Micro: conv1d, chunked delta update, qkvz projection as W8A16 vs
  W8A8 vs bf16.
- State traffic: bytes of recurrent state vs projection weights at
  M=1 and at MTP M.
- Quant reuse of qkvz (sibling prior: small lever). Measure.

Card0 || card1: split op inventory vs one micro, then swap.

## Record

Op name, shape, us, GB/s, whether DPAS appears in the IGC dump.
Do not quote tok/s.

## Exit

A list of GDN kernels with a roof (launch / BW / XMX). FINDINGS if
GDN is or is not the decode leftover after projections are INT8.
