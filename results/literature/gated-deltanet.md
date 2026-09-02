# Gated DeltaNet -- ops, shapes, complexity (Qwen3.8 linear-attn path)

Status: LANDED.
Primary: arXiv 2412.06464v3 (Yang, Kautz, Hatamizadeh), 2025-03-06.
Implementation reference: fla-org/flash-linear-attention `fla/layers/gated_deltanet.py`.
Follow-on (not Qwen3.8): Gated DeltaNet-2 arXiv 2605.22791 (channel-wise erase/write split).

Not local evidence. No serving rates. Qwen3.8-27B / 35B-A3B-class hybrids use this family of ops; inventory on B70 is K7.

## Why it is on the campaign

Qwen3.8-class models are GDN hybrids, not plain transformers. A GEMM win that leaves GDN in eager bf16 is not a model win. Decode GDN is a skinny recurrent update plus a short conv, not a fat DPAS tile. Prefill is closer to batched linear / chunk matmuls.

## Token-mixer block (paper Fig. 1 + FLA)

Per GDN layer, input `hidden_states: [B, T, H]`.

Projections (FLA, `use_gate=True`, the 6 H^2 layout):

```
q_proj:  H -> key_dim     key_dim   = num_heads * head_dim         (~0.75 H)
k_proj:  H -> key_dim
v_proj:  H -> value_dim   value_dim = num_v_heads * head_dim * expand_v   (~1.5 H)
g_proj:  H -> value_dim   (output gate)
o_proj:  value_dim -> H
a_proj:  H -> num_v_heads (decay / alpha, Mamba2-style)
b_proj:  H -> num_v_heads (beta, write strength)

plus A_log[num_v_heads], dt_bias[num_v_heads]
```

`num_heads * head_dim = 0.75 H` in the default gated layout. expand_v default 2 so head_v_dim = 2 * head_dim. Grouped value attention allowed if num_v_heads > num_heads and divisible.

Short conv1d on q, k, v (paper: "short convolution"; FLA default `conv_size=4`, SiLU, no bias):

```
q_conv1d: ShortConvolution(key_dim,   K=4)
k_conv1d: ShortConvolution(key_dim,   K=4)
v_conv1d: ShortConvolution(value_dim, K=4)
```

Decode keeps a conv state of length K-1 per channel (depthwise). Prefill can fuse qkv into one causal_conv1d when there is no cache/varlen.

Then: rearrange to heads, L2-norm q and k (paper: training stability; FLA can do it inside the kernel), sigmoid(beta), form alpha from a_proj / A_log / dt_bias.

## State and the gated delta rule

State is a matrix per head, not a KV cache of length S:

```
S_t  in  R^{d_v x d_k}
```

FLA default: `state_v_first=True`, so layout is d_v x d_k with d_v = expand_v * d_k.

Paper (Eq. 10), gated delta rule:

```
S_t = S_{t-1} * ( alpha_t * (I - beta_t k_t k_t^T) )  +  beta_t v_t k_t^T
o_t = S_t q_t
```

alpha_t in (0,1): forget / clear. beta_t in (0,1): write strength. (beta in (0,2) optional for negative eigenvalues / state tracking.)

Limits:

```
alpha -> 0 : wipe the state (Mamba2-style clear)
alpha -> 1 : pure delta rule (replace the current key's value)
```

Mamba2 (for contrast):

```
S_t = alpha_t S_{t-1} + v_t k_t^T
```

uniform decay, no targeted replace.

DeltaNet (for contrast):

```
S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
```

targeted replace, no global clear.

Output path: RMSNorm (gated if use_gate) then o_proj. Paper also applies SiLU on the output gate.

## Two kernels, two complexity regimes

FLA modes:

- `chunk` -- training and long prefill. Chunkwise WY / UT algorithm (paper 3.3).
- `fused_recurrent` -- decode / short T. FLA switches to this when `q_len <= 64` and not training.

### Decode / recurrent (T=1, or fused_recurrent)

Per token, per head:

```
v_old = S_{t-1} k_t                         // matvec,  d_v * d_k MACs
S_t   = alpha_t * (S_{t-1} - beta_t v_old k_t^T) + beta_t v_t k_t^T
o_t   = S_t q_t                             // matvec,  d_v * d_k MACs
```

State traffic: `d_v * d_k` elements per head, read and write, every token. Plus conv state `(K-1)*(key_dim + key_dim + value_dim)` and the projection weights.

This is XVE / bandwidth / launch-bound on a GPU unless someone maps the outer products onto XMX. It is not an M x N x K GEMM with large M.

### Prefill / chunk (chunk size C)

Paper 3.3, after absorbing gates into WY:

```
S_{[t+1]} = S_fwd + (U_tilde - W_bwd S^T)^T K_fwd
O_{[t]}   = Q_bwd S^T + (Q K^T odot M) (U_tilde - W_bwd S^T)
```

with decay arrows (multiply by prefix/suffix products of alpha inside the chunk) and

```
U_tilde = [ I + strictLower( diag(beta) (Gamma odot K K^T) ) ]^{-1} diag(beta) V
```

Dominant work per chunk per head: GEMMs of size about C x C, C x d_k, C x d_v, plus the triangular solve / inverse of a C x C factor (WY). C is a kernel block (commonly 64 in this literature). That is the prefill-shaped XMX candidate. MTP increases the row count (campaign K7).

## Cache contents (what K7 should inventory)

From FLA `update_layer_cache`:

```
recurrent_state:  S, shape ~ [num_v_heads, d_v, d_k]   (or swapped if not v_first)
conv_state:       (conv_q, conv_k, conv_v) each [B, dim, conv_size-1] when caching
offset:           sequence position
```

No per-token KV list. Bytes of state vs projection weights at M=1 vs MTP M is a K7 measurement.

## Hybrid

Paper stacks GDN with sliding-window attention (H1) or Mamba2+GDN+SWA (H2). Qwen3.8-class hybrids interleave GDN layers with full attention; exact layer map is a model-card / inventory job, not this paper.

## Gated DeltaNet-2 (follow-on)

arXiv 2605.22791. Splits erase and write:

```
S_t = (I - k_t (b_t odot k_t)^T) D_t S_{t-1}  +  k_t (w_t odot v_t)^T
```

b_t in [0,1]^{d_k} channel-wise erase, w_t in [0,1]^{d_v} channel-wise write, D_t = diag(alpha_t). Reduces to GDN when gates collapse to scalars. Extra ops, same state rank. Only relevant if a later Qwen uses it.

## K7 takeaways

Ops to micro:

1. q/k/v/g/o projections (GEMM, W8 vs bf16)
2. depthwise conv1d K=4 on q, k, v (tiny; launch vs BW)
3. a_proj / b_proj (thin GEMMs, heads-wide)
4. delta update: recurrent (decode) vs chunk (prefill)
5. gated RMSNorm + o_proj

Record shapes, us, GB/s, whether DPAS appears. Do not quote serving rates.
