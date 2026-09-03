# K7 Qwen3.8-27B GDN inventory (config, 2026-09-03)

Source: `b70_ai_things/models/files/qwen3.8-27b/bf16/config.json`
`text_config`. Backend later micros: `pytorch-xpu` on `sycl+l0`.
Not a serve. ASCII.

## Layer map

- model_type: qwen3_5 / qwen3_5_text
- hidden_size H = 5120
- intermediate_size = 17408 (FFN; already K2)
- num_hidden_layers = 64
- full_attention_interval = 4
- layer_types: 3x linear_attention + 1x full_attention, repeating
- GDN layers = 48, full-attn layers = 16

## GDN dims

- linear_num_key_heads = 16
- linear_key_head_dim = 128  -> key_dim = 2048
- linear_num_value_heads = 48
- linear_value_head_dim = 128 -> value_dim = 6144
- linear_conv_kernel_dim = 4
- head_dim (full attn) = 256, num_attention_heads = 24, kv_heads = 4

key_dim/H = 2048/5120 = 0.40 (not FLA default 0.75).
value_dim/H = 6144/5120 = 1.20.

## State bytes (bf16, CONFIG prior)

Per GDN layer recurrent S: 48 * 128 * 128 * 2 = 1.5 MiB.
48 GDN layers: 72 MiB. Conv state (K-1=3): q+k+v = 60 KiB / layer,
2.8 MiB all layers. Tiny vs 27B weights.

Decode GDN leftover is launch / skinny BW, not 72 MiB of state.

## Ops to time (this fire)

- depthwise conv1d K=4 on C=2048 (q,k) and C=6144 (v), T=1/64/256
- fused-recurrent delta update, 48 heads, 128x128 S, T=1
