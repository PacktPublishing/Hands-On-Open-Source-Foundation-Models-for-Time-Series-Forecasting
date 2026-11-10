"""Chapter 3, Listing 3.2: one causal attention head with PyTorch's built-in module.

Run from the chapter folder in the env-core environment (or any environment with torch):
    .venv-core/bin/python chapter_listings/causal_attention.py
"""
import torch
from torch import nn

torch.manual_seed(0)
N, d = 6, 16                                     # 6 patches, model width 16
x = torch.randn(1, N, d)                         # (batch, patches, width)

attn = nn.MultiheadAttention(d, num_heads=1, batch_first=True)
M = nn.Transformer.generate_square_subsequent_mask(N)  # -inf above the diagonal
out, w = attn(x, x, x, attn_mask=M)              # w: (batch, N, N) weights
print(out.shape)                                 # torch.Size([1, 6, 16])

# The attention equation with the module's own projections gives the same weights
W_q, W_k, _ = attn.in_proj_weight.chunk(3)
b_q, b_k, _ = attn.in_proj_bias.chunk(3)
Q, K = x[0] @ W_q.T + b_q, x[0] @ W_k.T + b_k
scores = Q @ K.T / d ** 0.5 + M
print(torch.allclose(w[0], torch.softmax(scores, dim=-1), atol=1e-6))
print(w[0, 2].detach().round(decimals=2))        # what patch 3 attends to
