"""Chapter 3, Listing 3.3: the pinball (quantile) loss.

Run from the chapter folder in the env-core environment (or any environment with torch):
    .venv-core/bin/python chapter_listings/pinball_loss.py
"""
import torch

def pinball_loss(y, q_pred, taus):
    """Mean quantile (pinball) loss. y: (B, H); q_pred: (B, H, K); taus: (K,)"""
    u = y.unsqueeze(-1) - q_pred                  # > 0 when forecast too low
    return torch.maximum(taus * u, (taus - 1) * u).mean()

taus = torch.tensor([0.1, 0.5, 0.9])
y = torch.tensor([[10.0, 12.0]])
q = torch.tensor([[[9.0, 10.0, 11.0], [10.0, 11.0, 12.0]]])
print(pinball_loss(y, q, taus))                   # tensor(0.1500)
