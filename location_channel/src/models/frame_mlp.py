from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import pool


class FrameMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, emb_dim: int = 256,
                 dropout: float = 0.1, pool_mode: str = 'logsumexp'):
        super().__init__()
        self.pool_mode = pool_mode
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> dict:
        # feat: (B,T,F)
        B, T, F = feat.shape
        x = feat.reshape(B * T, F)
        emb = self.mlp(x).reshape(B, T, -1)
        logits_frame = self.cls(emb)
        logits_clip = pool(logits_frame, mask, mode=self.pool_mode)
        return {'logits_frame': logits_frame, 'logits_clip': logits_clip, 'emb_frame': emb}
