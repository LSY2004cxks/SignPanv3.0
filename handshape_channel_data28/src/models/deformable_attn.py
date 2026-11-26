#!/usr/bin/env python3
"""Temporal deformable-style attention for ROI-Skeleton alignment.

This is a lightweight temporal adaptation of the previous 2D deformable idea:
- ROI sequence is key/value memory
- Skeleton sequence is query
- A learned temporal offset bias modulates attention logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, roi_seq: torch.Tensor, skel_seq: torch.Tensor) -> torch.Tensor:
        """Align ROI sequence to skeleton sequence.

        Args:
            roi_seq:  (B, T, C)
            skel_seq: (B, T, C)
        Returns:
            aligned: (B, T, C)
        """
        q = self.q_proj(skel_seq)
        k = self.k_proj(roi_seq)
        v = self.v_proj(roi_seq)

        # learned temporal offset prior from skeleton query
        # shape: (B,T,1) -> (B,T,T)
        offs = self.offset_mlp(skel_seq)
        pos = torch.linspace(-1.0, 1.0, steps=roi_seq.shape[1], device=roi_seq.device)
        pos = pos.view(1, 1, -1)
        bias = -torch.abs(pos - offs)  # larger when key time is close to predicted offset

        # MultiheadAttention does not accept per-batch attn bias directly in all versions,
        # so we fuse bias by modifying query with a residual gate.
        gate = torch.sigmoid(offs)
        q = q + gate * skel_seq

        attn_out, _ = self.attn(q, k, v, need_weights=False)

        # soft temporal reweighting based on bias (deformable prior)
        prior = F.softmax(bias, dim=-1)
        prior_out = torch.bmm(prior, roi_seq)

        fused = self.out(torch.cat([attn_out, prior_out], dim=-1))
        return fused
