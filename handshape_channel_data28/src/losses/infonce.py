#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce(feat_a: torch.Tensor, feat_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE with positive pairs on diagonal and batch negatives."""
    a = F.normalize(feat_a, dim=-1)
    b = F.normalize(feat_b, dim=-1)

    logits = torch.matmul(a, b.t()) / float(tau)
    y = torch.arange(logits.shape[0], device=logits.device)
    la = F.cross_entropy(logits, y)
    lb = F.cross_entropy(logits.t(), y)
    return 0.5 * (la + lb)
