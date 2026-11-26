#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_num_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    counts = counts.to(torch.float32)
    w = torch.ones_like(counts)
    nz = counts > 0
    w[nz] = (1.0 - beta) / (1.0 - torch.pow(beta, counts[nz]))
    if nz.any():
        w = w / w[nz].mean().clamp_min(1e-6)
    return w


class ClassBalancedCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        if sample_weight is not None:
            sw = sample_weight.to(per.device).to(per.dtype)
            per = per * sw
        return per.mean()


def build_class_balanced_ce(counts: torch.Tensor, beta: float = 0.9999) -> ClassBalancedCrossEntropy:
    w = effective_num_weights(counts, beta=beta)
    return ClassBalancedCrossEntropy(class_weights=w)
