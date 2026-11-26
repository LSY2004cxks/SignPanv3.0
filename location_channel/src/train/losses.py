from __future__ import annotations

import torch
import numpy as np


def compute_class_weights(labels: list[int], num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv / inv.mean()
    if smoothing > 0:
        inv = (1 - smoothing) * inv + smoothing * np.ones_like(inv)
    return torch.tensor(inv, dtype=torch.float32)
