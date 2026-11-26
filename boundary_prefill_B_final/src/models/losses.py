from typing import Dict

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * ((1.0 - pt).clamp_min(1e-6) ** gamma) * bce
    focal = focal * mask
    denom = mask.sum().clamp_min(1.0)
    return focal.sum() / denom


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def boundary_loss(
    logits: torch.Tensor,
    y_start: torch.Tensor,
    y_end: torch.Tensor,
    y_inside: torch.Tensor,
    mask: torch.Tensor,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    inside_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    logit_start = logits[..., 0]
    logit_end = logits[..., 1]
    logit_inside = logits[..., 2]

    loss_start = focal_bce_with_logits(
        logit_start, y_start, mask=mask, alpha=focal_alpha, gamma=focal_gamma
    )
    loss_end = focal_bce_with_logits(
        logit_end, y_end, mask=mask, alpha=focal_alpha, gamma=focal_gamma
    )
    loss_inside = masked_bce_with_logits(logit_inside, y_inside, mask=mask)

    total = loss_start + loss_end + float(inside_weight) * loss_inside
    return {
        "loss": total,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_inside": loss_inside,
    }
