#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


TIERS = ["A", "B", "C", "D"]


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def top1_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == targets).float().mean().item())


def hit_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    kk = min(k, logits.shape[-1])
    topk = logits.topk(kk, dim=-1).indices
    hit = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean()
    return float(hit.item())


def macro_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    y_true = targets.detach().cpu().numpy()
    y_pred = logits.argmax(dim=-1).detach().cpu().numpy()
    labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
    if labels.size == 0:
        return 0.0

    f1s = []
    for c in labels.tolist():
        tp = float(np.logical_and(y_true == c, y_pred == c).sum())
        fp = float(np.logical_and(y_true != c, y_pred == c).sum())
        fn = float(np.logical_and(y_true == c, y_pred != c).sum())

        if tp <= 0.0:
            f1s.append(0.0)
            continue

        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        if precision + recall <= 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))

    return float(np.mean(f1s)) if f1s else 0.0


def margin_and_entropy(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    p = torch.softmax(logits, dim=-1)
    top2 = p.topk(min(2, p.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return margin.detach().cpu().numpy(), entropy.detach().cpu().numpy()


def aggregate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    m, e = margin_and_entropy(logits)
    return {
        "top1_acc": top1_acc(logits, targets),
        "hit@3": hit_at_k(logits, targets, k=3),
        "macro_f1": macro_f1_from_logits(logits, targets),
        "mean_margin": float(np.mean(m)) if len(m) else 0.0,
        "mean_entropy": float(np.mean(e)) if len(e) else 0.0,
    }


def tierwise_metrics(logits: torch.Tensor, targets: torch.Tensor, tiers: Iterable[str]) -> Dict[str, Dict[str, float]]:
    tiers = list(tiers)
    out: Dict[str, Dict[str, float]] = {}

    for t in TIERS:
        idx = [i for i, v in enumerate(tiers) if str(v).upper() == t]
        if not idx:
            out[t] = {"n": 0, "top1_acc": 0.0, "hit@3": 0.0, "macro_f1": 0.0}
            continue

        idt = torch.tensor(idx, dtype=torch.long, device=logits.device)
        lg = logits.index_select(0, idt)
        tg = targets.index_select(0, idt)
        out[t] = {
            "n": len(idx),
            "top1_acc": top1_acc(lg, tg),
            "hit@3": hit_at_k(lg, tg, 3),
            "macro_f1": macro_f1_from_logits(lg, tg),
        }
    return out
