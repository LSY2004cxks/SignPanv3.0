import numpy as np
import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1,3)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = {}
    B = targets.numel()
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        res[f"acc@{k}"] = correct_k / B if B > 0 else 0.0
    return res


def mrr_score(logits: torch.Tensor, targets: torch.Tensor):
    # mean reciprocal rank
    _, rank = logits.sort(dim=1, descending=True)
    targets = targets.view(-1, 1)
    hits = (rank == targets).nonzero()
    # hits: (N,2) row idx, rank idx
    if hits.numel() == 0:
        return 0.0
    rr = 1.0 / (hits[:, 1].float() + 1.0)
    return rr.mean().item()


def per_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    pred = logits.argmax(dim=1)
    acc = {}
    for c in range(num_classes):
        mask = (targets == c)
        denom = mask.sum().item()
        if denom == 0:
            acc[c] = None
        else:
            acc[c] = (pred[mask] == targets[mask]).float().mean().item()
    return acc


def ece_score(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15):
    # Expected Calibration Error
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets)

    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_bounds[i], bin_bounds[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)
