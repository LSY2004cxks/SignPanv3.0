from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', labels=list(range(num_classes)), zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average='macro', labels=list(range(num_classes)), zero_division=0))

    # top-3 acc if probabilities provided
    top3 = 0.0
    if y_prob is not None:
        top3_pred = np.argsort(-y_prob, axis=1)[:, :3]
        top3 = float(np.mean([yt in top3_pred[i] for i, yt in enumerate(y_true)]))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        'acc': acc,
        'macro_f1': macro_f1,
        'macro_recall': macro_recall,
        'acc@3': top3,
        'confusion': cm,
    }
