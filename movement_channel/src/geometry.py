from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

GLOBAL_LABELS = ("HOLD", "ST", "CU", "CI", "NOM")


def _as_span(span: Any) -> Any:
    return span


def movement_from_wrist(
    points: np.ndarray,
    scores: Optional[np.ndarray] = None,
    *,
    hand: str,
    event_id: str,
    signer_id: str,
    video_id: str,
    span: Any = None,
    min_valid_frames: int = 3,
) -> Dict[str, Any]:
    """从手腕轨迹估计全局运动类型，输出与其他通道兼容的 items。"""
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[0] == 0:
        return _unavailable("empty_wrist_trajectory", "手腕轨迹为空")

    if scores is None:
        valid = np.linalg.norm(xyz[:, :2], axis=1) > 1e-6
    else:
        valid = np.asarray(scores, dtype=np.float32).reshape(-1) > 1e-6
        valid = valid[: xyz.shape[0]]

    if int(valid.sum()) < min_valid_frames:
        return _unavailable("insufficient_wrist_points", "有效手腕点不足，无法估计运动")

    traj = xyz[valid][:, :2]
    deltas = np.diff(traj, axis=0)
    step = np.linalg.norm(deltas, axis=1)
    path = float(step.sum()) if step.size else 0.0
    disp = float(np.linalg.norm(traj[-1] - traj[0]))
    extent = float(np.max(traj.max(axis=0) - traj.min(axis=0)))
    linearity = disp / max(path, 1e-6)
    closure = 1.0 - min(disp / max(path, 1e-6), 1.0)

    scores_map = {
        "HOLD": float(np.exp(-extent / 8.0) * np.exp(-disp / 10.0)),
        "ST": float(
            max(linearity - 0.62, 0.0)
            * _sigmoid((disp - 18.0) / 8.0)
            * _sigmoid((extent - 12.0) / 6.0)
        ),
        "CU": float(
            (1.0 - linearity)
            * _sigmoid((disp - 16.0) / 8.0)
            * (1.0 - 0.7 * closure)
        ),
        "CI": float(
            closure
            * _sigmoid((path - 40.0) / 12.0)
            * (1.0 - linearity)
            * _sigmoid((extent - 12.0) / 6.0)
        ),
        "NOM": 0.04,
    }
    labels, raw = zip(*((label, scores_map[label]) for label in GLOBAL_LABELS))
    logits = np.array(raw, dtype=np.float64)
    logits = np.clip(logits, 1e-8, None)
    probabilities = logits / logits.sum()
    order = np.argsort(-probabilities)[:3]
    top_labels = [labels[index] for index in order]
    top_probs = [float(probabilities[index]) for index in order]
    return {
        "channel": "movement",
        "source": "geometric_fallback",
        "schema": "movement_pred_v1",
        "status": "ok",
        "items": [
            _item(
                event_id=event_id,
                signer_id=signer_id,
                video_id=video_id,
                hand=hand,
                span=span,
                label=top_labels[0],
                top3=top_labels,
                probs=top_probs,
                local_label=None,
                track_point="WRIST",
                extras={
                    "path_length": path,
                    "net_displacement": disp,
                    "extent": extent,
                    "linearity": linearity,
                    "closure": closure,
                    "valid_frames": int(valid.sum()),
                },
            )
        ],
    }


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _unavailable(status: str, error: str) -> Dict[str, Any]:
    return {
        "channel": "movement",
        "source": "unavailable",
        "schema": "movement_pred_v1",
        "status": status,
        "error": error,
        "items": [],
    }


def _item(
    *,
    event_id: str,
    signer_id: str,
    video_id: str,
    hand: str,
    span: Any,
    label: str,
    top3: Sequence[str],
    probs: Sequence[float],
    local_label: Optional[str],
    track_point: str,
    extras: Optional[Dict[str, Any]] = None,
    gt: Optional[str] = None,
    reference_count: Optional[int] = None,
    repeat: int = 1,
) -> Dict[str, Any]:
    hand_norm = str(hand or "").lower()
    suffix = "RIGHT" if hand_norm.startswith("r") else "LEFT"
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0] if probs else 0.0)
    item: Dict[str, Any] = {
        "sample_id": f"{signer_id}_{video_id}_{event_id}_{suffix}",
        "event_id": event_id,
        "signer_id": signer_id,
        "video_id": video_id,
        "hand": hand_norm,
        "span": _as_span(span),
        "y_true": gt,
        "gt": gt,
        "y_pred_top1": label,
        "movement": label,
        "top3": list(top3),
        "top3_labels": list(top3),
        "prob_top3": [float(p) for p in probs],
        "top3_probs": [float(p) for p in probs],
        "conf": float(probs[0]) if probs else 0.0,
        "margin": margin,
        "local_label": local_label,
        "track_point": track_point,
        "repeat": repeat,
        "reference_count": reference_count,
    }
    if extras:
        item["geometry"] = extras
    return item
