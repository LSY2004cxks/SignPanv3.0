import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .keypoint_map import build_name_to_index, resolve_required_body_indices


def _pick_instance(instances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not instances:
        return None
    best = None
    best_score = -1.0
    for ins in instances:
        score = ins.get("bbox_score", 0.0)
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s > best_score:
            best_score = s
            best = ins
    return best or instances[0]


def _frame_to_instance(frame_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(frame_obj, dict):
        return None

    if isinstance(frame_obj.get("selected"), dict) and frame_obj.get("selected"):
        return frame_obj["selected"]

    instances = frame_obj.get("instances")
    if isinstance(instances, dict) and instances:
        # Some mmpose versions emit a single instance as dict instead of list.
        return instances
    if isinstance(instances, list):
        return _pick_instance(instances)

    return None


def _to_np_keypoints_scores(ins: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    kpts = ins.get("keypoints")
    scores = ins.get("keypoint_scores")
    if kpts is None:
        return None, None

    arr = np.asarray(kpts, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, None
    arr = arr[:, :2]

    if scores is None:
        sc = np.ones((arr.shape[0],), dtype=np.float32)
    else:
        sc = np.asarray(scores, dtype=np.float32)
        if sc.ndim == 2 and sc.shape[0] == 1:
            sc = sc[0]
        if sc.ndim != 1:
            sc = np.ones((arr.shape[0],), dtype=np.float32)
        if sc.shape[0] != arr.shape[0]:
            n = min(sc.shape[0], arr.shape[0])
            sc2 = np.ones((arr.shape[0],), dtype=np.float32)
            sc2[:n] = sc[:n]
            sc = sc2
    return arr, sc


def _load_pose_from_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    frames: List[Any]
    if isinstance(obj, list):
        frames = obj
    elif isinstance(obj, dict) and isinstance(obj.get("frames"), list):
        frames = obj["frames"]
    else:
        frames = []

    T = len(frames)
    # Find keypoint dimension from first valid frame.
    kpt_dim = None
    for fr in frames:
        ins = _frame_to_instance(fr)
        if ins is None:
            continue
        kp, _ = _to_np_keypoints_scores(ins)
        if kp is not None:
            kpt_dim = kp.shape[0]
            break
    if kpt_dim is None:
        kpt_dim = 133

    kpts = np.full((T, kpt_dim, 2), np.nan, dtype=np.float32)
    scores = np.zeros((T, kpt_dim), dtype=np.float32)
    for t, fr in enumerate(frames):
        ins = _frame_to_instance(fr)
        if ins is None:
            continue
        kp, sc = _to_np_keypoints_scores(ins)
        if kp is None or sc is None:
            continue
        n = min(kpt_dim, kp.shape[0])
        kpts[t, :n] = kp[:n]
        scores[t, :n] = sc[:n]

    name_to_idx, mapping_source = build_name_to_index(obj, kpt_dim)
    required_idx = resolve_required_body_indices(name_to_idx)
    return {
        "n_frames": T,
        "kpt_dim": kpt_dim,
        "keypoints": kpts,
        "scores": scores,
        "name_to_idx": name_to_idx,
        "mapping_source": mapping_source,
        "required_idx": required_idx,
    }


def _load_pose_from_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    if "keypoints" not in z.files:
        raise ValueError(f"pose npz missing keypoints: {path}")

    kpts = np.asarray(z["keypoints"], dtype=np.float32)
    if kpts.ndim != 3 or kpts.shape[2] < 2:
        raise ValueError(f"invalid keypoints shape in {path}: {kpts.shape}")
    kpts = kpts[:, :, :2]

    if "scores" in z.files:
        scores = np.asarray(z["scores"], dtype=np.float32)
        if scores.ndim != 2:
            scores = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
    else:
        scores = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)

    if scores.shape != (kpts.shape[0], kpts.shape[1]):
        s = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
        r = min(s.shape[0], scores.shape[0]) if scores.ndim > 0 else 0
        c = min(s.shape[1], scores.shape[1]) if scores.ndim > 1 else 0
        if r > 0 and c > 0:
            s[:r, :c] = scores[:r, :c]
        scores = s

    T, K, _ = kpts.shape
    # No explicit keypoint names in npz; rely on fallback maps by K.
    name_to_idx, mapping_source = build_name_to_index({}, K)
    required_idx = resolve_required_body_indices(name_to_idx)

    return {
        "n_frames": int(T),
        "kpt_dim": int(K),
        "keypoints": kpts.astype(np.float32),
        "scores": scores.astype(np.float32),
        "name_to_idx": name_to_idx,
        "mapping_source": f"{mapping_source}_npz",
        "required_idx": required_idx,
    }


def load_pose_json(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return _load_pose_from_npz(path)
    return _load_pose_from_json(path)
