from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_FLOW_KEYS = ["mu", "p95", "p99", "delta_mu", "u_mean", "v_mean"]


def _norm_video_id(v: str) -> str:
    s = str(v).strip()
    m = re.search(r"[Vv](\d+)", s)
    if m:
        return f"V{m.group(1)}"
    nums = re.findall(r"\d+", s)
    return f"V{nums[-1]}" if nums else s


def infer_signer_video_from_pose_path(pose_path: str) -> Tuple[Optional[str], Optional[str], str]:
    p = Path(pose_path)
    stem = p.stem
    signer = p.parent.name if re.fullmatch(r"S\d+", p.parent.name) else None
    video_id = None

    m = re.search(r"([Ss]\d+)_([Vv]\d+)", stem)
    if m:
        signer = m.group(1).upper()
        video_id = m.group(2).upper()
    else:
        mv = re.search(r"([Vv]\d+)", stem)
        if mv:
            video_id = mv.group(1).upper()

    if video_id is None:
        video_id = _norm_video_id(stem)
    return signer, video_id, stem


def resolve_flow_npz_path(flow_root: str, pose_path: str, hand: str) -> Optional[str]:
    signer, video_id, pose_stem = infer_signer_video_from_pose_path(pose_path)
    if hand not in ("L", "R"):
        return None
    hand_name = "left" if hand == "L" else "right"

    root = Path(flow_root)
    cands: List[Path] = []
    if signer:
        cands.append(root / signer / f"{pose_stem}_{hand_name}.npz")
        cands.append(root / signer / f"{signer}_{video_id}_{hand_name}.npz")
        cands.append(root / signer / f"{video_id}_{hand_name}.npz")
    cands.append(root / f"{pose_stem}_{hand_name}.npz")

    for c in cands:
        if c.exists():
            return str(c)
    return None


def _load_from_named_feat(z: Any, use_keys: Sequence[str]) -> Optional[np.ndarray]:
    if "feat" not in z.files or "feat_names" not in z.files:
        return None
    feat = np.asarray(z["feat"], dtype=np.float32)
    names = [str(x) for x in np.asarray(z["feat_names"]).reshape(-1).tolist()]
    name_to_col = {n: i for i, n in enumerate(names)}

    cols: List[np.ndarray] = []
    for k in use_keys:
        if k not in name_to_col:
            return None
        cols.append(feat[:, name_to_col[k] : name_to_col[k] + 1])
    return np.concatenate(cols, axis=1).astype(np.float32)


def _load_from_key_arrays(z: Any, use_keys: Sequence[str]) -> Optional[np.ndarray]:
    cols: List[np.ndarray] = []
    T = None
    for k in use_keys:
        if k not in z.files:
            return None
        arr = np.asarray(z[k], dtype=np.float32).reshape(-1)
        if T is None:
            T = int(arr.shape[0])
        if int(arr.shape[0]) != int(T):
            return None
        cols.append(arr[:, None])
    if not cols:
        return None
    return np.concatenate(cols, axis=1).astype(np.float32)


def load_flow_feature_npz(
    path: str,
    use_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    use_keys = list(use_keys or DEFAULT_FLOW_KEYS)
    z = np.load(path, allow_pickle=True)

    feat = _load_from_named_feat(z, use_keys)
    source = "feat+feat_names"
    if feat is None:
        feat = _load_from_key_arrays(z, use_keys)
        source = "key_arrays"
    if feat is None:
        raise ValueError(f"flow npz missing expected keys {use_keys}: {path}")

    frame_ids = None
    if "frame_ids" in z.files:
        frame_ids = np.asarray(z["frame_ids"]).reshape(-1).astype(np.int32)

    return {
        "features": feat.astype(np.float32),
        "feature_keys": use_keys,
        "source": source,
        "frame_ids": frame_ids,
    }
