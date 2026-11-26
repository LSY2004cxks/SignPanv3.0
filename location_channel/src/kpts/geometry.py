from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


def _joint_index(joint_names: List[str], name: str) -> int:
    if name not in joint_names:
        raise KeyError(f"Joint name {name} not found in joint_names")
    return joint_names.index(name)


def _joint_point_and_valid(
    pose_xy_norm: np.ndarray,
    pose_scores: Optional[np.ndarray],
    joint_names: List[str],
    name: str,
    score_thresh: float = 0.1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return joint points (T,2) and validity mask (T,)."""
    if name not in joint_names:
        return None, None
    j = _joint_index(joint_names, name)
    pt = pose_xy_norm[:, j, :]
    if pose_scores is None:
        valid = np.ones((pose_xy_norm.shape[0],), dtype=bool)
    else:
        valid = pose_scores[:, j] > score_thresh
    return pt, valid


def _weighted_mean_points(points: List[np.ndarray], valids: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted mean across candidate reference points.

    points: list of (T,2)
    valids: list of (T,) bool
    """
    T = points[0].shape[0]
    pts = np.stack(points, axis=0)  # (N,T,2)
    v = np.stack(valids, axis=0).astype(np.float32)[..., None]  # (N,T,1)
    denom = np.maximum(v.sum(axis=0), 1e-6)  # (T,1)
    out = (pts * v).sum(axis=0) / denom  # (T,2)
    valid_any = (v.sum(axis=0)[:, 0] > 0.0)
    return out.astype(np.float32), valid_any


def _reference_point(
    name: str,
    pose_xy_norm: np.ndarray,
    pose_scores: Optional[np.ndarray],
    joint_names: List[str],
    score_thresh: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve a reference point by name.

    Supports direct joints and derived aliases (e.g. ``mouth_center``).
    Returns:
      point: (T,2)
      valid: (T,) bool
    """
    T = pose_xy_norm.shape[0]

    # 1) direct joint name
    pt, valid = _joint_point_and_valid(pose_xy_norm, pose_scores, joint_names, name, score_thresh=score_thresh)
    if pt is not None and valid is not None:
        return pt.astype(np.float32), valid

    # 2) derived mouth center aliases
    if name == 'mouth_center':
        aliases = [
            ('mouth_left', 'mouth_right'),
            ('left_mouth', 'right_mouth'),
            ('left_mouth_corner', 'right_mouth_corner'),
            ('mouth_l', 'mouth_r'),
        ]
        for l_name, r_name in aliases:
            lp, lv = _joint_point_and_valid(pose_xy_norm, pose_scores, joint_names, l_name, score_thresh=score_thresh)
            rp, rv = _joint_point_and_valid(pose_xy_norm, pose_scores, joint_names, r_name, score_thresh=score_thresh)
            if lp is None or rp is None:
                continue
            out, valid_any = _weighted_mean_points([lp, rp], [lv, rv])
            return out, valid_any

        # Fallback: if no mouth keypoints are available in the detector output,
        # reuse nose as a stable face reference.
        np_pt, np_valid = _joint_point_and_valid(pose_xy_norm, pose_scores, joint_names, 'nose', score_thresh=score_thresh)
        if np_pt is not None and np_valid is not None:
            return np_pt.astype(np.float32), np_valid

    # 3) unresolved -> mark invalid and return zeros
    return np.zeros((T, 2), dtype=np.float32), np.zeros((T,), dtype=bool)


def root_scale_normalize(
    pose_xy: np.ndarray,
    pose_scores: Optional[np.ndarray],
    joint_names: List[str],
    left_shoulder: str,
    right_shoulder: str,
    left_hip: str,
    right_hip: str,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize pose by root/scale per frame.

    pose_xy: (T,J,2)
    pose_scores: (T,J) or None
    Returns:
      pose_norm: (T,J,2)
      root: (T,2)
      scale: (T,1)
    """
    T, J, _ = pose_xy.shape
    ls_i = _joint_index(joint_names, left_shoulder)
    rs_i = _joint_index(joint_names, right_shoulder)
    lh_i = _joint_index(joint_names, left_hip)
    rh_i = _joint_index(joint_names, right_hip)

    ls = pose_xy[:, ls_i, :]
    rs = pose_xy[:, rs_i, :]
    lh = pose_xy[:, lh_i, :]
    rh = pose_xy[:, rh_i, :]

    def is_valid(jidx):
        if pose_scores is None:
            return np.ones((T,), dtype=bool)
        return pose_scores[:, jidx] > 0.1

    valid_sh = is_valid(ls_i) & is_valid(rs_i)
    valid_hip = is_valid(lh_i) & is_valid(rh_i)

    root = np.zeros((T, 2), dtype=np.float32)
    scale = np.ones((T, 1), dtype=np.float32)

    # shoulders
    root[valid_sh] = (ls[valid_sh] + rs[valid_sh]) / 2.0
    scale[valid_sh, 0] = np.linalg.norm(rs[valid_sh] - ls[valid_sh], axis=-1)

    # hips fallback
    fallback = ~valid_sh & valid_hip
    root[fallback] = (lh[fallback] + rh[fallback]) / 2.0
    scale[fallback, 0] = np.linalg.norm(rh[fallback] - lh[fallback], axis=-1)

    # torso length fallback
    fallback2 = ~valid_sh & ~valid_hip
    if fallback2.any():
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        root[fallback2] = shoulder_mid[fallback2]
        scale[fallback2, 0] = np.linalg.norm(hip_mid[fallback2] - shoulder_mid[fallback2], axis=-1)

    scale = np.maximum(scale, eps)
    pose_norm = (pose_xy - root[:, None, :]) / scale[:, None, :]
    return pose_norm.astype(np.float32), root.astype(np.float32), scale.astype(np.float32)


def compute_frame_features(
    hand_xy_norm: np.ndarray,
    pose_xy_norm: np.ndarray,
    joint_names: List[str],
    joint_distances: List[str],
    include_dxy: bool = True,
    include_velocity: bool = False,
    pose_scores: Optional[np.ndarray] = None,
    score_thresh: float = 0.1,
) -> np.ndarray:
    """Build per-frame feature vectors.

    hand_xy_norm: (T,2)
    pose_xy_norm: (T,J,2)
    Returns: (T,F)
    """
    feats = []
    if include_dxy:
        feats.append(hand_xy_norm)

    # distances to selected joints
    if joint_distances:
        d_list = []
        for name in joint_distances:
            ref, valid = _reference_point(
                name,
                pose_xy_norm,
                pose_scores,
                joint_names,
                score_thresh=score_thresh,
            )
            d = np.linalg.norm(hand_xy_norm - ref, axis=-1, keepdims=True).astype(np.float32)
            # Do not inject low-confidence references into topology distances.
            d[~valid] = 0.0
            d_list.append(d)
        feats.append(np.concatenate(d_list, axis=-1))

    if include_velocity:
        vel = np.zeros_like(hand_xy_norm)
        vel[1:] = hand_xy_norm[1:] - hand_xy_norm[:-1]
        feats.append(vel)

    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_anchors(
    pose_xy_norm: np.ndarray,
    joint_names: List[str],
    anchor_defs: Dict[str, Dict[str, str]],
) -> Dict[str, np.ndarray]:
    """Compute anchor positions per frame.

    Returns dict code -> (T,2)
    """
    T, J, _ = pose_xy_norm.shape

    def get_joint(name: str) -> np.ndarray:
        idx = _joint_index(joint_names, name)
        return pose_xy_norm[:, idx, :]

    # helpers for torso grid
    ls = get_joint('left_shoulder') if 'left_shoulder' in joint_names else None
    rs = get_joint('right_shoulder') if 'right_shoulder' in joint_names else None
    lh = get_joint('left_hip') if 'left_hip' in joint_names else None
    rh = get_joint('right_hip') if 'right_hip' in joint_names else None

    if ls is not None and rs is not None:
        shoulder_mid = (ls + rs) / 2.0
    else:
        shoulder_mid = np.zeros((T, 2), dtype=np.float32)

    if lh is not None and rh is not None:
        hip_mid = (lh + rh) / 2.0
    else:
        hip_mid = shoulder_mid.copy()

    torso_vec = hip_mid - shoulder_mid
    TTC = shoulder_mid + 0.25 * torso_vec
    TBC = shoulder_mid + 0.75 * torso_vec

    anchors: Dict[str, np.ndarray] = {}
    for code, spec in anchor_defs.items():
        t = spec.get('type')
        if t == 'joint':
            anchors[code] = get_joint(spec['name'])
        elif t == 'mid':
            a = get_joint(spec['a'])
            b = get_joint(spec['b'])
            anchors[code] = (a + b) / 2.0
        elif t == 'interp':
            a = get_joint(spec['a'])
            b = get_joint(spec['b'])
            alpha = float(spec.get('alpha', 0.5))
            anchors[code] = a * (1.0 - alpha) + b * alpha
        elif t == 'segment_mid':
            a = get_joint(spec['a'])
            b = get_joint(spec['b'])
            anchors[code] = (a + b) / 2.0
        elif t == 'torso_grid':
            row = spec.get('row')
            col = spec.get('col')
            base = TTC if row == 'top' else TBC
            if col == 'center':
                anchors[code] = base
            elif col == 'left':
                if ls is not None:
                    anchors[code] = base + 0.35 * (ls - base)
                else:
                    anchors[code] = base
            elif col == 'right':
                if rs is not None:
                    anchors[code] = base + 0.35 * (rs - base)
                else:
                    anchors[code] = base
            else:
                anchors[code] = base
        else:
            raise ValueError(f"Unknown anchor type: {t}")

    return anchors
