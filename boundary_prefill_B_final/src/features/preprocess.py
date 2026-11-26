from typing import Any, Dict, Tuple

import numpy as np


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    t = np.arange(x.shape[0], dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.zeros_like(x)
    if mask.sum() == 1:
        return np.full_like(x, float(x[mask][0]))
    out = np.interp(t, t[mask], x[mask]).astype(np.float32)
    return out


def _interp_nan_nd(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    T = out.shape[0]
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        flat[:, j] = _interp_nan_1d(flat[:, j])
    return flat.reshape(arr.shape)


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    p = k // 2
    xp = np.pad(x, (p, p), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.median(xp[i : i + k])
    return out


def _median_filter_nd(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    out = arr.copy()
    T = out.shape[0]
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        flat[:, j] = _median_filter_1d(flat[:, j], k)
    return flat.reshape(arr.shape)


def _safe_norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=axis))


def _frame_bbox_center_diag(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # points: (T, K, 2)
    T = points.shape[0]
    centers = np.zeros((T, 2), dtype=np.float32)
    diags = np.ones((T,), dtype=np.float32)
    for t in range(T):
        xy = points[t]
        m = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if np.sum(m) == 0:
            continue
        x = xy[m, 0]
        y = xy[m, 1]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        centers[t, 0] = 0.5 * (xmin + xmax)
        centers[t, 1] = 0.5 * (ymin + ymax)
        diag = float(np.hypot(xmax - xmin, ymax - ymin))
        diags[t] = max(diag, 1.0)
    return centers, diags


def _resolve_body_indices(required_idx: Dict[str, Any], kpt_dim: int) -> Tuple[Dict[str, int], str]:
    idx = dict(required_idx)
    used = "name_map"
    # fallback for COCO body order when names are unavailable.
    fallback = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
    }
    missing = [k for k, v in idx.items() if v is None]
    if missing and kpt_dim >= 13:
        for k in missing:
            idx[k] = fallback.get(k)
        used = "fallback_body23_order"

    clean_idx: Dict[str, int] = {}
    for k, v in idx.items():
        if v is None:
            continue
        iv = int(v)
        if 0 <= iv < kpt_dim:
            clean_idx[k] = iv
    return clean_idx, used


def build_hand_features(
    pose_data: Dict[str, Any],
    score_thr: float = 0.3,
    median_k: int = 5,
) -> Dict[str, Any]:
    kpts = pose_data["keypoints"].astype(np.float32)
    scores = pose_data["scores"].astype(np.float32)
    T, K, _ = kpts.shape

    idx_map, idx_source = _resolve_body_indices(pose_data.get("required_idx", {}), K)

    needed = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]
    for name in needed:
        if name not in idx_map:
            raise ValueError(f"missing keypoint index for {name}")

    gated = kpts.copy()
    gated[scores < float(score_thr)] = np.nan

    interp = _interp_nan_nd(gated)
    smooth = _median_filter_nd(interp, int(median_k))

    ls = smooth[:, idx_map["left_shoulder"], :]
    rs = smooth[:, idx_map["right_shoulder"], :]
    lh = smooth[:, idx_map["left_hip"], :]
    rh = smooth[:, idx_map["right_hip"], :]

    centers_bbox, diag_bbox = _frame_bbox_center_diag(smooth)

    mid_shoulder = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)

    root = mid_shoulder.copy()
    shoulder_valid = np.isfinite(mid_shoulder[:, 0]) & np.isfinite(mid_shoulder[:, 1])
    hip_valid = np.isfinite(mid_hip[:, 0]) & np.isfinite(mid_hip[:, 1])

    root[~shoulder_valid & hip_valid] = mid_hip[~shoulder_valid & hip_valid]
    root[~shoulder_valid & ~hip_valid] = centers_bbox[~shoulder_valid & ~hip_valid]

    shoulder_w = _safe_norm(ls - rs)
    scale = shoulder_w.copy()
    bad = (~np.isfinite(scale)) | (scale < 1e-6)
    scale[bad] = diag_bbox[bad]
    scale[(~np.isfinite(scale)) | (scale < 1e-6)] = 1.0

    norm = (smooth - root[:, None, :]) / scale[:, None, None]

    def make_hand(hand: str) -> Dict[str, Any]:
        if hand == "L":
            s_idx = idx_map["left_shoulder"]
            e_idx = idx_map["left_elbow"]
            w_idx = idx_map["left_wrist"]
        else:
            s_idx = idx_map["right_shoulder"]
            e_idx = idx_map["right_elbow"]
            w_idx = idx_map["right_wrist"]

        pts = np.stack([norm[:, s_idx, :], norm[:, e_idx, :], norm[:, w_idx, :]], axis=1)  # (T,3,2)

        d1 = np.zeros_like(pts)
        d1[1:] = pts[1:] - pts[:-1]
        d2 = np.zeros_like(pts)
        d2[1:] = d1[1:] - d1[:-1]

        speed_joints = _safe_norm(d1, axis=2)
        wrist_speed = speed_joints[:, 2]
        upper_motion = np.mean(speed_joints, axis=1)

        speed_diff = np.zeros_like(wrist_speed)
        speed_diff[1:] = wrist_speed[1:] - wrist_speed[:-1]

        wrist_score = scores[:, w_idx]
        low_flag = (wrist_score < float(score_thr)).astype(np.float32)

        feat = np.concatenate(
            [
                pts.reshape(T, -1),
                d1.reshape(T, -1),
                d2.reshape(T, -1),
                wrist_speed[:, None],
                upper_motion[:, None],
                speed_diff[:, None],
                wrist_score[:, None],
                low_flag[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        return {
            "features": feat,
            "wrist_speed": wrist_speed.astype(np.float32),
            "upper_motion": upper_motion.astype(np.float32),
            "wrist_score": wrist_score.astype(np.float32),
            "low_score_flag": low_flag.astype(np.float32),
        }

    left = make_hand("L")
    right = make_hand("R")

    return {
        "L": left,
        "R": right,
        "feature_dim": int(left["features"].shape[1]),
        "idx_source": idx_source,
        "mapping_source": pose_data.get("mapping_source", "unknown"),
    }
