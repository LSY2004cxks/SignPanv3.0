from typing import Any, Dict, List, Optional

import numpy as np

from .flow_reader import load_flow_feature_npz, resolve_flow_npz_path
from .pose_reader import load_pose_json
from .preprocess import build_hand_features


class FeatureBuilder:
    def __init__(
        self,
        score_thr: float = 0.3,
        median_k: int = 5,
        feature_mode: str = "auto",
        flow_enabled: bool = False,
        flow_root: Optional[str] = None,
        flow_feature_keys: Optional[List[str]] = None,
        flow_allow_missing: bool = True,
        flow_feature_dim: int = 6,
    ):
        self.score_thr = float(score_thr)
        self.median_k = int(median_k)
        mode = str(feature_mode or "auto").strip().lower()
        aliases = {
            "pose": "pose_only",
            "poseonly": "pose_only",
            "flow": "flow_only",
            "flowonly": "flow_only",
            "pose+flow": "pose_flow",
            "pose_flow": "pose_flow",
            "poseflow": "pose_flow",
            "fusion": "pose_flow",
            "both": "pose_flow",
        }
        mode = aliases.get(mode, mode)
        if mode == "auto":
            mode = "pose_flow" if bool(flow_enabled) else "pose_only"
        if mode not in {"pose_only", "flow_only", "pose_flow"}:
            raise ValueError(f"unsupported feature_mode={feature_mode}, expected pose_only|flow_only|pose_flow")

        self.feature_mode = mode
        self.flow_enabled = bool(flow_enabled) or self.feature_mode in {"flow_only", "pose_flow"}
        self.flow_root = str(flow_root) if flow_root else None
        self.flow_feature_keys = list(flow_feature_keys or ["mu", "p95", "p99", "delta_mu", "u_mean", "v_mean"])
        self.flow_allow_missing = bool(flow_allow_missing)
        self.flow_feature_dim = int(flow_feature_dim)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.mapping_stats: Dict[str, int] = {}
        self.idx_stats: Dict[str, int] = {}
        self.flow_stats: Dict[str, int] = {}

    def _count(self, d: Dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    def _load_flow(self, pose_path: str, hand: str, length: int) -> np.ndarray:
        if not self.flow_enabled:
            return np.zeros((length, 0), dtype=np.float32)
        if not self.flow_root:
            raise ValueError("flow_enabled=True but flow_root is not set")

        flow_path = resolve_flow_npz_path(self.flow_root, pose_path, hand)
        if flow_path is None:
            self._count(self.flow_stats, "missing_file")
            if not self.flow_allow_missing:
                raise FileNotFoundError(f"missing flow npz for {pose_path} hand={hand} under {self.flow_root}")
            return np.zeros((length, self.flow_feature_dim), dtype=np.float32)

        try:
            flow_data = load_flow_feature_npz(flow_path, use_keys=self.flow_feature_keys)
            feat = flow_data["features"].astype(np.float32)
            self._count(self.flow_stats, f"loaded:{flow_data.get('source', 'unknown')}")
        except Exception as e:
            self._count(self.flow_stats, "load_error")
            if not self.flow_allow_missing:
                raise
            # keep stable dim when loading fails.
            self._count(self.flow_stats, f"error:{type(e).__name__}")
            return np.zeros((length, self.flow_feature_dim), dtype=np.float32)

        if feat.shape[1] != len(self.flow_feature_keys):
            raise ValueError(
                f"flow feature_dim mismatch at {flow_path}: got {feat.shape[1]} expected {len(self.flow_feature_keys)}"
            )
        return pad_or_trim(feat, length)

    def build_from_pose_path(self, pose_path: str) -> Dict[str, Any]:
        if pose_path in self.cache:
            return self.cache[pose_path]

        pose_data = load_pose_json(pose_path)
        feat_data = build_hand_features(
            pose_data,
            score_thr=self.score_thr,
            median_k=self.median_k,
        )

        self._count(self.mapping_stats, str(feat_data.get("mapping_source", "unknown")))
        self._count(self.idx_stats, str(feat_data.get("idx_source", "unknown")))

        for hand in ("L", "R"):
            pose_feat = feat_data[hand]["features"].astype(np.float32)
            flow_feat = self._load_flow(pose_path, hand, length=int(pose_feat.shape[0]))
            feat_data[hand]["pose_features"] = pose_feat
            feat_data[hand]["flow_features"] = flow_feat

            if self.feature_mode == "pose_only":
                out_feat = pose_feat
            elif self.feature_mode == "flow_only":
                out_feat = flow_feat
            else:
                out_feat = np.concatenate([pose_feat, flow_feat], axis=1).astype(np.float32)

            feat_data[hand]["features"] = out_feat.astype(np.float32)

        feat_data["flow_feature_dim"] = int(feat_data["L"]["flow_features"].shape[1])
        feat_data["pose_feature_dim"] = int(feat_data["L"]["pose_features"].shape[1])
        feat_data["feature_mode"] = self.feature_mode
        feat_data["feature_dim"] = int(feat_data["L"]["features"].shape[1])

        self.cache[pose_path] = feat_data
        return feat_data

    def get_report(self) -> Dict[str, Any]:
        return {
            "feature_mode": self.feature_mode,
            "mapping_source": self.mapping_stats,
            "index_source": self.idx_stats,
            "flow": {
                "enabled": self.flow_enabled,
                "root": self.flow_root,
                "feature_keys": self.flow_feature_keys,
                "allow_missing": self.flow_allow_missing,
                "stats": self.flow_stats,
            },
            "cache_size": len(self.cache),
        }


def pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    if arr.shape[0] == 0:
        return np.zeros((length, arr.shape[1]), dtype=arr.dtype)
    pad = np.repeat(arr[-1:, :], repeats=length - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)
