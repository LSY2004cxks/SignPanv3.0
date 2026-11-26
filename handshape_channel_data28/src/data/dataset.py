#!/usr/bin/env python3
"""Dataset utilities for handshape_channel_data28."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_signer_split(split_dir: Path, split: str) -> List[str]:
    p = split_dir / f"{split}_signers.txt"
    if not p.exists():
        raise FileNotFoundError(f"Split signer file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tier_from_meta(meta: Dict[str, Any]) -> str:
    t = str(meta.get("tier", "")).strip().upper()
    if t in {"A", "B", "C", "D"}:
        return t
    side = float(meta.get("bbox_side_px_median", 0.0) or 0.0)
    if side >= 96:
        return "A"
    if side >= 64:
        return "B"
    if side >= 48:
        return "C"
    return "D"


def sample_frame_indices(start: int, end: int, k: int) -> np.ndarray:
    if end < start:
        return np.full((k,), start, dtype=np.int64)
    if k <= 1:
        return np.array([start], dtype=np.int64)
    xs = np.linspace(start, end, num=k)
    idx = np.round(xs).astype(np.int64)
    idx = np.clip(idx, start, end)
    return idx


def fix_len(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.shape[0] == k:
        return arr
    if arr.shape[0] > k:
        idx = np.round(np.linspace(0, arr.shape[0] - 1, num=k)).astype(np.int64)
        return arr[idx]
    # pad with last frame
    out = [arr]
    pad_n = k - arr.shape[0]
    pad = np.repeat(arr[-1:], repeats=pad_n, axis=0)
    out.append(pad)
    return np.concatenate(out, axis=0)


def roi_to_tensor(roi_thwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(roi_thwc).to(torch.float32)
    if x.max().item() > 1.5:
        x = x / 255.0
    # (T,H,W,3) -> (3,T,H,W)
    x = x.permute(3, 0, 1, 2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


class HandshapeEventDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split_dir: str,
        split: str,
        mode: str,
        k: int,
        cache_roi_dir: str,
        cache_skeleton_dir: str,
        cache_meta_dir: str,
        tier_weights: Dict[str, float],
        on_the_fly_fallback: bool = True,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        assert mode in {"roi_only", "skel_only", "roi_skel_fusion", "roi_skel_deform_nce"}

        self.split = split
        self.mode = mode
        self.k = int(k)
        self.tier_weights = {k.upper(): float(v) for k, v in tier_weights.items()}
        self.on_the_fly_fallback = bool(on_the_fly_fallback)

        self.manifest_path = Path(manifest_path).resolve()
        self.split_dir = Path(split_dir).resolve()
        self.cache_roi_dir = Path(cache_roi_dir).resolve()
        self.cache_skeleton_dir = Path(cache_skeleton_dir).resolve()
        self.cache_meta_dir = Path(cache_meta_dir).resolve()

        rows = load_manifest(self.manifest_path)
        keep_signers = set(load_signer_split(self.split_dir, split))
        rows = [r for r in rows if str(r.get("signer_id")) in keep_signers]

        # label vocab across full manifest, deterministic by alpha
        all_labels = sorted({str(r.get("gt_label")) for r in load_manifest(self.manifest_path)})
        self.label_to_idx = {lb: i for i, lb in enumerate(all_labels)}
        self.idx_to_label = {i: lb for lb, i in self.label_to_idx.items()}

        self.samples: List[Dict[str, Any]] = []
        for r in rows:
            label = str(r.get("gt_label"))
            if label not in self.label_to_idx:
                continue
            sid = r["sample_id"]
            meta = self._read_meta(sid)
            tier = tier_from_meta(meta)
            tier_w = float(self.tier_weights.get(tier, 1.0))
            item = {
                "row": r,
                "sample_id": sid,
                "label": label,
                "label_idx": self.label_to_idx[label],
                "tier": tier,
                "tier_weight": tier_w,
                "meta": meta,
            }
            self.samples.append(item)

        self.num_classes = len(self.label_to_idx)

    def _external_npz_path(self, root: Path, sample_id: str) -> Optional[Path]:
        # map canonical sample_id:
        #   S58_V6419_L_L_Event_2
        # to external npz stem:
        #   S58_V6419_left_L_Event_2
        parts = sample_id.split("_")
        if len(parts) < 6:
            return None
        signer, video = parts[0], parts[1]
        side_tag = parts[2].upper()
        event_hand = parts[3]
        event_key = parts[4]
        event_idx = parts[5]
        if side_tag not in {"L", "R"}:
            return None
        side_word = "left" if side_tag == "L" else "right"
        stem = f"{signer}_{video}_{side_word}_{event_hand}_{event_key}_{event_idx}"
        p = root / signer / f"{stem}.npz"
        return p if p.exists() else None

    def _read_meta(self, sample_id: str) -> Dict[str, Any]:
        p = self.cache_meta_dir / f"{sample_id}.json"
        if not p.exists():
            return {}
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _load_cached_roi(self, sample_id: str) -> Optional[np.ndarray]:
        p = self.cache_roi_dir / f"{sample_id}.npy"
        if not p.exists():
            p_npz = self._external_npz_path(self.cache_roi_dir, sample_id)
            if p_npz is None:
                return None
            try:
                pack = np.load(p_npz, allow_pickle=True)
                if "roi" in pack:
                    arr = np.asarray(pack["roi"])
                    if arr.ndim == 4 and arr.shape[-1] == 3:
                        return fix_len(arr.astype(np.uint8), self.k)
            except Exception:
                return None
            return None
        try:
            arr = np.load(p)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                return fix_len(arr, self.k)
        except Exception:
            return None
        return None

    def _load_cached_skeleton(self, sample_id: str) -> Optional[np.ndarray]:
        p = self.cache_skeleton_dir / f"{sample_id}.npy"
        if not p.exists():
            p_npz = self._external_npz_path(self.cache_skeleton_dir, sample_id)
            if p_npz is None:
                return None
            try:
                pack = np.load(p_npz, allow_pickle=True)
                if "xyz" in pack:
                    arr = np.asarray(pack["xyz"])
                elif "keypoints" in pack:
                    arr = np.asarray(pack["keypoints"])
                else:
                    return None
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    arr = arr[..., :3].astype(np.float32)
                    return fix_len(arr, self.k)
            except Exception:
                return None
            return None
        try:
            arr = np.load(p)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3].astype(np.float32)
                return fix_len(arr, self.k)
        except Exception:
            return None
        return None

    def _fallback_decode(self, row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Slow fallback when cache is missing.

        ROI fallback: center crop from sampled frames.
        Skeleton fallback: zeros.
        """
        video_path = str(row["video_path"])
        start = int(row["span_start"])
        end = int(row["span_end"])
        idxs = sample_frame_indices(start, end, self.k)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # full-zero fallback
            roi = np.zeros((self.k, 224, 224, 3), dtype=np.uint8)
            sk = np.zeros((self.k, 21, 3), dtype=np.float32)
            return roi, sk

        rois: List[np.ndarray] = []
        for fi in idxs.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            h, w = frame.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = frame[y0 : y0 + side, x0 : x0 + side]
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            rois.append(crop)
        cap.release()

        roi = np.stack(rois, axis=0).astype(np.uint8)
        sk = np.zeros((self.k, 21, 3), dtype=np.float32)
        return roi, sk

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        row = s["row"]
        sample_id = s["sample_id"]

        roi_np: Optional[np.ndarray] = None
        sk_np: Optional[np.ndarray] = None

        use_roi = self.mode in {"roi_only", "roi_skel_fusion", "roi_skel_deform_nce"}
        use_skel = self.mode in {"skel_only", "roi_skel_fusion", "roi_skel_deform_nce"}

        if use_roi:
            roi_np = self._load_cached_roi(sample_id)
        if use_skel:
            sk_np = self._load_cached_skeleton(sample_id)

        if self.on_the_fly_fallback:
            if (use_roi and roi_np is None) or (use_skel and sk_np is None):
                roi_fb, sk_fb = self._fallback_decode(row)
                if use_roi and roi_np is None:
                    roi_np = roi_fb
                if use_skel and sk_np is None:
                    sk_np = sk_fb

        out: Dict[str, Any] = {
            "sample_id": sample_id,
            "gt": s["label"],
            "label_idx": int(s["label_idx"]),
            "tier": s["tier"],
            "tier_weight": float(s["tier_weight"]),
            "video_id": row.get("video_id"),
            "hand": row.get("hand"),
            "event_id": row.get("event_id"),
            "span_start": int(row.get("span_start", 0)),
            "span_end": int(row.get("span_end", 0)),
        }

        if use_roi:
            if roi_np is None:
                raise RuntimeError(f"ROI missing for sample_id={sample_id}")
            out["roi"] = roi_to_tensor(roi_np)

        if use_skel:
            if sk_np is None:
                raise RuntimeError(f"Skeleton missing for sample_id={sample_id}")
            out["skeleton"] = torch.from_numpy(sk_np.astype(np.float32))

        return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_id": [b["sample_id"] for b in batch],
        "gt": [b["gt"] for b in batch],
        "label_idx": torch.tensor([b["label_idx"] for b in batch], dtype=torch.long),
        "tier": [b["tier"] for b in batch],
        "tier_weight": torch.tensor([b["tier_weight"] for b in batch], dtype=torch.float32),
        "video_id": [b["video_id"] for b in batch],
        "hand": [b["hand"] for b in batch],
        "event_id": [b["event_id"] for b in batch],
        "span_start": [b["span_start"] for b in batch],
        "span_end": [b["span_end"] for b in batch],
    }

    if "roi" in batch[0]:
        out["roi"] = torch.stack([b["roi"] for b in batch], dim=0)

    if "skeleton" in batch[0]:
        out["skeleton"] = torch.stack([b["skeleton"] for b in batch], dim=0)

    return out


def build_class_balanced_sampler(dataset: HandshapeEventDataset) -> WeightedRandomSampler:
    labels = [s["label_idx"] for s in dataset.samples]
    if not labels:
        raise RuntimeError("Cannot build sampler for empty dataset")

    label_counts: Dict[int, int] = {}
    for y in labels:
        label_counts[y] = label_counts.get(y, 0) + 1

    weights = []
    for s in dataset.samples:
        cls_w = 1.0 / float(label_counts[s["label_idx"]])
        tier_w = float(s["tier_weight"])
        weights.append(cls_w * tier_w)

    w = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(weights=w, num_samples=len(weights), replacement=True)


def compute_class_counts(dataset: HandshapeEventDataset) -> torch.Tensor:
    counts = torch.zeros(dataset.num_classes, dtype=torch.long)
    for s in dataset.samples:
        counts[s["label_idx"]] += 1
    return counts
