from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import read_csv, load_yaml_with_base
from src.utils.io import read_json
from src.kpts.geometry import root_scale_normalize, compute_frame_features


@dataclass
class EventItem:
    sample_id: str
    event_id: str
    signer_id: str
    label: int
    label_str: str
    hand_xy: torch.Tensor  # (T,2)
    pose_xy: torch.Tensor  # (T,J,2)
    mask: torch.Tensor     # (T,)
    feat: torch.Tensor     # (T,F)


class EventDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        features_dir: str | Path,
        labels_config: str | Path,
        split_txt: Optional[str | Path] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.labels_csv = Path(labels_csv)
        self.features_dir = Path(features_dir)
        self.labels = read_json(Path(labels_config)).get('codes', [])
        self.code2id = {c: i for i, c in enumerate(self.labels)}

        self.cfg = cfg or {}
        self.pose_cfg = self.cfg.get('pose', {})
        self.feature_cfg = self.cfg.get('features', {})

        rows = read_csv(self.labels_csv)
        if split_txt is not None:
            keep = set([ln.strip() for ln in Path(split_txt).read_text(encoding='utf-8').splitlines() if ln.strip()])
            rows = [r for r in rows if r['sample_id'] in keep]

        # keep only rows with valid labels defined in label vocab
        rows = [r for r in rows if str(r.get('label_loc', '')).strip() in self.code2id]

        # filter by features existence
        rows = [r for r in rows if (self.features_dir / f"{r['sample_id']}.npz").exists()]

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> EventItem:
        r = self.rows[idx]
        sample_id = r['sample_id']
        label_str = r['label_loc']
        y = self.code2id[label_str]

        npz_path = self.features_dir / f"{sample_id}.npz"
        data = np.load(npz_path)
        pose_xy = data['pose_kpts'].astype(np.float32)  # (T,J,2)
        pose_score = data.get('pose_score')
        if pose_score is not None:
            pose_score = pose_score.astype(np.float32)
        hand_xy = data['hand_point'].astype(np.float32)  # (T,2)
        hand_score = data.get('hand_score')
        if hand_score is not None:
            hand_score = hand_score.astype(np.float32)

        # valid mask: both pose+hand are valid
        if 'valid_mask' in data:
            mask = data['valid_mask'].astype(np.float32)
        else:
            mask = np.ones((pose_xy.shape[0],), dtype=np.float32)

        # normalize pose by shoulders
        pose_norm, root, scale = root_scale_normalize(
            pose_xy,
            pose_score,
            joint_names=self.pose_cfg.get('joint_names', []),
            left_shoulder=self.pose_cfg.get('left_shoulder', 'left_shoulder'),
            right_shoulder=self.pose_cfg.get('right_shoulder', 'right_shoulder'),
            left_hip=self.pose_cfg.get('left_hip', 'left_hip'),
            right_hip=self.pose_cfg.get('right_hip', 'right_hip'),
        )

        # normalize hand by same root/scale
        hand_norm = (hand_xy - root) / np.maximum(scale, 1e-6)

        joint_distances = self.feature_cfg.get('joint_distances', [])
        if not bool(self.feature_cfg.get('include_dists', True)):
            joint_distances = []

        feat = compute_frame_features(
            hand_norm,
            pose_norm,
            joint_names=self.pose_cfg.get('joint_names', []),
            joint_distances=joint_distances,
            include_dxy=self.feature_cfg.get('include_dxy', True),
            include_velocity=self.feature_cfg.get('include_velocity', False),
            pose_scores=pose_score,
        )

        return EventItem(
            sample_id=sample_id,
            event_id=r['event_id'],
            signer_id=r['signer_id'],
            label=y,
            label_str=label_str,
            hand_xy=torch.from_numpy(hand_norm),
            pose_xy=torch.from_numpy(pose_norm),
            mask=torch.from_numpy(mask),
            feat=torch.from_numpy(feat),
        )


def collate_fn(batch: List[EventItem]) -> Dict[str, Any]:
    # pad to max T
    max_t = max([b.feat.shape[0] for b in batch])

    def pad(arr: torch.Tensor, pad_value=0.0):
        if arr.shape[0] == max_t:
            return arr
        pad_len = max_t - arr.shape[0]
        pad_shape = (pad_len,) + arr.shape[1:]
        pad_tensor = torch.full(pad_shape, pad_value, dtype=arr.dtype)
        return torch.cat([arr, pad_tensor], dim=0)

    feat = torch.stack([pad(b.feat) for b in batch], dim=0)  # (B,T,F)
    hand_xy = torch.stack([pad(b.hand_xy) for b in batch], dim=0)  # (B,T,2)
    pose_xy = torch.stack([pad(b.pose_xy) for b in batch], dim=0)  # (B,T,J,2)
    mask = torch.stack([pad(b.mask, pad_value=0.0) for b in batch], dim=0)  # (B,T)

    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    return {
        'feat': feat,
        'hand_xy': hand_xy,
        'pose_xy': pose_xy,
        'mask': mask,
        'label': labels,
        'sample_id': [b.sample_id for b in batch],
        'event_id': [b.event_id for b in batch],
        'signer_id': [b.signer_id for b in batch],
        'label_str': [b.label_str for b in batch],
    }
