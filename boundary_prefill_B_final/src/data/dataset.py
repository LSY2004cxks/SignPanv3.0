from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .indexer import ClipRecord
from .labels import build_targets
from ..features.builder import FeatureBuilder


@dataclass
class HandSample:
    clip: ClipRecord
    hand: str  # 'L' or 'R'


class BoundaryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: List[ClipRecord],
        feature_builder: FeatureBuilder,
        sigma: float,
        radius: int,
        use_valid_only: bool,
        expected_dim: Optional[int] = None,
        skip_missing_pose: bool = True,
    ) -> None:
        self.feature_builder = feature_builder
        self.sigma = float(sigma)
        self.radius = int(radius)
        self.use_valid_only = bool(use_valid_only)
        self.expected_dim = expected_dim
        self.skip_missing_pose = skip_missing_pose

        self.samples: List[HandSample] = []
        for rec in records:
            if rec.pose_path is None and skip_missing_pose:
                continue
            self.samples.append(HandSample(rec, "L"))
            self.samples.append(HandSample(rec, "R"))

        self._target_cache: Dict[str, Dict[str, Any]] = {}
        self.feature_dim: Optional[int] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_key(self, rec: ClipRecord, hand: str) -> str:
        return f"{rec.signer_id}|{rec.clip_id}|{hand}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        rec = item.clip
        hand = item.hand

        if rec.pose_path is None:
            raise RuntimeError(f"missing pose for {rec.clip_id}")

        feat_data = self.feature_builder.build_from_pose_path(rec.pose_path)
        feat = feat_data[hand]["features"]
        if self.feature_dim is None:
            self.feature_dim = int(feat.shape[1])
        if self.expected_dim is not None and feat.shape[1] != self.expected_dim:
            raise ValueError(
                f"feature_dim mismatch: got {feat.shape[1]} expected {self.expected_dim} ({rec.clip_id} {hand})"
            )

        T = int(feat.shape[0])
        key = self._cache_key(rec, hand)
        if key not in self._target_cache:
            targets = build_targets(
                rec.events_by_hand.get(hand, []),
                T=T,
                sigma=self.sigma,
                radius=self.radius,
                use_valid_only=self.use_valid_only,
            )
            self._target_cache[key] = targets
        else:
            targets = self._target_cache[key]

        return {
            "x": torch.from_numpy(feat.astype(np.float32)),
            "y_start": torch.from_numpy(targets["y_start"].astype(np.float32)),
            "y_end": torch.from_numpy(targets["y_end"].astype(np.float32)),
            "y_inside": torch.from_numpy(targets["y_inside"].astype(np.float32)),
            "length": T,
            "clip_id": rec.clip_id,
            "video_id": rec.video_id,
            "signer_id": rec.signer_id,
            "hand": hand,
            "gt_start": targets["gt_start"],
            "gt_end": targets["gt_end"],
            "gt_spans": targets["gt_spans"],
            "quality_issues": targets["quality_issues"],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [b["x"] for b in batch]
    y_starts = [b["y_start"] for b in batch]
    y_ends = [b["y_end"] for b in batch]
    y_insides = [b["y_inside"] for b in batch]

    x_pad = pad_sequence(xs, batch_first=True)
    ys_pad = pad_sequence(y_starts, batch_first=True)
    ye_pad = pad_sequence(y_ends, batch_first=True)
    yi_pad = pad_sequence(y_insides, batch_first=True)

    lengths = torch.tensor([int(b["length"]) for b in batch], dtype=torch.long)
    T = int(x_pad.shape[1])
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()

    return {
        "x": x_pad,
        "y_start": ys_pad,
        "y_end": ye_pad,
        "y_inside": yi_pad,
        "mask": mask,
        "lengths": lengths,
        "meta": [
            {
                "clip_id": b["clip_id"],
                "video_id": b["video_id"],
                "signer_id": b["signer_id"],
                "hand": b["hand"],
                "gt_start": b["gt_start"],
                "gt_end": b["gt_end"],
                "gt_spans": b["gt_spans"],
                "quality_issues": b["quality_issues"],
            }
            for b in batch
        ],
    }
