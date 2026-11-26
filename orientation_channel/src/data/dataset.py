import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .sampling import sample_indices

HAND_WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17


def normalize_landmarks(lm: np.ndarray):
    """lm: (T,21,3 or 4). returns normalized xyz and mask."""
    if lm.ndim != 3 or lm.shape[1] != 21:
        raise ValueError(f"Invalid landmarks shape: {lm.shape}")
    xyz = lm[..., :3].astype(np.float32)
    conf = None
    if lm.shape[-1] > 3:
        conf = lm[..., 3]
    # mask for missing/invalid frames
    geom_energy = np.linalg.norm(xyz, axis=-1).mean(axis=1)
    if conf is None:
        mask = geom_energy > 1e-6
    else:
        mask = (np.nan_to_num(conf.mean(axis=1)) > 0.1) & (geom_energy > 1e-6)
    # wrist-centered
    root = xyz[:, HAND_WRIST:HAND_WRIST+1, :]
    xyz = xyz - root
    # scale norm by palm width (index_mcp - pinky_mcp)
    palm = xyz[:, INDEX_MCP, :] - xyz[:, PINKY_MCP, :]
    s = np.linalg.norm(palm, axis=-1)  # (T,)
    s_mean = float(np.maximum(s.mean(), 1e-6))
    s = np.where(s < 1e-6, s_mean, s)
    xyz = xyz / s[:, None, None]
    return xyz, mask


def compute_palm_normal(xyz: np.ndarray, hand: str):
    """xyz: (T,21,3) normalized. returns (T,3)"""
    v1 = xyz[:, INDEX_MCP, :] - xyz[:, HAND_WRIST, :]
    v2 = xyz[:, PINKY_MCP, :] - xyz[:, HAND_WRIST, :]
    normal = np.cross(v1, v2)
    # handedness consistency
    if hand.lower() == 'left':
        normal = -normal
    # normalize
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-6)
    return normal.astype(np.float32)


class OrientationDataset(Dataset):
    def __init__(
        self,
        events_jsonl: str,
        labels: List[str],
        precomputed_dir: Optional[str],
        external_roi_dir: Optional[str] = None,
        external_skeleton_dir: Optional[str] = None,
        roi_size: int = 224,
        T: int = 8,
        sampling: str = 'biased_start',
        sampling_bias: float = 0.7,
        use_precomputed: bool = True,
        use_roi: bool = True,
        use_geom: bool = True,
        use_normal: bool = True,
        roi_input: str = 'feat',  # feat | img | auto
    ):
        # self.events = [json.loads(l) for l in Path(events_jsonl).read_text().splitlines() if l.strip()]
        self.events = [
            json.loads(l)
            for l in Path(events_jsonl).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        self.label2id = {k: i for i, k in enumerate(labels)}
        self.labels = labels
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.external_roi_dir = Path(external_roi_dir) if external_roi_dir else None
        self.external_skeleton_dir = Path(external_skeleton_dir) if external_skeleton_dir else None
        self.roi_size = int(roi_size)
        self.T = T
        self.sampling = sampling
        self.sampling_bias = sampling_bias
        self.use_precomputed = use_precomputed
        self.use_roi = use_roi
        self.use_geom = use_geom
        self.use_normal = use_normal
        self.roi_input = roi_input

    def __len__(self):
        return len(self.events)

    def _event_stem(self, e: Dict[str, Any]) -> str:
        signer = e.get('signer_id', 'UNK')
        return f"{signer}_{e['video_id']}_{e['event_id']}_{e['hand']}"

    def _event_stem_alt(self, e: Dict[str, Any]) -> str:
        # External caches in this project use: signer_video_hand_event
        signer = e.get('signer_id', 'UNK')
        return f"{signer}_{e['video_id']}_{e['hand']}_{e['event_id']}"

    def _load_precomputed(self, e: Dict[str, Any]):
        if self.precomputed_dir is None:
            return None
        stem = self._event_stem(e)
        npz = self.precomputed_dir / 'events' / f"{stem}.npz"
        if npz.exists():
            return np.load(npz, allow_pickle=True)

        # fallback: separate caches
        data = {}
        roi_feat = self.precomputed_dir / 'roi_feat' / f"{stem}.npy"
        lms = self.precomputed_dir / 'landmarks' / f"{stem}.npy"
        if roi_feat.exists():
            data['roi_feat'] = np.load(roi_feat)
        if lms.exists():
            data['landmarks'] = np.load(lms)
        if data:
            return data
        return None

    def _find_external_npz(self, root: Path, signer: str, stems: List[str]) -> Optional[Path]:
        for stem in stems:
            p1 = root / signer / f"{stem}.npz"
            if p1.exists():
                return p1
            p2 = root / f"{stem}.npz"
            if p2.exists():
                return p2
        return None

    def _load_external(self, e: Dict[str, Any]):
        if self.external_roi_dir is None and self.external_skeleton_dir is None:
            return None

        signer = str(e.get('signer_id', 'UNK'))
        stems = [self._event_stem(e), self._event_stem_alt(e)]
        data: Dict[str, Any] = {}

        quality = None
        if self.external_roi_dir is not None:
            roi_npz = self._find_external_npz(self.external_roi_dir, signer, stems)
            if roi_npz is not None:
                with np.load(roi_npz, allow_pickle=True) as d:
                    if 'roi' in d:
                        data['roi_img'] = d['roi']
                    if 'valid' in d:
                        valid = d['valid'].astype(np.uint8)
                        quality = np.where(valid > 0, 'OK', 'NO_HAND')

        if self.external_skeleton_dir is not None:
            sk_npz = self._find_external_npz(self.external_skeleton_dir, signer, stems)
            if sk_npz is not None:
                with np.load(sk_npz, allow_pickle=True) as d:
                    if 'xyz' in d:
                        xyz = d['xyz'].astype(np.float32)
                        if 'detected' in d:
                            det = d['detected'].astype(np.float32)
                            conf = np.repeat(det[:, None, None], xyz.shape[1], axis=1)
                            landmarks = np.concatenate([xyz, conf], axis=-1)
                            if quality is None:
                                quality = np.where(det > 0, 'OK', 'NO_HAND')
                            else:
                                quality = np.where(det > 0, quality, 'NO_HAND')
                        else:
                            landmarks = xyz
                        data['landmarks'] = landmarks

        if quality is not None:
            data['quality'] = np.asarray(quality, dtype=object)
        return data if data else None

    def _unwrap(self, arr):
        if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
            return arr.item()
        return arr

    def __getitem__(self, idx):
        e = self.events[idx]
        label = e['label_orientation']
        y = self.label2id[label]

        start = int(e['span']['start_frame'])
        end = int(e['span']['end_frame'])
        idxs = sample_indices(start, end, self.T, self.sampling, self.sampling_bias)

        roi_feats = None
        roi_imgs = None
        landmarks = None
        mask = None
        palm_normal = None
        quality_flags = None
        quality_seq = None

        if self.use_precomputed:
            data = self._load_precomputed(e)
            if data is None:
                data = self._load_external(e)
            if data is None:
                raise FileNotFoundError('Missing precomputed for event')
            if self.use_roi and self.roi_input in ('feat', 'auto') and 'roi_feat' in data:
                feat = self._unwrap(data['roi_feat'])
                if isinstance(feat, np.ndarray):
                    roi_feats = feat  # (T,Dv)
            if self.use_roi and roi_feats is None and self.roi_input in ('img', 'auto') and 'roi_img' in data:
                imgs = self._unwrap(data['roi_img'])
                if isinstance(imgs, np.ndarray):
                    roi_imgs = imgs  # (T,H,W,3)
            # palm_normal depends on landmarks even when geom branch is disabled
            if (self.use_geom or self.use_normal) and 'landmarks' in data:
                landmarks = data['landmarks']  # (T,21,3/4)
                xyz, mask = normalize_landmarks(landmarks)
                landmarks = xyz
                if self.use_normal:
                    palm_normal = compute_palm_normal(xyz, e['hand'])
            if 'quality' in data:
                quality = self._unwrap(data['quality'])
                if isinstance(quality, np.ndarray):
                    quality_seq = quality
                    quality_flags = sorted(set([str(q) for q in quality.tolist() if q is not None]))
        else:
            raise RuntimeError('Non-precomputed path not implemented in dataset. Use preprocess script.')

        # index selected frames
        def _take_or_resample(arr):
            if arr is None:
                return None
            if arr.shape[0] == 0:
                return arr
            if max(idxs) < arr.shape[0]:
                return arr[idxs]
            # Precomputed caches may already be event-sampled to T frames.
            if arr.shape[0] == self.T:
                return arr
            ridx = np.linspace(0, arr.shape[0] - 1, self.T).astype(np.int64)
            return arr[ridx]

        roi_feats = _take_or_resample(roi_feats)
        roi_imgs = _take_or_resample(roi_imgs)
        landmarks = _take_or_resample(landmarks)
        mask = _take_or_resample(mask) if mask is not None else None
        palm_normal = _take_or_resample(palm_normal)
        quality_seq = _take_or_resample(quality_seq)

        if quality_seq is not None:
            q = np.array([str(x) for x in quality_seq.tolist()])
            quality_valid = ~np.isin(q, ['FRAME_MISSING', 'NO_HAND'])
            if mask is None:
                mask = quality_valid
            else:
                mask = np.logical_and(mask.astype(bool), quality_valid)

        out = {
            'label': torch.tensor(y, dtype=torch.long),
            'hand': e['hand'],
            'video_id': e['video_id'],
            'event_id': e['event_id'],
            'span': e['span'],
        }
        if roi_feats is not None:
            out['roi_feat'] = torch.tensor(roi_feats, dtype=torch.float32)
        if roi_imgs is not None:
            # Normalize ROI size to reduce memory and keep consistent backbone input.
            h, w = int(roi_imgs.shape[1]), int(roi_imgs.shape[2])
            if h != self.roi_size or w != self.roi_size:
                roi_imgs = np.stack(
                    [cv2.resize(roi_imgs[i], (self.roi_size, self.roi_size)) for i in range(roi_imgs.shape[0])],
                    axis=0,
                )
            # to float in [0,1], channel-first
            roi_imgs = roi_imgs.astype(np.float32) / 255.0
            roi_imgs = roi_imgs.transpose(0, 3, 1, 2)
            out['roi_img'] = torch.tensor(roi_imgs, dtype=torch.float32)
        if landmarks is not None:
            out['landmarks'] = torch.tensor(landmarks, dtype=torch.float32)
        if palm_normal is not None:
            out['palm_normal'] = torch.tensor(palm_normal, dtype=torch.float32)
        if mask is not None:
            out['mask'] = torch.tensor(mask.astype(np.float32))
        if quality_flags is not None:
            out['quality_flags'] = quality_flags
        return out
