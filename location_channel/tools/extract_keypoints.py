import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Allow running as standalone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from tqdm import tqdm

from src.utils.io import load_yaml_with_base, ensure_dir, write_json
from src.kpts.mediapipe_hand import MediaPipeHandDetector
from src.kpts.mmpose_wrapper import MMPoseWrapper
from src.kpts.align import ResizeTransform


def read_labels(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def sample_indices(start: int, end: int, num_frames: int) -> List[int]:
    """Uniformly sample fixed-length indices from [start, end).

    If the span is shorter than ``num_frames``, indices are repeated by
    interpolation/rounding (same behavior as other channels' fixed-T sampling).
    """
    if num_frames <= 0:
        return []
    if end <= start:
        return [start] * num_frames
    span = end - start
    idxs = np.linspace(0, span - 1, num_frames)
    idxs = [start + int(round(x)) for x in idxs]
    return idxs


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'frames': 0, 'fps': None}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {'frames': n, 'fps': fps}


def load_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    last_ok = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            last_ok = frame
            continue
        if last_ok is not None:
            # Keep fixed-length output even when occasional frame decode fails.
            frames.append(last_ok.copy())
    cap.release()
    if frames and len(frames) < len(frame_indices):
        pad_n = len(frame_indices) - len(frames)
        frames.extend([frames[-1].copy() for _ in range(pad_n)])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--num_frames', type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_with_base(Path(args.config))
    labels = read_labels(Path(args.labels))

    out_dir = ensure_dir(Path(args.out))

    num_frames = args.num_frames or cfg['sampling']['num_frames']

    # init detectors
    mp_hand = MediaPipeHandDetector(
        min_detection_confidence=cfg['hand']['min_detection_confidence'],
        min_tracking_confidence=cfg['hand']['min_tracking_confidence'],
        hand_center=cfg['hand'].get('hand_center', 'wrist')
    )

    # MMPose
    mm_cfg = cfg['pose']
    pose = MMPoseWrapper(mm_cfg.get('config'), mm_cfg.get('checkpoint'), device=mm_cfg.get('device', 'cpu'))

    for row in tqdm(labels, desc='Extract keypoints'):
        sample_id = row['sample_id']
        clip_path = row.get('clip_path')
        video_path = row.get('video_path')

        # use clip if exists else video+span
        path = clip_path if clip_path else video_path
        if not path:
            continue
        path = str(path)

        source_start = int(row['start_frame'])
        source_end = int(row['end_frame'])
        start_frame = source_start
        end_frame = source_end

        # if using pre-cut clip, ignore original span and use full clip
        if clip_path:
            info = get_video_info(path)
            if info['frames'] and info['frames'] > 0:
                start_frame = 0
                end_frame = info['frames']

        use_all = bool(cfg['sampling'].get('use_all_frames', False))
        max_frames = int(cfg['sampling'].get('max_frames', 120))
        if use_all:
            frame_indices = list(range(start_frame, end_frame))
            if len(frame_indices) > max_frames:
                frame_indices = sample_indices(start_frame, end_frame, max_frames)
        else:
            frame_indices = sample_indices(start_frame, end_frame, num_frames)
        frames = load_frames(path, frame_indices)
        if not frames:
            continue

        # optional resize transform (placeholder)
        transform = None

        T = len(frames)
        # pose arrays
        pose_kpts = []
        pose_score = []
        hand_point = []
        hand_score = []
        valid_mask = []

        expected_hand = 'L' if row.get('hand') == 'L' else 'R'

        expected_j = len(cfg['pose']['joint_names'])
        for frame in frames:
            img = frame
            if transform is not None:
                img = transform.apply_image(img)

            pose_res = pose.infer(img)
            hand_res = mp_hand.infer(img, expected_hand=expected_hand)

            if pose_res is None or hand_res is None:
                valid_mask.append(0)
                # fill placeholders
                pose_kpts.append(np.zeros((len(cfg['pose']['joint_names']), 2), dtype=np.float32))
                pose_score.append(np.zeros((len(cfg['pose']['joint_names']),), dtype=np.float32))
                hand_point.append(np.zeros((2,), dtype=np.float32))
                hand_score.append(np.zeros((1,), dtype=np.float32))
                continue

            pk = pose_res['pose_kpts']
            ps = pose_res['pose_score']
            hp = hand_res['hand_point']
            hs = np.array([hand_res['hand_score']], dtype=np.float32)

            if transform is not None:
                pk = transform.invert_points(pk)
                hp = transform.invert_points(hp)

            # align joint count to config
            if pk.shape[0] != expected_j:
                if pk.shape[0] > expected_j:
                    pk = pk[:expected_j]
                    ps = ps[:expected_j]
                else:
                    pad = expected_j - pk.shape[0]
                    pk = np.concatenate([pk, np.zeros((pad, 2), dtype=pk.dtype)], axis=0)
                    ps = np.concatenate([ps, np.zeros((pad,), dtype=ps.dtype)], axis=0)

            pose_kpts.append(pk)
            pose_score.append(ps)
            hand_point.append(hp)
            hand_score.append(hs)
            valid_mask.append(1)

        npz_path = out_dir / f"{sample_id}.npz"
        np.savez(npz_path,
                 pose_kpts=np.stack(pose_kpts, axis=0),
                 pose_score=np.stack(pose_score, axis=0),
                 hand_point=np.stack(hand_point, axis=0),
                 hand_score=np.stack(hand_score, axis=0).squeeze(-1),
                 valid_mask=np.array(valid_mask, dtype=np.uint8),
                 frame_indices=np.array(frame_indices, dtype=np.int32),
                 )

        meta = {
            'sample_id': sample_id,
            'video_path': path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'source_start_frame': source_start,
            'source_end_frame': source_end,
            'clip_used': bool(clip_path),
            'frame_indices': frame_indices,
            'transform': None if transform is None else {
                'src_size': transform.src_size,
                'dst_size': transform.dst_size,
            }
        }
        write_json(out_dir / f"{sample_id}.meta.json", meta)

    print(f"Done. Saved to {out_dir}")


if __name__ == '__main__':
    main()
