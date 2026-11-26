#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# COCO-17 keypoint connections
COCO17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def read_pose_frames(pose_path: Path) -> List[Dict]:
    data = json.loads(pose_path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f"Invalid pose json format: {pose_path}")
    return data


def extract_kpts(frame_obj: Dict) -> Tuple[np.ndarray, np.ndarray]:
    inst = frame_obj.get('instances')
    if not isinstance(inst, dict):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    kpts = np.array(inst.get('keypoints', []), dtype=np.float32)
    scores = np.array(inst.get('keypoint_scores', []), dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if scores.ndim != 1 or scores.shape[0] != kpts.shape[0]:
        scores = np.ones((kpts.shape[0],), dtype=np.float32)
    return kpts, scores


def draw_skeleton(img: np.ndarray, kpts: np.ndarray, scores: np.ndarray, score_thr: float) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()

    # draw edges
    for a, b in COCO17_EDGES:
        if a >= kpts.shape[0] or b >= kpts.shape[0]:
            continue
        if scores[a] < score_thr or scores[b] < score_thr:
            continue
        xa, ya = int(kpts[a, 0]), int(kpts[a, 1])
        xb, yb = int(kpts[b, 0]), int(kpts[b, 1])
        if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
            cv2.line(out, (xa, ya), (xb, yb), (0, 255, 0), 2)

    # draw points
    for i in range(min(kpts.shape[0], 17)):
        if scores[i] < score_thr:
            continue
        x, y = int(kpts[i, 0]), int(kpts[i, 1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay pose keypoints on video frames")
    ap.add_argument('--video', required=True)
    ap.add_argument('--pose', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--max_frames', type=int, default=0, help='0 means all')
    ap.add_argument('--every', type=int, default=1, help='save every N frames')
    args = ap.parse_args()

    video_path = Path(args.video)
    pose_path = Path(args.pose)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_pose_frames(pose_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = len(frames)
    max_frames = args.max_frames if args.max_frames > 0 else total

    saved = 0
    for idx in range(min(total, max_frames)):
        if args.every > 1 and (idx % args.every != 0):
            continue
        frame_obj = frames[idx]
        abs_frame_id = int(frame_obj.get('frame_id', idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        kpts, scores = extract_kpts(frame_obj)
        if kpts.shape[0] == 0:
            continue
        vis = draw_skeleton(frame, kpts, scores, args.score_thr)
        out_path = out_dir / f"{video_path.stem}_local{idx:05d}_abs{abs_frame_id:05d}.png"
        cv2.imwrite(str(out_path), vis)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == '__main__':
    main()
