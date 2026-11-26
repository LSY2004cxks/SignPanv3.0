import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2


def _safe_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _raw_stem(signer_id: str, video_id: str, hand: str, event_id: str) -> str:
    side = "left" if hand.upper() == "L" else "right"
    return f"{signer_id}_{video_id}_{side}_{event_id}"


def _resolve_video_path(video_root: Path, signer_id: str, video_path: str) -> Path:
    p = Path(video_path)
    if p.is_absolute():
        return p
    c1 = video_root / signer_id / p
    if c1.exists():
        return c1
    return video_root / p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build location_channel data/features from precomputed pose+hand caches"
    )
    ap.add_argument("--labels", required=True, help="location_channel/data/labels.csv")
    ap.add_argument("--pose_root", required=True, help="data28_pose_effective")
    ap.add_argument("--hand_root", required=True, help="data28_mp_hand3d_effective")
    ap.add_argument("--out", required=True, help="location_channel/data/features")
    ap.add_argument(
        "--video_root",
        default="/home/yuan/桌面/SignPanv2.0/data28",
        help="root directory for videos referenced by labels.csv video_path",
    )
    ap.add_argument("--pose_joint_count", type=int, default=17, help="use first N pose joints")
    ap.add_argument("--hand_center", choices=["wrist", "mean"], default="wrist")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    pose_root = Path(args.pose_root)
    hand_root = Path(args.hand_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)

    rows = list(csv.DictReader(labels_path.open("r", encoding="utf-8")))
    wh_cache: Dict[str, tuple[float, float]] = {}

    built = 0
    miss_pose = 0
    miss_hand = 0
    failures = 0
    fallback_wh = 0

    for r in rows:
        sample_id = r["sample_id"]
        signer_id = r["signer_id"]
        video_id = r["video_id"]
        event_id = r["event_id"]
        hand = r["hand"]
        video_path = r.get("video_path", "")

        stem = _raw_stem(signer_id, video_id, hand, event_id)
        pose_path = pose_root / signer_id / f"{stem}.npz"
        hand_path = hand_root / signer_id / f"{stem}.npz"
        out_path = out_dir / f"{sample_id}.npz"

        if not pose_path.exists():
            miss_pose += 1
            continue
        if not hand_path.exists():
            miss_hand += 1
            continue

        try:
            pose = np.load(pose_path, allow_pickle=True)
            hand3d = np.load(hand_path, allow_pickle=True)

            pose_kpts = pose["keypoints"].astype(np.float32)[:, : args.pose_joint_count, :]
            pose_score = pose["scores"].astype(np.float32)[:, : args.pose_joint_count]

            xyz = hand3d["xyz"].astype(np.float32)  # (T,21,3)
            hand_xy = xyz[:, :, :2].astype(np.float32)
            if args.hand_center == "mean":
                hand_point = hand_xy.mean(axis=1)
            else:
                hand_point = hand_xy[:, 0, :]  # wrist

            # Precomputed MediaPipe xy is normalized to [0,1] in image coordinates.
            # Convert to pixel space before fusing with MMPose xy.
            wh_key = f"{signer_id}::{video_path}"
            if wh_key not in wh_cache:
                vp = _resolve_video_path(video_root, signer_id, video_path)
                cap = cv2.VideoCapture(str(vp))
                w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w <= 1.0 or h <= 1.0:
                    # Fallback: estimate frame size from pose coordinates.
                    pose_x = pose_kpts[..., 0]
                    pose_y = pose_kpts[..., 1]
                    w = float(np.nanpercentile(np.clip(pose_x, 0, None), 99.5))
                    h = float(np.nanpercentile(np.clip(pose_y, 0, None), 99.5))
                    if w <= 1.0 or h <= 1.0:
                        raise RuntimeError(f"Cannot read/estimate valid video size: {vp}")
                    fallback_wh += 1
                wh_cache[wh_key] = (w, h)
            w, h = wh_cache[wh_key]
            hand_point = hand_point.copy()
            hand_point[:, 0] *= w
            hand_point[:, 1] *= h

            if "detected" in hand3d.files:
                hand_score = hand3d["detected"].astype(np.float32)
            else:
                hand_score = np.ones((xyz.shape[0],), dtype=np.float32)

            pose_valid = (pose_score.max(axis=1) > 0.0).astype(np.uint8)
            hand_valid = (hand_score > 0.0).astype(np.uint8)
            valid_mask = (pose_valid * hand_valid).astype(np.uint8)

            frame_indices = (
                pose["frame_ids"].astype(np.int32)
                if "frame_ids" in pose.files
                else np.arange(pose_kpts.shape[0], dtype=np.int32)
            )

            np.savez(
                out_path,
                pose_kpts=pose_kpts,
                pose_score=pose_score,
                hand_point=hand_point.astype(np.float32),
                hand_score=hand_score.astype(np.float32),
                valid_mask=valid_mask,
                frame_indices=frame_indices,
                source_pose=str(pose_path),
                source_hand=str(hand_path),
            )
            built += 1
        except Exception:
            failures += 1
            continue

    report = {
        "labels_rows": len(rows),
        "built_features": built,
        "missing_pose": miss_pose,
        "missing_hand": miss_hand,
        "failures": failures,
        "fallback_wh": fallback_wh,
        "video_size_cache": len(wh_cache),
        "out_dir": str(out_dir),
    }
    (out_dir / "_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
