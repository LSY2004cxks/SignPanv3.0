#!/usr/bin/env python3
"""Generate ROI crops and skeleton cache from keypoint cache.

Outputs:
- cache/roi/{sample_id}.npy           (K,224,224,3)
- cache/skeleton/{sample_id}.npy      (K,21,3)
- cache/meta/{sample_id}.json         (updated with tier + quality)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def tier_from_bbox_side(side: float) -> str:
    if side >= 96:
        return "A"
    if side >= 64:
        return "B"
    if side >= 48:
        return "C"
    return "D"


def crop_with_bbox(frame: np.ndarray, bbox_xyxy: np.ndarray, out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy.astype(np.float32).tolist()
    x0 = int(np.clip(round(x0), 0, max(0, w - 1)))
    y0 = int(np.clip(round(y0), 0, max(0, h - 1)))
    x1 = int(np.clip(round(x1), x0 + 1, max(1, w - 1)))
    y1 = int(np.clip(round(y1), y0 + 1, max(1, h - 1)))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def expand_bbox(frame: np.ndarray, bbox_xyxy: np.ndarray, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy.astype(np.float32).tolist()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    bw2 = 0.5 * bw * float(scale)
    bh2 = 0.5 * bh * float(scale)

    nx0 = int(np.clip(round(cx - bw2), 0, max(0, w - 1)))
    ny0 = int(np.clip(round(cy - bh2), 0, max(0, h - 1)))
    nx1 = int(np.clip(round(cx + bw2), nx0 + 1, max(1, w - 1)))
    ny1 = int(np.clip(round(cy + bh2), ny0 + 1, max(1, h - 1)))
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)


def center_crop(frame: np.ndarray, out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config).resolve())
    data_cfg = cfg.get("data", {})

    manifest_path = Path(args.manifest or data_cfg.get("manifest_path", "manifest/manifest.jsonl")).resolve()
    rows = read_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    keypoint_dir = Path(data_cfg.get("cache_keypoints_dir", "cache/keypoints")).resolve()
    meta_dir = Path(data_cfg.get("cache_meta_dir", "cache/meta")).resolve()
    roi_dir = Path(data_cfg.get("cache_roi_dir", "cache/roi")).resolve()
    skeleton_dir = Path(data_cfg.get("cache_skeleton_dir", "cache/skeleton")).resolve()

    roi_size = int(data_cfg.get("roi_size", 224))
    roi_expand_small = bool(data_cfg.get("roi_expand_small", False))
    roi_expand_min_side = float(data_cfg.get("roi_expand_min_side", 48.0))
    roi_expand_scale = float(data_cfg.get("roi_expand_scale", 1.25))

    ensure_dir(meta_dir)
    ensure_dir(roi_dir)
    ensure_dir(skeleton_dir)

    failures = 0

    for i, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        out_roi = roi_dir / f"{sample_id}.npy"
        out_skel = skeleton_dir / f"{sample_id}.npy"
        out_meta = meta_dir / f"{sample_id}.json"

        if out_roi.exists() and out_skel.exists() and (not args.overwrite):
            continue

        kp_file = keypoint_dir / f"{sample_id}.npz"
        if not kp_file.exists():
            failures += 1
            print(f"[WARN] missing keypoint cache for {sample_id}: {kp_file}")
            continue

        try:
            pack = np.load(kp_file)
            frame_indices = np.asarray(pack["frame_indices"]).astype(np.int64)
            keypoints = np.asarray(pack["keypoints"]).astype(np.float32)
            bboxes = np.asarray(pack["bboxes"]).astype(np.float32)
            side_px = np.asarray(pack["bbox_side_px"]).astype(np.float32)
            lowq = np.asarray(pack["low_quality"]).astype(np.uint8)

            cap = cv2.VideoCapture(str(row["video_path"]))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {row['video_path']}")

            rois: List[np.ndarray] = []
            for fi, bbox in zip(frame_indices.tolist(), bboxes):
                frame = read_frame_at(cap, fi)
                if frame is None:
                    # fallback: black frame
                    rois.append(np.zeros((roi_size, roi_size, 3), dtype=np.uint8))
                    continue
                side = min(max(0.0, float(bbox[2] - bbox[0])), max(0.0, float(bbox[3] - bbox[1])))
                if side > 1.0:
                    bbox_use = bbox
                    if roi_expand_small and side < roi_expand_min_side:
                        bbox_use = expand_bbox(frame, bbox, roi_expand_scale)
                    roi = crop_with_bbox(frame, bbox_use, roi_size)
                else:
                    roi = center_crop(frame, roi_size)
                rois.append(roi.astype(np.uint8))

            cap.release()

            roi_np = np.stack(rois, axis=0).astype(np.uint8)
            sk_np = keypoints.astype(np.float32)

            np.save(out_roi, roi_np)
            np.save(out_skel, sk_np)

            side_median = float(np.median(side_px)) if side_px.size else 0.0
            tier = tier_from_bbox_side(side_median)
            roi_too_small = bool(side_median < 48.0)

            # Merge existing meta generated by preprocess_keypoints
            meta: Dict[str, Any] = {}
            if out_meta.exists():
                try:
                    with out_meta.open("r", encoding="utf-8") as f:
                        old_meta = json.load(f)
                    if isinstance(old_meta, dict):
                        meta.update(old_meta)
                except Exception:
                    pass

            meta.update(
                {
                    "sample_id": sample_id,
                    "bbox_side_px_median": side_median,
                    "bbox_side_px": side_px.tolist(),
                    "tier": tier,
                    "ROI_TOO_SMALL": roi_too_small,
                    "low_quality_ratio": float(np.mean(lowq.astype(np.float32))) if lowq.size else 1.0,
                    "k": int(roi_np.shape[0]),
                    "roi_shape": [int(x) for x in roi_np.shape],
                    "skeleton_shape": [int(x) for x in sk_np.shape],
                }
            )
            with out_meta.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            failures += 1
            print(f"[WARN] failed sample={sample_id}: {e}")

        if i % 100 == 0:
            print(f"processed {i}/{len(rows)} failures={failures}")

    print(
        json.dumps(
            {
                "processed": len(rows),
                "failures": failures,
                "cache_roi_dir": str(roi_dir),
                "cache_skeleton_dir": str(skeleton_dir),
                "cache_meta_dir": str(meta_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
