#!/usr/bin/env python3
"""Preprocess event-level keypoints and bboxes.

Outputs:
- cache/keypoints/{sample_id}.npz
- cache/meta/{sample_id}.json

Engine policy:
- default engine from config (usually mmpose)
- if health csv exists and has best_engine, override per video
- hybrid: run mmpose + mediapipe on same frame and choose by
  success > bbox_side_px > confidence
- if both fail: fallback previous bbox up to 3 frames; mark low_quality
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


MP_LEFT = "Left"
MP_RIGHT = "Right"


@dataclass
class DetectResult:
    success: bool
    keypoints: np.ndarray  # (21,3), normalized xy in [0,1] if success
    bbox_xyxy: np.ndarray  # (4,), pixel space
    bbox_side_px: float
    score: float
    engine: str
    low_quality: bool = False


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


def sample_frame_indices(start: int, end: int, k: int) -> np.ndarray:
    if end < start:
        return np.full((k,), start, dtype=np.int64)
    if k <= 1:
        return np.array([start], dtype=np.int64)
    xs = np.linspace(start, end, num=k)
    idx = np.round(xs).astype(np.int64)
    idx = np.clip(idx, start, end)
    return idx


def compute_bbox_side(bbox_xyxy: np.ndarray) -> float:
    w = max(0.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
    h = max(0.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
    return min(w, h)


def empty_result(engine: str) -> DetectResult:
    return DetectResult(
        success=False,
        keypoints=np.zeros((21, 3), dtype=np.float32),
        bbox_xyxy=np.zeros((4,), dtype=np.float32),
        bbox_side_px=0.0,
        score=0.0,
        engine=engine,
        low_quality=True,
    )


class MediaPipeDetector:
    def __init__(self, min_det_conf: float = 0.4) -> None:
        import mediapipe as mp

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=float(min_det_conf),
        )

    def close(self) -> None:
        self.hands.close()

    def detect(self, frame_bgr: np.ndarray, hand: str, need_swap: bool) -> DetectResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.hands.process(rgb)

        if not out.multi_hand_landmarks or not out.multi_handedness:
            return empty_result("mediapipe")

        target = MP_LEFT if hand == "L" else MP_RIGHT
        if need_swap:
            target = MP_RIGHT if target == MP_LEFT else MP_LEFT

        best_idx = -1
        best_score = -1.0
        for i, handedness in enumerate(out.multi_handedness):
            cls = handedness.classification[0]
            label = str(cls.label)
            score = float(cls.score)
            if label == target and score > best_score:
                best_idx = i
                best_score = score

        # fallback to highest-score hand if target side was not found
        if best_idx < 0:
            for i, handedness in enumerate(out.multi_handedness):
                cls = handedness.classification[0]
                score = float(cls.score)
                if score > best_score:
                    best_idx = i
                    best_score = score

        if best_idx < 0:
            return empty_result("mediapipe")

        lm = out.multi_hand_landmarks[best_idx]
        kps = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

        xs = np.clip(kps[:, 0] * w, 0, w - 1)
        ys = np.clip(kps[:, 1] * h, 0, h - 1)
        x0, y0 = float(xs.min()), float(ys.min())
        x1, y1 = float(xs.max()), float(ys.max())
        bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
        side = compute_bbox_side(bbox)

        return DetectResult(
            success=True,
            keypoints=kps,
            bbox_xyxy=bbox,
            bbox_side_px=side,
            score=max(0.0, best_score),
            engine="mediapipe",
            low_quality=False,
        )


class MMPoseDetector:
    """MMPose wholebody detector wrapper.

    Requires mmpose installation. We use MMPoseInferencer and extract
    hand keypoints from COCO-wholebody indices:
    - left hand: 91..111 (21)
    - right hand: 112..132 (21)
    """

    def __init__(self, pose2d: str = "wholebody") -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except Exception as e:
            raise RuntimeError(
                "mmpose is required for engine=mmpose/hybrid but is not available."
            ) from e

        # tolerate API differences by using keyword when possible
        try:
            self.inferencer = MMPoseInferencer(pose2d=pose2d)
        except Exception:
            self.inferencer = MMPoseInferencer(pose2d)

    def _extract_predictions(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        try:
            result = next(self.inferencer(frame_bgr, return_vis=False))
        except TypeError:
            # fallback for older signatures
            result = next(self.inferencer(frame_bgr))

        preds = result.get("predictions", [])
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        if not isinstance(preds, list):
            return []
        return [p for p in preds if isinstance(p, dict)]

    def detect(self, frame_bgr: np.ndarray, hand: str, need_swap: bool) -> DetectResult:
        h, w = frame_bgr.shape[:2]
        preds = self._extract_predictions(frame_bgr)
        if not preds:
            return empty_result("mmpose")

        # choose best person by bbox_score if available
        def person_score(p: Dict[str, Any]) -> float:
            bs = p.get("bbox_score")
            if isinstance(bs, (int, float)):
                return float(bs)
            return 0.0

        person = max(preds, key=person_score)
        kps = np.asarray(person.get("keypoints", []), dtype=np.float32)
        kps_score = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)
        if kps.ndim != 2 or kps.shape[0] < 133 or kps.shape[1] < 2:
            return empty_result("mmpose")

        # signer hand side with optional swap
        target = "L" if hand == "L" else "R"
        if need_swap:
            target = "R" if target == "L" else "L"

        if target == "L":
            idx0, idx1 = 91, 112
        else:
            idx0, idx1 = 112, 133

        hand_kps = kps[idx0:idx1, :2]
        if hand_kps.shape[0] != 21:
            return empty_result("mmpose")

        # normalize to [0,1] to keep skeleton format consistent
        hand_norm = np.zeros((21, 3), dtype=np.float32)
        hand_norm[:, 0] = np.clip(hand_kps[:, 0] / max(1, w), 0.0, 1.0)
        hand_norm[:, 1] = np.clip(hand_kps[:, 1] / max(1, h), 0.0, 1.0)

        xs = np.clip(hand_kps[:, 0], 0, w - 1)
        ys = np.clip(hand_kps[:, 1], 0, h - 1)
        bbox = np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)
        side = compute_bbox_side(bbox)

        if kps_score.size >= idx1:
            score = float(np.nanmean(kps_score[idx0:idx1]))
        else:
            score = person_score(person)

        valid = math.isfinite(side) and side > 0.0
        if not valid:
            return empty_result("mmpose")

        return DetectResult(
            success=True,
            keypoints=hand_norm,
            bbox_xyxy=bbox,
            bbox_side_px=side,
            score=max(0.0, score),
            engine="mmpose",
            low_quality=False,
        )


def parse_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def load_health_csv(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load optional health overrides by video_id / video basename.

    Expected columns (any subset):
    - video_id
    - video_path / path
    - best_engine
    - sample_strategy.swap_left_right / swap_left_right / sample_strategy
    """
    if path is None or (not path.exists()):
        return {}

    table: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue

            video_id = (row.get("video_id") or "").strip()
            vpath = (row.get("video_path") or row.get("path") or "").strip()
            key_candidates: List[str] = []
            if video_id:
                key_candidates.append(video_id)
            if vpath:
                key_candidates.append(Path(vpath).name)

            best_engine = (row.get("best_engine") or "").strip().lower() or None

            swap = parse_bool(row.get("sample_strategy.swap_left_right"))
            if swap is None:
                swap = parse_bool(row.get("swap_left_right"))
            if swap is None:
                raw = row.get("sample_strategy")
                if raw:
                    try:
                        obj = json.loads(raw)
                        swap = parse_bool((obj or {}).get("swap_left_right"))
                    except Exception:
                        swap = None

            payload = {
                "best_engine": best_engine,
                "swap_left_right": swap,
            }
            for key in key_candidates:
                table[key] = payload
    return table


def pick_engine(
    row: Dict[str, Any],
    default_engine: str,
    health: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[bool]]:
    video_id = str(row.get("video_id", ""))
    vpath = str(row.get("video_path", ""))
    vname = Path(vpath).name

    h = health.get(video_id) or health.get(vname)
    if not h:
        return default_engine, None

    eng = h.get("best_engine") or default_engine
    swap = h.get("swap_left_right")
    return str(eng).lower(), swap


def choose_hybrid(a: DetectResult, b: DetectResult) -> DetectResult:
    if a.success and (not b.success):
        return a
    if b.success and (not a.success):
        return b
    if (not a.success) and (not b.success):
        return empty_result("hybrid")

    # both success: prefer larger bbox_side_px, then higher score
    if a.bbox_side_px > b.bbox_side_px:
        return a
    if b.bbox_side_px > a.bbox_side_px:
        return b
    if a.score >= b.score:
        return a
    return b


def clamp_bbox(bbox: np.ndarray, w: int, h: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox.tolist()
    x0 = float(np.clip(x0, 0, max(0, w - 1)))
    y0 = float(np.clip(y0, 0, max(0, h - 1)))
    x1 = float(np.clip(x1, x0 + 1, max(1, w - 1)))
    y1 = float(np.clip(y1, y0 + 1, max(1, h - 1)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def process_sample(
    row: Dict[str, Any],
    k: int,
    engine_name: str,
    need_swap: bool,
    mp_detector: Optional[MediaPipeDetector],
    mm_detector: Optional[MMPoseDetector],
    fallback_max_frames: int,
    out_npz: Path,
    out_meta: Path,
    overwrite: bool,
) -> None:
    if out_npz.exists() and (not overwrite):
        return

    video_path = str(row["video_path"])
    hand = str(row["hand"])
    start = int(row["span_start"])
    end = int(row["span_end"])

    idxs = sample_frame_indices(start, end, k)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    kps_all = np.zeros((k, 21, 3), dtype=np.float32)
    bbox_all = np.zeros((k, 4), dtype=np.float32)
    side_all = np.zeros((k,), dtype=np.float32)
    score_all = np.zeros((k,), dtype=np.float32)
    lowq_all = np.ones((k,), dtype=np.uint8)
    engine_all: List[str] = []

    prev_bbox: Optional[np.ndarray] = None
    prev_kps: Optional[np.ndarray] = None
    fallback_left = 0

    for i, fi in enumerate(idxs.tolist()):
        frame = read_frame_at(cap, fi)
        if frame is None:
            det = empty_result(engine_name)
        else:
            h, w = frame.shape[:2]
            if engine_name == "mediapipe":
                if mp_detector is None:
                    raise RuntimeError("mediapipe engine requested but detector is not initialized")
                det = mp_detector.detect(frame, hand, need_swap)
            elif engine_name == "mmpose":
                if mm_detector is None:
                    raise RuntimeError("mmpose engine requested but detector is not initialized")
                det = mm_detector.detect(frame, hand, need_swap)
            elif engine_name == "hybrid":
                if mp_detector is None:
                    raise RuntimeError("hybrid engine requested but mediapipe detector is not initialized")
                if mm_detector is None:
                    raise RuntimeError("hybrid engine requested but mmpose detector is not initialized")
                det_mm = mm_detector.detect(frame, hand, need_swap)
                det_mp = mp_detector.detect(frame, hand, need_swap)
                det = choose_hybrid(det_mm, det_mp)
                det.engine = "hybrid:" + det.engine
            else:
                raise ValueError(f"Unsupported engine: {engine_name}")

            if det.success:
                bbox = clamp_bbox(det.bbox_xyxy, w, h)
                det.bbox_xyxy = bbox
                det.bbox_side_px = compute_bbox_side(bbox)
                prev_bbox = bbox.copy()
                prev_kps = det.keypoints.copy()
                fallback_left = fallback_max_frames
                det.low_quality = False
            else:
                # bounded fallback with previous bbox/keypoints
                if prev_bbox is not None and fallback_left > 0:
                    det.success = True
                    det.bbox_xyxy = prev_bbox.copy()
                    det.bbox_side_px = compute_bbox_side(det.bbox_xyxy)
                    det.keypoints = prev_kps.copy() if prev_kps is not None else np.zeros((21, 3), dtype=np.float32)
                    det.score = 0.0
                    det.low_quality = True
                    det.engine = det.engine + ":fallback_prev"
                    fallback_left -= 1

        kps_all[i] = det.keypoints
        bbox_all[i] = det.bbox_xyxy
        side_all[i] = float(det.bbox_side_px)
        score_all[i] = float(det.score)
        lowq_all[i] = 1 if det.low_quality else 0
        engine_all.append(det.engine)

    cap.release()

    ensure_dir(out_npz.parent)
    ensure_dir(out_meta.parent)
    np.savez_compressed(
        out_npz,
        frame_indices=idxs.astype(np.int64),
        keypoints=kps_all,
        bboxes=bbox_all,
        bbox_side_px=side_all,
        scores=score_all,
        low_quality=lowq_all,
    )

    meta = {
        "sample_id": row["sample_id"],
        "video_id": row.get("video_id"),
        "signer_id": row.get("signer_id"),
        "hand": row.get("hand"),
        "event_id": row.get("event_id"),
        "span": [int(row["span_start"]), int(row["span_end"])],
        "engine_requested": engine_name,
        "engine_used_per_frame": engine_all,
        "need_swap": bool(need_swap),
        "k": int(k),
        "bbox_side_px": side_all.tolist(),
        "bbox_side_px_median": float(np.median(side_all)) if side_all.size else 0.0,
        "low_quality_ratio": float(np.mean(lowq_all.astype(np.float32))) if lowq_all.size else 1.0,
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--health_csv", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config).resolve())

    data_cfg = cfg.get("data", {})
    prep_cfg = cfg.get("preprocess", {})
    engine_cfg = prep_cfg.get("engine", {})

    manifest_path = Path(args.manifest or data_cfg.get("manifest_path", "manifest/manifest.jsonl")).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    health_csv = args.health_csv or engine_cfg.get("health_csv")
    health = load_health_csv(Path(health_csv).resolve() if health_csv else None)

    default_engine = str(engine_cfg.get("default", "mmpose")).lower()
    fallback_max_frames = int(engine_cfg.get("fallback_max_frames", 3))
    global_swap = bool(engine_cfg.get("swap_left_right", False))
    k = int(data_cfg.get("k", 8))

    keypoint_dir = Path(data_cfg.get("cache_keypoints_dir", "cache/keypoints")).resolve()
    meta_dir = Path(data_cfg.get("cache_meta_dir", "cache/meta")).resolve()
    ensure_dir(keypoint_dir)
    ensure_dir(meta_dir)

    # detect actual engine requirements from effective run plan
    need_mmpose = False
    need_mediapipe = False
    for r in rows:
        e, _ = pick_engine(r, default_engine, health)
        if e == "mmpose":
            need_mmpose = True
        elif e == "mediapipe":
            need_mediapipe = True
        elif e == "hybrid":
            need_mmpose = True
            need_mediapipe = True
        else:
            raise ValueError(f"Unsupported engine in plan: {e}")

    mp_detector: Optional[MediaPipeDetector] = None
    mm_detector: Optional[MMPoseDetector] = None

    if need_mediapipe:
        mp_detector = MediaPipeDetector(min_det_conf=float(engine_cfg.get("mediapipe_min_det_conf", 0.4)))
    if need_mmpose:
        pose_alias = str(engine_cfg.get("mmpose_pose2d", "wholebody"))
        mm_detector = MMPoseDetector(pose2d=pose_alias)

    failures = 0
    for i, row in enumerate(rows, start=1):
        engine_name, swap_override = pick_engine(row, default_engine, health)
        need_swap = bool(global_swap or (swap_override is True))

        out_npz = keypoint_dir / f"{row['sample_id']}.npz"
        out_meta = meta_dir / f"{row['sample_id']}.json"

        try:
            process_sample(
                row=row,
                k=k,
                engine_name=engine_name,
                need_swap=need_swap,
                mp_detector=mp_detector,
                mm_detector=mm_detector,
                fallback_max_frames=fallback_max_frames,
                out_npz=out_npz,
                out_meta=out_meta,
                overwrite=args.overwrite,
            )
        except Exception as e:
            failures += 1
            print(f"[WARN] failed sample={row['sample_id']}: {e}")

        if i % 100 == 0:
            print(f"processed {i}/{len(rows)} failures={failures}")

    if mp_detector is not None:
        mp_detector.close()
    print(
        json.dumps(
            {
                "processed": len(rows),
                "failures": failures,
                "cache_keypoints_dir": str(keypoint_dir),
                "cache_meta_dir": str(meta_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
