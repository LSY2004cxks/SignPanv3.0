#!/usr/bin/env python3
"""Build event-level manifest from mhsl-1.0-lite JSON annotations.

Rules:
- Recursively scan all *.json
- Resolve video by video.path and signer folder
- Sort events by span.start_frame per hand before processing
- Keep only event_type == effective and handshape.label != NOH
- Output event-level manifest jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VIDEO_EXTS = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV")


@dataclass
class BuildStats:
    json_scanned: int = 0
    json_failed: int = 0
    events_total: int = 0
    events_kept: int = 0
    events_filtered_non_effective: int = 0
    events_filtered_noh: int = 0
    events_filtered_missing_label: int = 0
    events_filtered_bad_span: int = 0
    videos_missing: int = 0


def normalize_signer_id(raw: Any) -> str:
    if raw is None:
        return "S_UNKNOWN"
    sid = str(raw).strip()
    if sid.isdigit():
        return f"S{sid}"
    sid = sid.upper()
    if not sid.startswith("S") and sid:
        sid = f"S{sid}"
    return sid or "S_UNKNOWN"


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def build_video_index(data_root: Path) -> Dict[str, List[str]]:
    """Index video basename(lower) -> absolute paths for quick fallback lookup."""
    index: Dict[str, List[str]] = {}
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.MKV"):
        for p in data_root.rglob(ext):
            key = p.name.lower()
            index.setdefault(key, []).append(str(p.resolve()))
    return index


def resolve_video_path(
    data_root: Path,
    signer_id: str,
    video_path_value: str,
    video_index: Dict[str, List[str]],
) -> Optional[str]:
    """Resolve video file using signer folder + robust fallback by basename.

    The dataset stores relative filenames in video.path, e.g. 'xxx-36-6413.mp4'.
    """
    if not video_path_value:
        return None

    vp = video_path_value.replace("\\", "/")
    vp_name = Path(vp).name

    candidates: List[Path] = []
    signer_dir = data_root / signer_id

    # Preferred: signer folder + exact relative path/name
    candidates.append(signer_dir / vp)
    candidates.append(signer_dir / vp_name)

    # Try extension variants on same stem
    stem = Path(vp_name).stem
    for ext in VIDEO_EXTS:
        candidates.append(signer_dir / f"{stem}{ext}")

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # Recursive inside signer dir by basename (case-insensitive)
    if signer_dir.exists():
        target = vp_name.lower()
        for p in signer_dir.rglob("*"):
            if p.is_file() and p.name.lower() == target:
                return str(p.resolve())

    # Global fallback via prebuilt index
    key = vp_name.lower()
    if key in video_index and video_index[key]:
        # Prefer path containing signer_id if available
        signer_key = f"/{signer_id}/"
        for p in video_index[key]:
            norm = p.replace("\\", "/")
            if signer_key in norm:
                return p
        return video_index[key][0]

    return None


def iter_events_sorted(hands_obj: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for hand_key, hand_tag in (("left", "L"), ("right", "R")):
        hand_info = hands_obj.get(hand_key) or {}
        events = hand_info.get("events") or []
        if not isinstance(events, list):
            continue

        def event_start(ev: Dict[str, Any]) -> int:
            span = ev.get("span") or {}
            s = span.get("start_frame")
            if isinstance(s, int):
                return s
            try:
                return int(s)
            except Exception:
                return 0

        for ev in sorted((e for e in events if isinstance(e, dict)), key=event_start):
            yield hand_key, hand_tag, ev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data-28")
    parser.add_argument("--out_manifest", type=str, default="manifest/manifest.jsonl")
    parser.add_argument("--out_orphans", type=str, default="manifest/orphans.jsonl")
    parser.add_argument("--out_stats", type=str, default="manifest/manifest_stats.json")
    parser.add_argument("--allow_non_mhsl", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    out_orphans = Path(args.out_orphans).resolve()
    out_stats = Path(args.out_stats).resolve()

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_orphans.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_root.rglob("*.json"))
    stats = BuildStats()
    video_index = build_video_index(data_root)

    kept_rows: List[Dict[str, Any]] = []
    orphan_rows: List[Dict[str, Any]] = []
    labels: Dict[str, int] = {}

    for jp in json_files:
        stats.json_scanned += 1
        obj = safe_load_json(jp)
        if obj is None:
            stats.json_failed += 1
            continue

        schema_version = str(obj.get("schema_version", ""))
        if (not args.allow_non_mhsl) and (not schema_version.startswith("mhsl-1.0-lite")):
            # Non-target schema is skipped quietly
            continue

        video = obj.get("video") or {}
        hands = obj.get("hands") or {}
        if not isinstance(video, dict) or not isinstance(hands, dict):
            continue

        signer_id = normalize_signer_id(video.get("signer_id"))
        video_id = str(video.get("video_id", "")).strip()
        video_path_rel = str(video.get("path", "")).strip()
        fps = video.get("fps")
        num_frames = video.get("num_frames")

        resolved_video = resolve_video_path(data_root, signer_id, video_path_rel, video_index)
        if resolved_video is None:
            stats.videos_missing += 1
            orphan_rows.append(
                {
                    "json_path": str(jp.resolve()),
                    "signer_id": signer_id,
                    "video_id": video_id,
                    "video_path_rel": video_path_rel,
                    "reason": "video_not_found",
                }
            )
            continue

        for hand_key, hand_tag, ev in iter_events_sorted(hands):
            stats.events_total += 1
            event_type = str(ev.get("event_type", "")).strip()
            if event_type != "effective":
                stats.events_filtered_non_effective += 1
                continue

            hs = ev.get("handshape") or {}
            label = str((hs.get("label") if isinstance(hs, dict) else "") or "").strip()
            if not label:
                stats.events_filtered_missing_label += 1
                continue
            if label == "NOH":
                stats.events_filtered_noh += 1
                continue

            span = ev.get("span") or {}
            start = span.get("start_frame")
            end = span.get("end_frame")
            try:
                start_i = int(start)
                end_i = int(end)
            except Exception:
                stats.events_filtered_bad_span += 1
                continue

            if start_i < 0 or end_i < start_i:
                stats.events_filtered_bad_span += 1
                continue

            if isinstance(num_frames, int) and (start_i >= num_frames or end_i >= num_frames):
                stats.events_filtered_bad_span += 1
                continue

            event_id = str(ev.get("event_id", "")).strip() or "UNKNOWN_EVENT"
            sample_id = f"{signer_id}_{video_id}_{hand_tag}_{event_id}"

            row = {
                "sample_id": sample_id,
                "signer_id": signer_id,
                "video_id": video_id,
                "video_path": resolved_video,
                "video_path_rel": video_path_rel,
                "json_path": str(jp.resolve()),
                "hand": hand_tag,
                "hand_key": hand_key,
                "event_id": event_id,
                "span_start": start_i,
                "span_end": end_i,
                "num_frames": int(num_frames) if isinstance(num_frames, int) else None,
                "fps": float(fps) if isinstance(fps, (int, float)) else None,
                "gt_label": label,
                "event_type": event_type,
                "schema_version": schema_version,
            }
            kept_rows.append(row)
            labels[label] = labels.get(label, 0) + 1
            stats.events_kept += 1

    # deterministic ordering
    kept_rows.sort(key=lambda r: (r["signer_id"], r["video_id"], r["hand"], r["span_start"], r["event_id"]))

    with out_manifest.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_orphans.open("w", encoding="utf-8") as f:
        for row in orphan_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "data_root": str(data_root),
        "out_manifest": str(out_manifest),
        "out_orphans": str(out_orphans),
        "stats": vars(stats),
        "num_labels": len(labels),
        "label_counts": dict(sorted(labels.items(), key=lambda kv: kv[0])),
    }
    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
