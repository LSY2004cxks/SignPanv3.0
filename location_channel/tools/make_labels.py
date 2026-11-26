import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as standalone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml, ensure_dir
from src.utils.io import read_json


def get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for p in path.split('.'):
        if not isinstance(cur, dict):
            return None
        if p not in cur:
            return None
        cur = cur[p]
    return cur


def get_field(obj: Any, field: Optional[str]) -> Any:
    if not field:
        return None
    if "." in field:
        return get_by_path(obj, field)
    if isinstance(obj, dict):
        return obj.get(field)
    return None


def get_by_paths(obj: Any, paths: List[str]) -> Any:
    for p in paths:
        v = get_by_path(obj, p)
        if v is not None:
            return v
    return None


def resolve_span(event: Dict[str, Any], mapping: Dict[str, str], fps: Optional[float]) -> Optional[Tuple[int, int]]:
    s_frame = event.get(mapping.get('span_start_field', 'start_frame'))
    e_frame = event.get(mapping.get('span_end_field', 'end_frame'))
    if isinstance(s_frame, (int, float)) and isinstance(e_frame, (int, float)):
        return int(s_frame), int(e_frame)

    # mhsl-1.0-lite span fallback
    span = event.get('span')
    if isinstance(span, dict):
        s_frame = span.get('start_frame')
        e_frame = span.get('end_frame')
        if isinstance(s_frame, (int, float)) and isinstance(e_frame, (int, float)):
            return int(s_frame), int(e_frame)

    # time-based fallback
    s_time = event.get(mapping.get('start_time_field', 'start_s'))
    e_time = event.get(mapping.get('end_time_field', 'end_s'))
    if fps is None:
        return None
    if isinstance(s_time, (int, float)) and isinstance(e_time, (int, float)):
        return int(round(s_time * fps)), int(round(e_time * fps))
    return None


def resolve_video_path(
    video_id: Optional[str],
    video_path: Optional[str],
    video_root: Optional[str],
    signer_id: Optional[str],
) -> Optional[str]:
    if isinstance(video_path, str) and len(video_path) > 0:
        p = Path(video_path)
        if p.is_absolute():
            return video_path
        if video_root:
            # try signer subdir first
            if signer_id:
                cand = Path(video_root) / signer_id / p.name
                if cand.exists():
                    return str(cand)
            cand = Path(video_root) / p.name
            if cand.exists():
                return str(cand)
        return video_path
    if isinstance(video_id, str) and video_root:
        # try signer subdir first
        if signer_id:
            cand = Path(video_root) / signer_id / f"{video_id}.mp4"
            if cand.exists():
                return str(cand)
        return str(Path(video_root) / video_id)
    return None


def maybe_find_clip(clip_root: Optional[str], sample_id: str) -> Optional[str]:
    if clip_root is None:
        return None
    p = Path(clip_root)
    if not p.exists():
        return None
    # attempt exact stem match
    cands = list(p.rglob(f"{sample_id}.*"))
    if not cands:
        return None
    # prefer mp4
    for c in cands:
        if c.suffix.lower() == '.mp4':
            return str(c)
    return str(cands[0])


def main():
    ap = argparse.ArgumentParser(description="Generate labels.csv from mhsl-1.0-lite annotations")
    ap.add_argument('--config', type=str, default='configs/dataset.yaml')
    ap.add_argument('--ann_dir', type=str, default=None)
    ap.add_argument('--out', type=str, default='data/labels.csv')
    ap.add_argument('--save_parquet', action='store_true')
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    mapping = cfg.get('mapping', {})
    paths = cfg.get('paths', {})
    label_cfg = cfg.get('label', {})

    ann_dir = Path(args.ann_dir or paths.get('ann_dir', ''))
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")

    video_root = paths.get('video_root')
    clip_root = paths.get('clips_root')

    ignore_labels = set(label_cfg.get('ignore_labels', []))
    location_paths = mapping.get('location_field_paths', ['anchors.start.location', 'position.region'])

    json_files = sorted(ann_dir.rglob('*.json'))

    rows = []

    for jf in json_files:
        data = read_json(jf)
        if not isinstance(data, dict):
            continue

        signer_id = get_field(data, mapping.get('signer_id_field', 'signer_id'))
        if isinstance(signer_id, int):
            signer_id = f"S{signer_id}"
        elif isinstance(signer_id, str) and signer_id.isdigit():
            signer_id = f"S{signer_id}"
        video_id = get_field(data, mapping.get('video_id_field', 'video_id'))
        fps = get_field(data, mapping.get('fps_field', 'fps'))
        if isinstance(fps, str):
            try:
                fps = float(fps)
            except Exception:
                fps = None

        video_path = resolve_video_path(
            video_id,
            get_field(data, mapping.get('video_path_field', 'video_path')),
            video_root,
            signer_id,
        )

        events = data.get(mapping.get('events_field', 'events'))
        if not isinstance(events, dict):
            continue

        for hand_key, hand_tag in [(mapping.get('left_hand_field', 'left_hand'), 'L'),
                                   (mapping.get('right_hand_field', 'right_hand'), 'R')]:
            ev_list = events.get(hand_key)
            if isinstance(ev_list, dict) and 'events' in ev_list:
                ev_list = ev_list.get('events')
            if not isinstance(ev_list, list):
                continue
            for ev in ev_list:
                if not isinstance(ev, dict):
                    continue
                if ev.get('event_type') is not None and ev.get('event_type') != 'effective':
                    continue
                event_id = ev.get(mapping.get('event_id_field', 'id'))
                if not isinstance(event_id, str):
                    continue
                loc = get_by_paths(ev, location_paths)
                if not isinstance(loc, str) or loc in ignore_labels:
                    continue

                span = resolve_span(ev, mapping, fps)
                if span is None:
                    continue
                start_frame, end_frame = span
                if end_frame <= start_frame:
                    continue

                # Build sample_id (stable key)
                vid_stem = Path(video_id).stem if isinstance(video_id, str) else 'video'
                signer_tag = signer_id if isinstance(signer_id, str) and signer_id else 'UNK'
                sample_id = f"{signer_tag}_{vid_stem}_{event_id}_{'LEFT' if hand_tag=='L' else 'RIGHT'}"

                clip_path = maybe_find_clip(clip_root, sample_id)

                rows.append({
                    'sample_id': sample_id,
                    'event_id': event_id,
                    'signer_id': signer_id,
                    'video_id': video_id,
                    'hand': hand_tag,
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'label_loc': loc,
                    'clip_path': clip_path,
                    'video_path': video_path,
                    'fps': fps,
                    'ann_path': str(jf)
                })

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'sample_id','event_id','signer_id','video_id','hand','start_frame','end_frame',
            'label_loc','clip_path','video_path','fps','ann_path'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.save_parquet:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_parquet(out_path.with_suffix('.parquet'))
        except Exception as e:
            print(f"[WARN] Parquet save failed: {e}")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
