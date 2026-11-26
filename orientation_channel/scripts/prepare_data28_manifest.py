import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VIDEO_EXTS = ('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV')


def iter_mhsl_files(label_root: Path) -> Iterable[Path]:
    for p in sorted(label_root.rglob('*.json')):
        if p.is_file():
            yield p


def load_mhsl(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get('schema_version') != 'mhsl-1.0-lite':
        return None
    return obj


def signer_from_path(json_path: Path) -> str:
    parent = json_path.parent.name
    m = re.fullmatch(r'(S\d+)-label', parent)
    return m.group(1) if m else ''


def normalize_signer_id(raw: Any, fallback: str = '') -> str:
    s = str(raw).strip() if raw is not None else ''
    if not s:
        return fallback
    if re.fullmatch(r'S\d+', s):
        return s
    if re.fullmatch(r'\d+', s):
        return f'S{s}'
    return s


def resolve_video_path(
    data_root: Path,
    signer_id: str,
    video_id: str,
    raw_video_path: str,
) -> Optional[Path]:
    candidates: List[Path] = []
    if raw_video_path:
        p = Path(raw_video_path)
        candidates.append(p)
        candidates.append(data_root / p)
        if signer_id:
            candidates.append(data_root / signer_id / p.name)
    if signer_id and video_id:
        for ext in VIDEO_EXTS:
            candidates.append(data_root / signer_id / f'{video_id}{ext}')

    for c in candidates:
        if c.exists():
            return c.resolve()

    if signer_id:
        signer_dir = data_root / signer_id
        if signer_dir.exists():
            preferred_name = Path(raw_video_path).name if raw_video_path else ''
            if preferred_name:
                exact = signer_dir / preferred_name
                if exact.exists():
                    return exact.resolve()
            for p in signer_dir.iterdir():
                if p.is_file() and p.suffix in VIDEO_EXTS and video_id and video_id in p.stem:
                    return p.resolve()
    return None


def extract_events(
    mhsl: Dict[str, Any],
    signer_id: str,
    video_id: str,
    video_path: str,
    json_path: str,
) -> List[Dict[str, Any]]:
    video = mhsl.get('video', {}) or {}
    fps = video.get('fps')
    num_frames = video.get('num_frames')
    video_uid = f'{signer_id}_{video_id}'

    out: List[Dict[str, Any]] = []
    hands = mhsl.get('hands', {}) or {}
    for hand in ('left', 'right'):
        events = (hands.get(hand) or {}).get('events') or []

        def start_frame_of(ev: Dict[str, Any]) -> int:
            span = ev.get('span') or {}
            return int(span.get('start_frame', 10**9))

        for idx, ev in enumerate(sorted(events, key=start_frame_of)):
            if ev.get('event_type') != 'effective':
                continue
            span = ev.get('span') or {}
            s = span.get('start_frame')
            e = span.get('end_frame')
            if s is None or e is None:
                continue
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if e < s:
                continue

            start_anchor = (ev.get('anchors') or {}).get('start') or {}
            orientation = (start_anchor.get('orientation') or {}).get('label')
            if orientation is None:
                continue
            orientation = str(orientation).strip()
            if (not orientation) or orientation.upper() == 'NOO':
                continue

            event_id = str(ev.get('event_id') or f'{hand}_event_{idx + 1}')
            out.append(
                {
                    'video_id': video_id,
                    'video_uid': video_uid,
                    'signer_id': signer_id,
                    'video_path': video_path,
                    'json_path': json_path,
                    'hand': hand,
                    'event_id': event_id,
                    'span': {'start_frame': s, 'end_frame': e},
                    'label_orientation': orientation,
                    'fps': fps,
                    'num_frames': num_frames,
                }
            )
    return out


def to_rel_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--label_root', default=None)
    parser.add_argument('--metadata_dir', default=None)
    parser.add_argument('--annotations_dir', default=None)
    parser.add_argument('--events_name', default='mhsl_events.jsonl')
    parser.add_argument('--orphan_name', default='orphans.jsonl')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    label_root = Path(args.label_root).resolve() if args.label_root else data_root
    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else data_root / 'metadata'
    annotations_dir = Path(args.annotations_dir).resolve() if args.annotations_dir else data_root / 'annotations'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    events_path = annotations_dir / args.events_name
    orphans_path = annotations_dir / args.orphan_name
    signers_path = metadata_dir / 'signers.csv'

    videos: Dict[Tuple[str, str], Dict[str, str]] = {}
    all_events: List[Dict[str, Any]] = []
    orphans: List[Dict[str, Any]] = []
    scanned = 0
    valid_mhsl = 0

    for json_file in iter_mhsl_files(label_root):
        scanned += 1
        mhsl = load_mhsl(json_file)
        if mhsl is None:
            continue
        valid_mhsl += 1

        video = mhsl.get('video', {}) or {}
        signer_id = normalize_signer_id(video.get('signer_id'), fallback=signer_from_path(json_file))
        video_id = str(video.get('video_id') or json_file.stem)
        raw_video_path = str(video.get('path') or '')

        resolved = resolve_video_path(data_root, signer_id, video_id, raw_video_path)
        if resolved is None:
            orphans.append(
                {
                    'json_path': str(json_file),
                    'signer_id': signer_id,
                    'video_id': video_id,
                    'video_path_raw': raw_video_path,
                    'reason': 'VIDEO_NOT_FOUND',
                }
            )
            continue

        rel_video_path = to_rel_path(resolved, data_root)
        video_uid = f'{signer_id}_{video_id}'
        videos[(signer_id, video_id)] = {
            'video_id': video_id,
            'video_uid': video_uid,
            'signer_id': signer_id,
            'video_path': rel_video_path,
        }

        events = extract_events(
            mhsl=mhsl,
            signer_id=signer_id,
            video_id=video_id,
            video_path=rel_video_path,
            json_path=str(json_file),
        )
        all_events.extend(events)

    with signers_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['video_id', 'video_uid', 'signer_id', 'video_path'],
        )
        writer.writeheader()
        for _, row in sorted(videos.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            writer.writerow(row)

    with events_path.open('w', encoding='utf-8') as f:
        for ev in all_events:
            f.write(json.dumps(ev, ensure_ascii=False) + '\n')

    with orphans_path.open('w', encoding='utf-8') as f:
        for row in orphans:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'Wrote: {signers_path}')
    print(f'Wrote: {events_path}')
    print(f'Wrote: {orphans_path}')
    print(
        f'Scanned JSON: {scanned} | Valid MHSL: {valid_mhsl} | '
        f'Videos: {len(videos)} | Events: {len(all_events)} | Orphans: {len(orphans)}'
    )


if __name__ == '__main__':
    main()
