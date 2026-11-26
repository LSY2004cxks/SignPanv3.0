from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from movement_channel.src.geometry import _item


def _span_tuple(span: Any) -> Optional[Tuple[int, int]]:
    if span is None:
        return None
    if isinstance(span, dict):
        start = span.get("start_frame", span.get("start"))
        end = span.get("end_frame", span.get("end"))
    elif isinstance(span, (list, tuple)) and len(span) >= 2:
        start, end = span[0], span[1]
    else:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def span_iou(left: Any, right: Any) -> float:
    a = _span_tuple(left)
    b = _span_tuple(right)
    if a is None or b is None:
        return 0.0
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _hand_key(hand: Optional[str]) -> str:
    value = str(hand or "").lower()
    if value.startswith("l"):
        return "left"
    return "right"


def _label_sequence(block: Dict[str, Any], key: str) -> Tuple[Optional[str], int, str]:
    payload = (block.get(key) or {}) if isinstance(block, dict) else {}
    labels = payload.get("label") or []
    types: List[str] = []
    repeat = 1
    for item in labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("type") or "").strip()
        if label:
            types.append(label)
            repeat = max(repeat, int(item.get("repeat") or 1))
    joined = "+".join(types) if types else None
    track_point = str(payload.get("track_point") or "WRIST")
    return joined, repeat, track_point


def annotation_path(ann_root: Path, signer_id: str, video_id: str) -> Path:
    return Path(ann_root) / f"{signer_id}-label" / f"{signer_id}_{video_id}.json"


def load_hand_events(path: Path, hand: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    return list((obj.get("hands") or {}).get(_hand_key(hand), {}).get("events") or [])


def _event_to_item(
    event: Dict[str, Any],
    *,
    signer_id: str,
    video_id: str,
    hand: str,
    requested_event_id: Optional[str],
) -> Dict[str, Any]:
    movement = event.get("movement") or {}
    global_label, repeat, track_point = _label_sequence(movement, "global")
    local_label, _, _ = _label_sequence(movement, "local")
    label = global_label or "NOM"
    return _item(
        event_id=str(event.get("event_id") or requested_event_id or ""),
        signer_id=signer_id,
        video_id=video_id,
        hand=hand,
        span=event.get("span"),
        label=label,
        top3=[label],
        probs=[1.0],
        local_label=local_label,
        track_point=track_point,
        gt=label,
        repeat=repeat,
    )


def select_event(
    events: Sequence[Dict[str, Any]],
    *,
    event_id: Optional[str],
    span: Any = None,
    min_iou: float = 0.3,
) -> Optional[Dict[str, Any]]:
    if span is not None:
        ranked = sorted(
            ((span_iou(span, event.get("span")), event) for event in events),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if ranked and ranked[0][0] >= min_iou:
            return ranked[0][1]
    if event_id:
        for event in events:
            if str(event.get("event_id")) == event_id:
                return event
    return None


def lookup_gold(
    *,
    ann_root: Path,
    signer_id: Optional[str],
    video_id: Optional[str],
    event_id: Optional[str],
    hand: Optional[str],
    span: Any = None,
) -> Optional[Dict[str, Any]]:
    if not signer_id or not video_id or not hand:
        return None
    events = load_hand_events(annotation_path(ann_root, signer_id, video_id), hand)
    event = select_event(events, event_id=event_id, span=span)
    if event is None:
        return None
    return _event_to_item(
        event,
        signer_id=signer_id,
        video_id=video_id,
        hand=hand,
        requested_event_id=event_id,
    )


def iter_video_annotations(ann_root: Path, video_id: str) -> Iterable[Path]:
    return Path(ann_root).glob(f"S*-label/*_{video_id}.json")


def cross_signer_items(
    *,
    ann_root: Path,
    signer_id: str,
    video_id: str,
    event_id: Optional[str],
    hand: str,
    span: Any = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in iter_video_annotations(ann_root, video_id):
        other_signer = path.stem.split("_", 1)[0]
        if other_signer == signer_id:
            continue
        events = load_hand_events(path, hand)
        event = select_event(events, event_id=event_id, span=span)
        if event is None:
            continue
        items.append(
            _event_to_item(
                event,
                signer_id=other_signer,
                video_id=video_id,
                hand=hand,
                requested_event_id=event_id,
            )
        )
    return items


def aggregate_reference_items(
    items: List[Dict[str, Any]],
    *,
    signer_id: str,
    video_id: str,
    event_id: str,
    hand: str,
    span: Any = None,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    local_counts: Dict[str, int] = {}
    for item in items:
        label = str(item.get("y_pred_top1") or "")
        if label:
            counts[label] = counts.get(label, 0) + 1
        local = str(item.get("local_label") or "")
        if local:
            local_counts[local] = local_counts.get(local, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    total = max(sum(counts.values()), 1)
    top3 = [label for label, _ in ranked[:3]] or ["NOM"]
    probs = [count / total for _, count in ranked[:3]] or [1.0]
    local_label = None
    if local_counts:
        local_label = max(local_counts.items(), key=lambda pair: pair[1])[0]
    return _item(
        event_id=event_id,
        signer_id=signer_id,
        video_id=video_id,
        hand=hand,
        span=span,
        label=top3[0],
        top3=top3,
        probs=probs,
        local_label=local_label,
        track_point="WRIST",
        reference_count=len(items),
    )
