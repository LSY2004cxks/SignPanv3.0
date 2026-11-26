import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRUE_LABELS = {
    "valid",
    "effective",
    "eff",
    "true",
    "yes",
    "1",
}
FALSE_LABELS = {
    "invalid",
    "non_effective",
    "ineffective",
    "false",
    "no",
    "0",
    "noise",
}


@dataclass
class ParsedEvent:
    event_idx: int
    start: int
    end: int
    explicit_valid: Optional[bool]
    inferred_valid: bool
    train_valid: bool
    has_anchor_end: bool
    movement_global_count: int
    movement_local_count: int
    raw_type: Optional[str]


@dataclass
class ParsedLabel:
    ok: bool
    schema_type: str
    clip_id: str
    video_id: str
    signer_id: str
    fps: Optional[float]
    n_frames: Optional[int]
    events_by_hand: Dict[str, List[ParsedEvent]]
    parse_errors: List[str]


def normalize_hand(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"l", "left", "lefthand", "left_hand"}:
        return "L"
    if s in {"r", "right", "righthand", "right_hand"}:
        return "R"
    return None


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def extract_span(event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    span = event.get("span")
    if isinstance(span, dict):
        s = _int_or_none(span.get("start_frame", span.get("start")))
        e = _int_or_none(span.get("end_frame", span.get("end")))
        if s is not None and e is not None:
            return s, e
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        s = _int_or_none(span[0])
        e = _int_or_none(span[1])
        if s is not None and e is not None:
            return s, e

    s = _int_or_none(event.get("start_frame", event.get("start")))
    e = _int_or_none(event.get("end_frame", event.get("end")))
    if s is not None and e is not None:
        return s, e
    return None


def _explicit_valid(event: Dict[str, Any]) -> Optional[bool]:
    if isinstance(event.get("valid"), bool):
        return bool(event["valid"])
    if isinstance(event.get("is_valid"), bool):
        return bool(event["is_valid"])

    for key in ("type", "event_type"):
        if key in event and event[key] is not None:
            s = str(event[key]).strip().lower()
            if s in TRUE_LABELS:
                return True
            if s in FALSE_LABELS:
                return False
    return None


def _inferred_valid(event: Dict[str, Any]) -> bool:
    handshape_ok = False
    handshape = event.get("handshape")
    if isinstance(handshape, dict):
        handshape_ok = "label" in handshape and handshape.get("label") is not None
    elif handshape is not None:
        handshape_ok = True

    anchors = event.get("anchors") if isinstance(event.get("anchors"), dict) else {}
    astart = anchors.get("start") if isinstance(anchors.get("start"), dict) else {}
    loc_ok = isinstance(astart.get("location"), dict)
    ori_ok = isinstance(astart.get("orientation"), dict)
    return bool(handshape_ok and loc_ok and ori_ok)


def _count_list_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def _event_to_parsed(idx: int, event: Dict[str, Any]) -> Optional[ParsedEvent]:
    span = extract_span(event)
    if span is None:
        return None
    start, end = span

    explicit = _explicit_valid(event)
    inferred = _inferred_valid(event)
    train_valid = explicit if explicit is not None else inferred

    anchors = event.get("anchors") if isinstance(event.get("anchors"), dict) else {}
    has_anchor_end = anchors.get("end") is not None
    movement = event.get("movement") if isinstance(event.get("movement"), dict) else {}
    mg = _count_list_len(movement.get("global"))
    ml = _count_list_len(movement.get("local"))

    raw_type = None
    if event.get("event_type") is not None:
        raw_type = str(event.get("event_type"))
    elif event.get("type") is not None:
        raw_type = str(event.get("type"))

    return ParsedEvent(
        event_idx=idx,
        start=start,
        end=end,
        explicit_valid=explicit,
        inferred_valid=inferred,
        train_valid=bool(train_valid),
        has_anchor_end=bool(has_anchor_end),
        movement_global_count=mg,
        movement_local_count=ml,
        raw_type=raw_type,
    )


def _extract_schema_events(label: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, List[Dict[str, Any]]]], str]:
    # Schema A: label["left"]["events"], label["right"]["events"]
    if isinstance(label.get("left"), dict) and isinstance(label.get("right"), dict):
        le = label["left"].get("events")
        re = label["right"].get("events")
        if isinstance(le, list) and isinstance(re, list):
            return "A", {"L": le, "R": re}, "ok"

    # Schema B: label["hands"][L/R or left/right].events
    hands = label.get("hands")
    if isinstance(hands, dict):
        l_obj = hands.get("L") if isinstance(hands.get("L"), dict) else hands.get("left")
        r_obj = hands.get("R") if isinstance(hands.get("R"), dict) else hands.get("right")
        if isinstance(l_obj, dict) and isinstance(r_obj, dict):
            le = l_obj.get("events")
            re = r_obj.get("events")
            if isinstance(le, list) and isinstance(re, list):
                return "B", {"L": le, "R": re}, "ok"

    # Schema C: label["events"] list with event["hand"]
    events = label.get("events")
    if isinstance(events, list):
        out = {"L": [], "R": []}
        has_hand_tag = False
        for ev in events:
            if not isinstance(ev, dict):
                continue
            hand = normalize_hand(ev.get("hand"))
            if hand is None:
                continue
            has_hand_tag = True
            out[hand].append(ev)
        if has_hand_tag:
            return "C", out, "ok"

    return "unknown", None, "cannot locate event list for schema A/B/C"


def parse_label_file(path: str) -> ParsedLabel:
    clip_id = Path(path).stem
    try:
        with open(path, "r", encoding="utf-8") as f:
            label = json.load(f)
    except Exception as e:
        return ParsedLabel(
            ok=False,
            schema_type="invalid_json",
            clip_id=clip_id,
            video_id=clip_id,
            signer_id="",
            fps=None,
            n_frames=None,
            events_by_hand={"L": [], "R": []},
            parse_errors=[f"json_error: {e}"],
        )

    schema_type, raw_events, reason = _extract_schema_events(label)
    if raw_events is None:
        return ParsedLabel(
            ok=False,
            schema_type=schema_type,
            clip_id=clip_id,
            video_id=clip_id,
            signer_id="",
            fps=None,
            n_frames=None,
            events_by_hand={"L": [], "R": []},
            parse_errors=[reason],
        )

    video_meta = label.get("video") if isinstance(label.get("video"), dict) else {}
    video_id = str(video_meta.get("video_id", clip_id))
    signer_id = str(video_meta.get("signer_id", ""))
    fps = video_meta.get("fps")
    n_frames = video_meta.get("num_frames", video_meta.get("n_frames"))
    fps_val = float(fps) if fps is not None else None
    n_frames_val = _int_or_none(n_frames)

    parse_errors: List[str] = []
    out: Dict[str, List[ParsedEvent]] = {"L": [], "R": []}
    for hand in ("L", "R"):
        events = raw_events.get(hand, [])
        if not isinstance(events, list):
            parse_errors.append(f"hand {hand} events is not list")
            continue
        for i, ev in enumerate(events):
            if not isinstance(ev, dict):
                parse_errors.append(f"hand {hand} event_idx={i} not dict")
                continue
            parsed = _event_to_parsed(i, ev)
            if parsed is None:
                parse_errors.append(f"hand {hand} event_idx={i} missing span")
                continue
            out[hand].append(parsed)

    return ParsedLabel(
        ok=True,
        schema_type=schema_type,
        clip_id=clip_id,
        video_id=video_id,
        signer_id=signer_id,
        fps=fps_val,
        n_frames=n_frames_val,
        events_by_hand=out,
        parse_errors=parse_errors,
    )
