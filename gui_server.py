"""
gui_server.py — Flask 后端，接收视频 → 调 video2json → 返回 JSON
用法: python gui_server.py
"""

from __future__ import annotations

import copy
import csv
import html
import io
import json
import math
import subprocess
import sys
import threading
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__, static_folder=str(ROOT), static_url_path="/static")

# 任务状态存储
tasks = {}
ANNOTATION_DIR = ROOT / "output" / "annotations"
CORRECTIONS_FILE = ANNOTATION_DIR / "_corrections.json"
AUTHORS_FILE = ROOT / "AUTHORS.md"
annotation_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SPEC_LABELS = {
    "handshape": frozenset(
        "HA HBO HBY HCH HES HFF HJI HM HMQ HNG HNI HSE HNU HPJ HS HSA HSB HSC HSD HSF HSG HSH HSJ HSK HSL HSM HSO HSQ HSS HSU HSV HSW HSX HSY HWC HX HXX HYB HZH HSN HSP HST HTS HFY HLS HAS HSI HHE HT HLX HB HZ HFS HGG HW HC HTH HUN HMO NOH".split()
    ),
    "location": frozenset(
        "HT HTL HTR HD HDL HDR FC FL FR EYL EYR EBL EBR CL CR NOC LT LD MC CC NC SL SR TTL TTC TTR TML TMC TMR TBL TBC TBR WL WR ULL ULR UAL UAR WFL WFR WBL WBR FBL FBR FFL FFR LBW RBW LFW RFW H P PTO PTI PEO PEI PFO PFI PMO PHO PHI PIO PII PDO PDI PRO PRI PPO PPI OFP OFM PP PT MT MI MM MR MP FT FI FM RF PF NOL".split()
    ),
    "orientation": frozenset("TB TU TL TLU AFB TD TR TRU TLD TRD NOO".split()),
    "movement": frozenset(
        "CI ST CU HOLD NOM TMS TMC TMR TMH TMW TME TML TMA TMT TMN TMI".split()
    ),
    "facial": frozenset(
        "ebr ebf eow en cp nw mo mr mof lfr lr ltr mcf lv lpt ldb ls mct mcd tpc to tp th tl tc hts htf htb hf ht".split()
    ),
    "torso": frozenset("tf b ll llt lr lrt c s n bf".split()),
}
MANUAL_CHANNELS = ("handshape", "location", "orientation", "movement")
# 短于 0.1 秒的切段不显示，也不写入识别结果。
MIN_ANNOTATED_SEGMENT_MS = 100


def _validate_result(data):
    if not isinstance(data, list):
        raise ValueError("result 必须是事件数组")
    for event in data:
        if not isinstance(event, dict) or not isinstance(event.get("channels"), dict):
            raise ValueError("每个事件必须包含 channels 对象")
    return data


def _require_standard_manual_labels(result, annotation_meta):
    """人工改过的符号必须属于规范附录，模型原样输出不在这里拦截。"""
    for event in result:
        channels = event.get("channels") or {}
        for name in MANUAL_CHANNELS:
            channel = channels.get(name)
            if not isinstance(channel, dict):
                continue
            metadata = channel.get("metadata") or {}
            human = bool(metadata.get("human_modified")) or metadata.get("source") == "human_annotation"
            if not human:
                continue
            label = str((channel.get("prediction") or {}).get("label") or "").strip()
            if label not in SPEC_LABELS[name]:
                raise ValueError("输入错误")
    nmf = {}
    if isinstance(annotation_meta, dict):
        raw_nmf = annotation_meta.get("nmf") or {}
        if isinstance(raw_nmf, dict):
            nmf = raw_nmf
    for bucket, key in (("facial_events", "facial"), ("torso_events", "torso")):
        events = nmf.get(bucket) or []
        if not isinstance(events, list):
            raise ValueError("输入错误")
        for item in events:
            if not isinstance(item, dict):
                raise ValueError("输入错误")
            label = str(item.get("label") or "").strip()
            if label not in SPEC_LABELS[key]:
                raise ValueError("输入错误")


def _label_at(line: str, labels: tuple[str, ...]) -> int | None:
    lowered = line.lower()
    for label in labels:
        index = lowered.find(label)
        if index >= 0:
            return index
    return None


def load_credits(text: str | None = None) -> dict:
    """读取 AUTHORS.md。中文“导师/作者”和英文 Supervisor/Authors 都可以。"""
    if text is None:
        try:
            text = AUTHORS_FILE.read_text(encoding="utf-8")
        except OSError:
            text = ""
    advisor = ""
    authors = ""
    for raw_line in text.splitlines():
        line = raw_line.strip().replace("：", ":")
        if not line or line.startswith("#"):
            continue
        author_at = _label_at(line, ("作者:", "authors:"))
        if author_at is not None:
            left = line[:author_at]
            authors = line[author_at:].split(":", 1)[1].strip(" ,，")
            advisor_at = _label_at(left, ("导师:", "supervisor:"))
            if advisor_at is not None:
                advisor = left[advisor_at:].split(":", 1)[1].strip(" ,，")
            continue
        advisor_at = _label_at(line, ("导师:", "supervisor:"))
        if advisor_at is not None:
            advisor = line[advisor_at:].split(":", 1)[1].strip(" ,，")
    chinese_parts = []
    english_parts = []
    if advisor:
        chinese_parts.append(f"导师：{advisor}")
        english_parts.append(f"Supervisor: {advisor}")
    if authors:
        chinese_parts.append(f"作者：{authors}")
        english_parts.append(f"Authors: {authors}")
    return {
        "advisor": advisor,
        "authors": authors,
        "display": "，".join(chinese_parts),
        "display_en": ". ".join(english_parts),
    }


def _safe_annotation_name(name: str) -> str:
    safe_name = Path(name).name
    if (
        not name
        or safe_name != name
        or safe_name.startswith("_")
        or not safe_name.endswith(".json")
    ):
        raise ValueError("无效文件名")
    return safe_name


def _saved_annotation_records() -> list[dict]:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(
        ANNOTATION_DIR.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            "name": path.name,
            "task_id": payload.get("task_id"),
            "saved_at": payload.get("saved_at"),
            "video_name": payload.get("video_name"),
            "event_count": len(payload.get("result", [])),
        })
    return records


def _annotation_path(task_id: str) -> Path:
    safe_id = "".join(c for c in task_id if c.isalnum() or c in "-_")
    if not safe_id:
        raise ValueError("无效任务 ID")
    return ANNOTATION_DIR / f"{safe_id}.json"


def _correction_key(event, channel_name: str, channel) -> str:
    metadata = channel.get("metadata", {})
    signer_id = event.get("signer_id") or metadata.get("signer_id") or ""
    video_id = event.get("video_id") or metadata.get("video_id") or ""
    event_id = event.get("event_id") or channel.get("event_id") or ""
    hand = event.get("hand") or metadata.get("hand") or ""
    if not event_id:
        return ""
    return "|".join(
        [str(signer_id), str(video_id), str(event_id), str(hand).lower(), channel_name]
    )


def _load_corrections():
    if not CORRECTIONS_FILE.exists():
        return {}
    try:
        payload = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_corrections(corrections):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CORRECTIONS_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(corrections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(CORRECTIONS_FILE)


def _remember_corrections(result):
    """把本次人工标签写入按事件和通道索引的持久化存储。"""
    with annotation_lock:
        corrections = _load_corrections()
        for event in result:
            for channel_name, channel in event.get("channels", {}).items():
                metadata = channel.get("metadata", {})
                if not metadata.get("human_modified"):
                    continue
                key = _correction_key(event, channel_name, channel)
                label = metadata.get("human_label") or channel.get(
                    "prediction", {}
                ).get("label")
                if not key or not label:
                    continue
                corrections[key] = {
                    "label": label,
                    "updated_at": metadata.get("modified_at") or _utc_now(),
                }
        _save_corrections(corrections)


def _apply_corrections(result):
    """在新模型结果上叠加已保存人工标签，同时保留原始模型输出。"""
    with annotation_lock:
        corrections = _load_corrections()
    applied = 0
    for event in result:
        for channel_name, channel in event.get("channels", {}).items():
            correction = corrections.get(
                _correction_key(event, channel_name, channel)
            )
            if not correction:
                continue
            prediction = channel.setdefault("prediction", {})
            metadata = channel.setdefault("metadata", {})
            metadata["model_label"] = prediction.get("label")
            metadata["original_source"] = metadata.get("source")
            metadata["human_label"] = correction["label"]
            metadata["human_modified"] = True
            metadata["correction_applied"] = True
            metadata["modified_at"] = correction.get("updated_at")
            metadata["source"] = "human_annotation"
            prediction["label"] = correction["label"]
            channel["status"] = {"code": "OK", "message": "已应用历史人工标注"}
            applied += 1
    return applied


def _xml_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _set_attrs(element: ET.Element, mapping: dict) -> None:
    for key, value in mapping.items():
        if value is None or value == "":
            continue
        element.set(key, _xml_text(value))


def _normalize_hand_side(value) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("l"):
        return "Left"
    return "Right"


def _event_span(event: dict):
    segmentation = event.get("segmentation") or {}
    span = segmentation.get("span")
    if not span:
        span = event.get("span")
    if isinstance(span, dict):
        start = span.get("start_frame", span.get("start"))
        end = span.get("end_frame", span.get("end"))
    elif isinstance(span, (list, tuple)) and len(span) >= 2:
        start, end = span[0], span[1]
    else:
        for channel in (event.get("channels") or {}).values():
            meta_span = (channel.get("metadata") or {}).get("span")
            if isinstance(meta_span, dict):
                start = meta_span.get("start_frame", meta_span.get("start"))
                end = meta_span.get("end_frame", meta_span.get("end"))
                break
            if isinstance(meta_span, (list, tuple)) and len(meta_span) >= 2:
                start, end = meta_span[0], meta_span[1]
                break
        else:
            return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def _event_fps(event: dict, default: float = 30.0) -> float:
    segmentation = event.get("segmentation") or {}
    for value in (segmentation.get("fps"), event.get("fps"), default):
        try:
            fps = float(value)
            if fps > 0:
                return fps
        except (TypeError, ValueError):
            continue
    return default


def _frame_to_ms(frame: float, fps: float) -> int:
    return max(0, int(round(float(frame) * 1000.0 / fps)))


def _standard_event_interval(event: dict, fallback_index: int) -> tuple[int, int]:
    """Return the v3.0 exchange interval in integer milliseconds."""
    standard = event.get("standard") or {}
    try:
        start_ms = int(standard["start_ms"])
        end_ms = int(standard["end_ms"])
        if start_ms < end_ms:
            return start_ms, end_ms
    except (KeyError, TypeError, ValueError):
        pass
    span = _event_span(event)
    if span:
        fps = _event_fps(event)
        start_ms = _frame_to_ms(span[0], fps)
        end_ms = _frame_to_ms(span[1], fps)
        if start_ms < end_ms:
            return start_ms, end_ms
    return fallback_index * 1000, (fallback_index + 1) * 1000


def _movement_parts(label: str | None) -> tuple[str, str]:
    """规范把运动分成整体轨迹和局部运动。模型只给一个代码时按附录归类。"""
    code = str(label or "").strip().upper()
    if code in {"", "NOM"}:
        return "NOM", "NOM"
    if code == "HOLD":
        return "HOLD", "HOLD"
    if code.startswith("TM"):
        return "HOLD", code
    if code in {"CI", "ST", "CU"}:
        return code, "HOLD"
    return code, "HOLD"


def align_events_to_segments(
    events: list | None,
    dominant_hand: str = "right",
) -> list:
    """把左右手切到同一条时间轴上。

    同一只手的重叠区间改成前后相接的段，后出现的状态占用重叠部分。
    两只手落在同一段里，各自保留自己的标签。
    """
    prepared = []
    for index, event in enumerate(events or []):
        if not isinstance(event, dict):
            continue
        start_ms, end_ms = _standard_event_interval(event, index)
        if start_ms >= end_ms:
            continue
        hand = str(event.get("hand") or "").lower()
        if hand not in {"left", "right"}:
            continue
        prepared.append({
            "event": event,
            "start": start_ms,
            "end": end_ms,
            "hand": hand,
        })

    def resolve(hand_items: list[dict]) -> list[dict]:
        points = sorted({
            point
            for item in hand_items
            for point in (item["start"], item["end"])
        })
        pieces = []
        for left, right in zip(points, points[1:]):
            covering = [
                item for item in hand_items
                if item["start"] <= left and item["end"] >= right
            ]
            if not covering:
                continue
            covering.sort(key=lambda item: (-item["start"], item["end"] - item["start"]))
            winner = covering[0]
            if (
                pieces
                and pieces[-1]["event"] is winner["event"]
                and pieces[-1]["end"] == left
            ):
                pieces[-1]["end"] = right
            else:
                pieces.append({
                    "event": winner["event"],
                    "start": left,
                    "end": right,
                    "hand": winner["hand"],
                })
        return pieces

    resolved = resolve([item for item in prepared if item["hand"] == "left"])
    resolved.extend(resolve([item for item in prepared if item["hand"] == "right"]))
    points = sorted({
        point
        for item in resolved
        for point in (item["start"], item["end"])
    })
    slices = []
    for left, right in zip(points, points[1:]):
        active = [
            item for item in resolved
            if item["start"] <= left and item["end"] >= right
        ]
        if not active:
            continue
        signature = tuple(sorted(
            (item["hand"], id(item["event"])) for item in active
        ))
        if slices and slices[-1]["signature"] == signature and slices[-1]["end"] == left:
            slices[-1]["end"] = right
        else:
            slices.append({
                "start": left,
                "end": right,
                "signature": signature,
                "active": active,
            })

    dominant = str(dominant_hand or "right").lower()
    aligned = []
    for segment_index, segment in enumerate(slices, 1):
        segment_id = f"S{segment_index}"
        for item in sorted(segment["active"], key=lambda entry: entry["hand"]):
            event = copy.deepcopy(item["event"])
            standard = dict(event.get("standard") or {})
            side = item["hand"]
            role = standard.get("hand_role")
            if role not in {"H1", "H2"}:
                role = "H1" if side == dominant else "H2"
            source = (event.get("segmentation") or {}).get("source")
            default_status = "uncertain" if source == "unnumbered" else "observed"
            standard.update({
                "segment_id": segment_id,
                "start_ms": segment["start"],
                "end_ms": segment["end"],
                "hand_role": role,
                "participation": standard.get("participation") or "active",
                "annotation_status": standard.get("annotation_status") or default_status,
            })
            event["standard"] = standard
            aligned.append(event)
    return aligned


def _segment_too_short(start_ms: int, end_ms: int) -> bool:
    return end_ms - start_ms < MIN_ANNOTATED_SEGMENT_MS


def _blank_standard_hand() -> dict:
    return {
        "participation": "inactive",
        "handshape": {"value": ""},
        "location": {"value": "", "start": "", "end": ""},
        "orientation": {"value": ""},
        "movement": {"value": "", "path": "", "local": ""},
    }


def _standard_channel_value(event: dict, channel_name: str):
    channel = (event.get("channels") or {}).get(channel_name) or {}
    return (channel.get("prediction") or {}).get("label")


def _nmf_payload(annotation_meta: dict | None) -> dict:
    meta = annotation_meta if isinstance(annotation_meta, dict) else {}
    raw = meta.get("nmf") if isinstance(meta.get("nmf"), dict) else {}
    facial = raw.get("facial_events")
    torso = raw.get("torso_events")
    facial_events = facial if isinstance(facial, list) else []
    torso_events = torso if isinstance(torso, list) else []
    status = str(raw.get("status") or "").strip()
    if status not in {"observed", "observed_absent", "unobservable", "uncertain"}:
        status = "observed" if facial_events or torso_events else "observed_absent"
    return {
        "status": status,
        "facial_events": facial_events,
        "torso_events": torso_events,
    }


def result_to_standard_json(
    result: list,
    video_path: str | None = None,
    annotation_meta: dict | None = None,
) -> dict:
    """Map implemented channels to a v3.0-referenced project exchange structure."""
    meta = annotation_meta if isinstance(annotation_meta, dict) else {}
    dominant_hand = str(meta.get("dominant_hand") or "right").lower()
    record_id = (
        str(meta.get("record_id") or "").strip()
        or (Path(video_path).stem if video_path else "")
        or "CSL_RECORD"
    )
    result = align_events_to_segments(result, dominant_hand)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for index, event in enumerate(result or []):
        interval = _standard_event_interval(event, index)
        grouped.setdefault(interval, []).append(event)

    segments = []
    for segment_index, ((start_ms, end_ms), events) in enumerate(
        sorted(grouped.items()),
        1,
    ):
        segment_id = next(
            (
                str((event.get("standard") or {}).get("segment_id"))
                for event in events
                if (event.get("standard") or {}).get("segment_id")
            ),
            f"S{segment_index}",
        )
        hands = {
            role: {
                "participation": "inactive",
                "handshape": {"value": "NOH"},
                "location": {"value": "NOL", "start": "NOL", "end": "NOL"},
                "orientation": {"value": "NOO"},
                "movement": {"value": "NOM", "path": "NOM", "local": "NOM"},
            }
            for role in ("H1", "H2")
        }
        source_event_ids = []
        for event in events:
            standard = event.get("standard") or {}
            side = str(event.get("hand") or "").lower()
            role = standard.get("hand_role")
            if role not in {"H1", "H2"}:
                role = "H1" if side == dominant_hand else "H2"
            participation = standard.get("participation", "active")
            source_event_ids.append(event.get("event_id"))
            if participation == "inactive":
                continue
            status = standard.get("annotation_status", "observed")
            location = _standard_channel_value(event, "location") or "NOL"
            movement = _standard_channel_value(event, "movement") or "NOM"
            path, local = _movement_parts(movement)
            hands[role] = {
                "participation": "active",
                "physical_hand": side or None,
                "handshape": {
                    "value": _standard_channel_value(event, "handshape") or "NOH",
                    "status": status,
                },
                "location": {
                    "value": location,
                    "start": location,
                    "end": location,
                    "status": status,
                },
                "orientation": {
                    "value": _standard_channel_value(event, "orientation") or "NOO",
                    "status": status,
                },
                "movement": {
                    "value": movement,
                    "path": path,
                    "local": local,
                    "status": status,
                },
            }
        too_short = _segment_too_short(start_ms, end_ms)
        segments.append({
            "segment_id": segment_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "annotated": not too_short,
            "hands": {
                role: _blank_standard_hand() for role in ("H1", "H2")
            } if too_short else hands,
            "source_event_ids": [
                value for value in source_event_ids if value is not None
            ],
        })

    return {
        "schema_version": "1.0",
        "standard_version": "3.0",
        "annotation_scope": {
            "implemented": [
                "segment",
                "H1_H2",
                "handshape",
                "location",
                "orientation",
                "movement",
                "non_manual_features",
            ],
            "not_implemented": [
                "non_target_intervals",
                "gloss",
            ],
        },
        "record_id": record_id,
        "revision_version": str(meta.get("revision_version") or "1"),
        "created_at": meta.get("created_at") or _utc_now(),
        "modified_at": _utc_now(),
        "media": {
            "video": Path(video_path).name if video_path else None,
            "timeline_unit": "ms",
        },
        "dominant_hand": dominant_hand,
        "segments": segments,
        "nmf": _nmf_payload(meta),
        "review": {
            "status": meta.get("review_status") or "in_progress",
            "annotator": meta.get("annotator") or "",
            "reviewer": meta.get("reviewer") or "",
        },
    }


def _media_url(video_path: str | None) -> str | None:
    """生成 ELAN 能解析的本地 file URL，中文文件名保持原字，不做百分号编码。"""
    if not video_path:
        return None
    path = Path(video_path)
    try:
        path = path.resolve()
    except OSError:
        path = Path(video_path)
    posix = path.as_posix()
    if path.drive:
        return "file:///" + posix
    if posix.startswith("//"):
        return "file:" + posix
    if not posix.startswith("/"):
        posix = "/" + posix
    return "file://" + posix


def _relative_media_url(video_path: str | None) -> str | None:
    if not video_path:
        return None
    return "./" + Path(video_path).name


def _mime_type(video_path: str | None) -> str:
    suffix = Path(video_path or "").suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }.get(suffix, "video/mp4")


XSI = "http://www.w3.org/2001/XMLSchema-instance"
CHANNEL_TIER_NAMES = (
    ("handshape", "Handshape"),
    ("movement", "Movement"),
    ("location", "Location"),
    ("orientation", "Orientation"),
)
EAF_CONSTRAINTS = (
    (
        "Time_Subdivision",
        "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval",
    ),
    (
        "Symbolic_Subdivision",
        "Symbolic subdivision of a parent annotation. Annotations referring to the same parent are ordered",
    ),
    (
        "Symbolic_Association",
        "1-1 association with a parent annotation",
    ),
    (
        "Included_In",
        "Time alignable annotations within the parent annotation's time interval, gaps are allowed",
    ),
)


def _non_overlapping_annotations(items: list[dict]) -> list[dict]:
    """ELAN 拒绝同一层上时间重叠或零时长的标注，这里按时间把它们排开。"""
    ordered = sorted(
        items,
        key=lambda item: (item["start_ms"], item["end_ms"], item["index"]),
    )
    placed: list[dict] = []
    for item in ordered:
        start = max(0, int(item["start_ms"]))
        end = int(item["end_ms"])
        if placed and start < placed[-1]["end_ms"]:
            start = placed[-1]["end_ms"]
        if end <= start:
            end = start + 1
        updated = dict(item)
        updated["start_ms"] = start
        updated["end_ms"] = end
        placed.append(updated)
    return placed


def result_to_eaf(
    result,
    video_path: str | None = None,
    annotation_meta: dict | None = None,
) -> bytes:
    """把统一 JSON 结果转成兼容 ELAN 与 Simple-ELAN 的多层 EAF 3.0。"""
    source_result = result
    dominant = "right"
    if isinstance(annotation_meta, dict):
        dominant = str(annotation_meta.get("dominant_hand") or "right")
    result = align_events_to_segments(result, dominant)
    ET.register_namespace("xsi", XSI)
    root = ET.Element("ANNOTATION_DOCUMENT")
    root.set("AUTHOR", "SignPan")
    # ELAN 的日期解析不接受小数秒，必须是无小数的 xsd:dateTime。
    root.set(
        "DATE",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    )
    root.set("FORMAT", "3.0")
    root.set("VERSION", "3.0")
    root.set(
        f"{{{XSI}}}noNamespaceSchemaLocation",
        "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
    )

    header = ET.SubElement(root, "HEADER")
    header.set("MEDIA_FILE", "")
    header.set("TIME_UNITS", "milliseconds")
    media_url = _media_url(video_path)
    relative_media = _relative_media_url(video_path)
    if media_url:
        media = ET.SubElement(header, "MEDIA_DESCRIPTOR")
        media.set("MEDIA_URL", media_url)
        media.set("MIME_TYPE", _mime_type(video_path))
        if relative_media:
            media.set("RELATIVE_MEDIA_URL", relative_media)
        media.set("TIME_ORIGIN", "0")

    events: list[dict] = []
    cursor_ms = 0
    for index, event in enumerate(result or []):
        fps = _event_fps(event)
        span = _event_span(event)
        if event.get("standard"):
            start_ms, end_ms = _standard_event_interval(event, index)
        elif span:
            start_ms = _frame_to_ms(span[0], fps)
            end_ms = _frame_to_ms(span[1], fps)
            if end_ms <= start_ms:
                end_ms = start_ms + max(1, int(round(1000.0 / fps)))
        else:
            start_ms = cursor_ms
            end_ms = cursor_ms + 1000
            cursor_ms = end_ms
        event_id = event.get("event_id") or f"Event_{index + 1}"
        hand = _normalize_hand_side(event.get("hand") or event_id)
        events.append({
            "index": index,
            "event": event,
            "event_id": str(event_id),
            "hand": hand,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
    by_hand: dict[str, list[dict]] = {"Left": [], "Right": []}
    for item in events:
        by_hand.setdefault(item["hand"], []).append(item)
    events = []
    for hand in ("Left", "Right"):
        events.extend(_non_overlapping_annotations(by_hand.get(hand, [])))
    events.sort(key=lambda item: (
        item["start_ms"],
        item["end_ms"],
        item["hand"],
        item["index"],
    ))
    annotation_meta = (
        annotation_meta if isinstance(annotation_meta, dict) else {}
    )
    nmf_items = []
    for tier_name, event_key in (
        ("Facial", "facial_events"),
        ("Torso", "torso_events"),
    ):
        for index, raw in enumerate(_nmf_payload(annotation_meta)[event_key]):
            if not isinstance(raw, dict):
                continue
            try:
                start_ms = int(raw.get("start_ms"))
                end_ms = int(raw.get("end_ms"))
            except (TypeError, ValueError):
                continue
            label = str(raw.get("label") or "").strip()
            if end_ms <= start_ms or not label:
                continue
            nmf_items.append({
                "tier": tier_name,
                "label": label,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "index": index,
            })
    slot_ids = {
        ms: f"ts{index}"
        for index, ms in enumerate(
            sorted({
                ms
                for item in events
                for ms in (item["start_ms"], item["end_ms"])
            } | {
                ms
                for item in nmf_items
                for ms in (item["start_ms"], item["end_ms"])
            }),
            start=1,
        )
    }
    for item in events:
        item["start_ref"] = slot_ids[item["start_ms"]]
        item["end_ref"] = slot_ids[item["end_ms"]]
    for item in nmf_items:
        item["start_ref"] = slot_ids[item["start_ms"]]
        item["end_ref"] = slot_ids[item["end_ms"]]

    time_order = ET.SubElement(root, "TIME_ORDER")
    for ms in sorted(slot_ids):
        slot = ET.SubElement(time_order, "TIME_SLOT")
        slot.set("TIME_SLOT_ID", slot_ids[ms])
        slot.set("TIME_VALUE", str(ms))

    signer = next(
        (str(event.get("signer_id")) for event in (result or []) if event.get("signer_id")),
        "",
    )
    ann_index = 1

    def next_annotation_id() -> str:
        nonlocal ann_index
        annotation_id = f"a{ann_index}"
        ann_index += 1
        return annotation_id

    def create_tier(
        tier_id: str,
        linguistic_type: str,
        *,
        parent_ref: str | None = None,
    ) -> ET.Element:
        tier = ET.SubElement(root, "TIER")
        tier.set("TIER_ID", tier_id)
        tier.set("LINGUISTIC_TYPE_REF", linguistic_type)
        tier.set("ANNOTATOR", "SignPan")
        tier.set("DEFAULT_LOCALE", "zh")
        tier.set("LANG_REF", "sgn")
        if signer:
            tier.set("PARTICIPANT", signer)
        if parent_ref:
            tier.set("PARENT_REF", parent_ref)
        return tier

    # Utterance 表示整句时间范围。其余层保持独立时间对齐，以兼容
    # Simple-ELAN（该软件不显示 REF_ANNOTATION / PARENT_REF 子层）。
    timed_items = events + nmf_items
    utterance_tier = create_tier("Utterance", "utterance")
    if timed_items:
        utterance_wrapper = ET.SubElement(utterance_tier, "ANNOTATION")
        utterance = ET.SubElement(utterance_wrapper, "ALIGNABLE_ANNOTATION")
        utterance.set("ANNOTATION_ID", next_annotation_id())
        utterance.set(
            "TIME_SLOT_REF1",
            slot_ids[min(item["start_ms"] for item in timed_items)],
        )
        utterance.set(
            "TIME_SLOT_REF2",
            slot_ids[max(item["end_ms"] for item in timed_items)],
        )
        utterance_value = Path(video_path).stem if video_path else "Utterance_1"
        ET.SubElement(utterance, "ANNOTATION_VALUE").text = utterance_value

    event_tiers: dict[str, ET.Element] = {}
    feature_tiers: dict[tuple[str, str], ET.Element] = {}
    for hand in ("Left", "Right"):
        event_tier_id = f"Event@{hand}"
        event_tiers[hand] = create_tier(
            event_tier_id,
            "event",
        )
        for channel_key, tier_name in CHANNEL_TIER_NAMES:
            feature_tiers[(hand, channel_key)] = create_tier(
                f"{tier_name}@{hand}",
                "phonological_feature",
            )
    nmf_tiers = {}
    for tier_name in ("Facial", "Torso"):
        if any(item["tier"] == tier_name for item in nmf_items):
            nmf_tiers[tier_name] = create_tier(tier_name, "phonological_feature")

    # 事件本身与媒体时间轴对齐。
    for item in events:
        if _segment_too_short(item["start_ms"], item["end_ms"]):
            continue
        wrapper = ET.SubElement(event_tiers[item["hand"]], "ANNOTATION")
        alignable = ET.SubElement(wrapper, "ALIGNABLE_ANNOTATION")
        annotation_id = next_annotation_id()
        alignable.set("ANNOTATION_ID", annotation_id)
        alignable.set("TIME_SLOT_REF1", item["start_ref"])
        alignable.set("TIME_SLOT_REF2", item["end_ref"])
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = item["event_id"]

    # 四个音系层均直接绑定同一组时间槽。这样每一层都能被官方 ELAN 和
    # Simple-ELAN 识别、选择、编辑并在时间轴上同步显示。
    for item in events:
        if _segment_too_short(item["start_ms"], item["end_ms"]):
            continue
        channels = item["event"].get("channels") or {}
        for channel_key, _ in CHANNEL_TIER_NAMES:
            channel = channels.get(channel_key) or {}
            label = (channel.get("prediction") or {}).get("label")
            if label in (None, "", "?"):
                continue
            wrapper = ET.SubElement(
                feature_tiers[(item["hand"], channel_key)],
                "ANNOTATION",
            )
            alignable = ET.SubElement(wrapper, "ALIGNABLE_ANNOTATION")
            alignable.set("ANNOTATION_ID", next_annotation_id())
            alignable.set("TIME_SLOT_REF1", item["start_ref"])
            alignable.set("TIME_SLOT_REF2", item["end_ref"])
            ET.SubElement(alignable, "ANNOTATION_VALUE").text = str(label)

    for item in nmf_items:
        wrapper = ET.SubElement(nmf_tiers[item["tier"]], "ANNOTATION")
        alignable = ET.SubElement(wrapper, "ALIGNABLE_ANNOTATION")
        alignable.set("ANNOTATION_ID", next_annotation_id())
        alignable.set("TIME_SLOT_REF1", item["start_ref"])
        alignable.set("TIME_SLOT_REF2", item["end_ref"])
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = item["label"]

    last_used = ET.SubElement(header, "PROPERTY")
    last_used.set("NAME", "lastUsedAnnotationId")
    last_used.text = str(max(ann_index - 1, 0))

    for name, value in (
        ("SignPan.schema", "sign_phonology_v1"),
        ("SignPan.exporter", "SignPanv3.0"),
        (
            "SignPan.sourceJSON",
            json.dumps(source_result or [], ensure_ascii=False, separators=(",", ":")),
        ),
        (
            "SignPan.standardJSON",
            json.dumps(
                result_to_standard_json(
                    result or [],
                    video_path=video_path,
                    annotation_meta=annotation_meta,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
        ("SignPan.standardVersion", "3.0"),
        (
            "SignPan.tierHierarchy",
            "Utterance > Event@Left|Event@Right > "
            "Handshape|Movement|Location|Orientation; Facial; Torso",
        ),
    ):
        prop = ET.SubElement(header, "PROPERTY")
        prop.set("NAME", name)
        prop.text = value

    for type_id in ("utterance", "event", "phonological_feature"):
        linguistic_type = ET.SubElement(root, "LINGUISTIC_TYPE")
        linguistic_type.set("GRAPHIC_REFERENCES", "false")
        linguistic_type.set("LINGUISTIC_TYPE_ID", type_id)
        linguistic_type.set("TIME_ALIGNABLE", "true")

    locale = ET.SubElement(root, "LOCALE")
    locale.set("LANGUAGE_CODE", "zh")
    locale.set("COUNTRY_CODE", "CN")

    language = ET.SubElement(root, "LANGUAGE")
    language.set("LANG_ID", "sgn")
    language.set("LANG_DEF", "https://iso639-3.sil.org/code/sgn")
    language.set("LANG_LABEL", "Sign languages")

    for stereotype, description in EAF_CONSTRAINTS:
        constraint = ET.SubElement(root, "CONSTRAINT")
        constraint.set("DESCRIPTION", description)
        constraint.set("STEREOTYPE", stereotype)

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def result_to_xml(result, video_path: str | None = None) -> bytes:
    """兼容旧接口：XML 导出即为 EAF。"""
    return result_to_eaf(result, video_path=video_path)


def run_prediction(task_id: str, video_path: str):
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10

        # 清理旧缓存（用绝对路径）
        cache_patterns = [
            ROOT / "handshape_channel_data28" / "cache" / "keypoints" / "*.npz",
            ROOT / "location_channel" / "data" / "features" / "*.npz",
            ROOT / "orientation_channel" / "data" / "precomputed" / "*.npz",
            ROOT / "movement_channel" / "data" / "features" / "*.npz",
        ]
        for pattern in cache_patterns:
            for f in pattern.parent.glob("*.npz"):
                f.unlink()

        tasks[task_id]["progress"] = 30

        # 调用 video2json.py（关键：用 -m 方式调用，确保路径正确）
        segment_mode = tasks[task_id].get("segment_mode", "auto")
        pred_json = ROOT / "output" / "predictions.json"
        if pred_json.exists():
            pred_json.unlink()
        log_path = ROOT / "output" / "predict.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "video2json",
                    "--video",
                    video_path,
                    "--run_all",
                    "--segment_mode",
                    segment_mode,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                cwd=str(ROOT),
                env={
                    **__import__('os').environ,
                    "PYTHONPATH": str(ROOT),
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            )
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")

        tasks[task_id]["progress"] = 90

        # MediaPipe 在结果写完后，进程退出码仍可能不是 0。有 JSON 就采用。
        if not pred_json.exists():
            tasks[task_id]["status"] = "error"
            text = log_text.strip()
            if "未找到任何 event" in text or "未切分出有效" in text:
                tasks[task_id]["error"] = (
                    "未从该视频中切分出有效手语事件。"
                    "请确认画面中的手部清晰可见，或换一段手语更完整的视频。"
                )
            else:
                tasks[task_id]["error"] = "预测没有生成结果。\n" + text[-500:]
            return

        with open(pred_json, encoding="utf-8") as f:
            data = json.load(f)

        if not data or len(data) == 0:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = (
                "未从该视频中切分出有效手语事件。"
                "请使用 BoundaryTCN 自动切分，并确认画面中的手部清晰可见。"
            )
            return

        _apply_corrections(data)
        data = align_events_to_segments(data)

        tasks[task_id]["status"] = "done"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["result"] = data

    except subprocess.TimeoutExpired:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = "处理超时（>900秒）"
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)

@app.route("/")
def index():
    """返回前端页面"""
    return send_from_directory(str(ROOT), "gui.html")


@app.route("/en")
def index_english():
    """Return the English interface."""
    return send_from_directory(str(ROOT), "gui_en.html")


@app.route("/gui.css")
def serve_gui_css():
    """Shared stylesheet for the Chinese and English interfaces."""
    return send_from_directory(str(ROOT), "gui.css")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    original_filename = Path(file.filename).name
    ext = Path(original_filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi"}:
        return jsonify({"error": "仅支持 .mp4、.mov、.avi 视频文件"}), 400

    original_name = Path(original_filename).stem
    if not original_name:
        return jsonify({"error": "文件名为空"}), 400

    # 保存上传的视频。事件身份不再依赖固定文件名；普通名称会在
    # video2json 中映射为稳定的合成 signer/video ID。
    task_id = str(uuid.uuid4())[:8]
    upload_dir = ROOT / "output" / "uploads" / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / f"{original_name}{ext}"

    file.save(str(video_path))
    segment_mode = request.form.get("segment_mode", "auto")
    if segment_mode not in {"auto", "model", "annotation"}:
        return jsonify({"error": "无效切分模式"}), 400

    # 初始化任务
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "video_path": str(video_path),
        "segment_mode": segment_mode,
        "annotation_meta": {},
        "result": None,
        "error": None,
    }

    thread = threading.Thread(target=run_prediction, args=(task_id, str(video_path)), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/status/<task_id>")
def status(task_id: str):
    """查询任务进度"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404

    task = tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
    }

    if task["status"] == "done":
        response["result"] = task["result"]
        response["annotation_meta"] = task.get("annotation_meta", {})
    elif task["status"] == "error":
        response["error"] = task["error"]

    return jsonify(response)


@app.route("/api/annotations", methods=["GET"])
def list_annotations():
    """分页列出已保存的人工标注会话。"""
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 8))
    except (TypeError, ValueError):
        return jsonify({"error": "页码无效"}), 400
    page_size = min(max(page_size, 1), 50)
    records = _saved_annotation_records()
    total = len(records)
    pages = max(1, math.ceil(total / page_size))
    page = min(max(page, 1), pages)
    start = (page - 1) * page_size
    return jsonify({
        "items": records[start:start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    })


@app.route("/api/annotations/<name>", methods=["GET"])
def load_annotation(name: str):
    """加载历史标注，并创建可继续编辑的临时任务。"""
    try:
        safe_name = _safe_annotation_name(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    path = ANNOTATION_DIR / safe_name
    if not path.exists():
        return jsonify({"error": "标注不存在"}), 404
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = _validate_result(payload.get("result"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    task_id = f"history-{uuid.uuid4().hex[:8]}"
    tasks[task_id] = {
        "status": "done",
        "progress": 100,
        "video_path": payload.get("video_path"),
        "annotation_meta": payload.get("annotation_meta") or {},
        "result": result,
        "error": None,
    }
    return jsonify({
        "task_id": task_id,
        "result": result,
        "annotation_meta": payload.get("annotation_meta") or {},
        "metadata": payload,
    })


@app.route("/api/annotations/<name>", methods=["DELETE"])
def delete_annotation(name: str):
    """删除一条已保存的历史标注。不删除跨任务修订索引。"""
    try:
        safe_name = _safe_annotation_name(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    path = ANNOTATION_DIR / safe_name
    with annotation_lock:
        if not path.is_file():
            return jsonify({"error": "标注不存在"}), 404
        path.unlink()
    return jsonify({"ok": True, "name": safe_name})


def _render_about(language: str) -> str:
    credits = load_credits()
    if language == "en":
        display = credits["display_en"] or "Supervisor: (not set). Authors: (not set)"
    else:
        display = credits["display"] or "导师：（未填写），作者：（未填写）"
    escaped_display = html.escape(display)
    if language == "en":
        body = f"""
        <p class="kicker">SignPanv3.0</p>
        <h1>About</h1>
        <p>SignPanv3.0, the Sign Language Phonological Parameter Recognition and Automatic Annotation System, is a research system for sign-language phonology recognition and human annotation. Given a signing video, it separates the left and right hands into events, predicts four phonological features for each event, and leaves the final record to a person who can check the video and correct the labels.</p>
        <p>This version is a working research prototype. Event boundaries come from BoundaryTCN. When a channel does not have a model result for the current sample, SignPan uses a labeled fallback and asks for human review. A fallback is not a neural-network prediction for that video.</p>
        <h2>What the four channels describe</h2>
        <ul>
            <li><strong>Handshape.</strong> The shape formed by the palm and fingers.</li>
            <li><strong>Location.</strong> Where that hand is relative to the body or the other hand.</li>
            <li><strong>Orientation.</strong> The general direction the palm faces.</li>
            <li><strong>Movement.</strong> The type of movement inside the event.</li>
        </ul>
        <p>The two hands are annotated separately. In the utterance view, H1 is the dominant hand and H2 is the other hand. The dominant hand defaults to the right hand and can be changed in the annotation record.</p>
        <h2>How a video is segmented</h2>
        <ul>
            <li><strong>Automatic.</strong> BoundaryTCN runs first. If it finds no events or fails, SignPan uses existing annotation boundaries.</li>
            <li><strong>Model only.</strong> BoundaryTCN is the only source of boundaries.</li>
            <li><strong>Annotation boundaries only.</strong> Events come from the annotation file. A new video that does not match that file will not produce events in this mode.</li>
        </ul>
        <p>Accepted video files are .mp4, .mov, and .avi. The file name does not have to follow the original corpus pattern. Names from the original corpus keep their signer and video identity; other names receive a stable generated identity.</p>
        <h2>What you can do in the interface</h2>
        <ol>
            <li>Start the local server and open the Chinese page or the English page.</li>
            <li>Drop in a video, choose a segmentation mode, and start prediction.</li>
            <li>Wait through segmentation. The progress bar stays near the preprocessing step while both hands are scanned frame by frame, so a long video takes several minutes.</li>
            <li>Move the playhead. The utterance sequence and the annotation table show only the segment under the playhead. Outside every segment, those two areas stay empty.</li>
            <li>Correct a label, the start or end time, participation, or observation status, then save.</li>
            <li>Open an earlier save from the history list, or delete one saved record. History is shown eight items at a time.</li>
        </ol>
        <p>Saved sessions are stored under <code>output/annotations/</code>. If the same signer, video, event, hand, and channel appear again, the previous human label is restored. Deleting one history entry removes that saved file. It does not erase the correction index used for later restoration.</p>
        <h2>Export</h2>
        <ul>
            <li><strong>Standard v3.0 JSON.</strong> One record organized by segment, with H1 and H2 values for the four channels.</li>
            <li><strong>CSV.</strong> One row per channel, for checking labels in a spreadsheet.</li>
            <li><strong>EAF.</strong> An ELAN 3.0 file. Every tier is aligned directly to the timeline, so both Simple-ELAN and ELAN can open it. Keep the EAF beside the video; otherwise the program may report a missing media file while the annotations themselves remain intact.</li>
        </ul>
        <h2>Credits</h2>
        <p>The supervisor and authors of this version are:</p>
        <p class="credit-line">{escaped_display}</p>
        <h2>Copyright</h2>
        <p>The copyright holders of the SignPanv3.0 program are the people named above. Until an open-source license is stated, all rights in the program, interface, and documentation are reserved.</p>
        <p>This notice does not license the sign-language videos, annotations, derived data, model checkpoints, training logs, or files produced in <code>output/</code>. Python, Flask, PyTorch, OpenCV, MediaPipe, and the ELAN EAF 3.0 format remain under their own licenses. Using the EAF format does not imply an affiliation with the Max Planck Institute.</p>
        <p><a href="/">中文界面</a> · <a href="/en">English interface</a> · <a href="/about">中文关于页</a></p>
        """
        page_title = "About SignPan"
    else:
        body = f"""
        <p class="kicker">SignPanv3.0</p>
        <h1>关于</h1>
        <p>SignPanv3.0 是手语音系学参数识别与自动标注系统。上传一段手语视频后，它先把连续动作切成左右手事件，再为每个事件预测手形、位置、手掌朝向和运动。标注员对照视频核对这些结果，修改标签和时间，保存后可以继续导出。</p>
        <p>当前版本是可运行的研究原型。事件切分使用 BoundaryTCN。某个通道如果没有当前样本的模型结果，系统会改用已经标明来源的回退结果，并提示人工复核。回退结果不是这段视频的神经网络推理，不能按模型预测来引用。</p>
        <h2>四个通道分别表示什么</h2>
        <ul>
            <li><strong>手形 Handshape。</strong>手掌和手指构成的形状。</li>
            <li><strong>位置 Location。</strong>这只手相对身体或另一只手所在的位置。</li>
            <li><strong>手掌朝向 Orientation。</strong>掌心大致朝向哪个方向。</li>
            <li><strong>运动 Movement。</strong>这个事件里手部运动的类型。</li>
        </ul>
        <p>左右手分开标注。整句序列里的 H1 是优势手，H2 是另一只手。优势手默认是右手，可以在标注记录里修改。</p>
        <h2>视频怎样切分</h2>
        <ul>
            <li><strong>自动。</strong>先用 BoundaryTCN。没有切出事件或切分失败时，再使用已有标注边界。</li>
            <li><strong>仅模型。</strong>只使用 BoundaryTCN。</li>
            <li><strong>仅标注边界。</strong>只使用标注库中的事件边界。新视频如果对不上标注库，这种模式不会产生事件。</li>
        </ul>
        <p>视频格式可以是 .mp4、.mov 或 .avi，文件名不要求沿用原始语料的写法。原始语料的命名会保留原来的打手和视频身份；其他文件名会得到一个稳定的合成身份。</p>
        <h2>标注界面可以做什么</h2>
        <ol>
            <li>在本机启动服务，打开中文页或英文页。</li>
            <li>放入视频，选择切分方式，开始预测。</li>
            <li>等待切分。进度会在预处理附近停留一段时间，因为系统要逐帧查看左右手。视频较长时，这一步会到几分钟。</li>
            <li>拖动播放进度。整句标注序列和标注层只显示播放头所在的切分段。播放头不在任何切分段内时，这两处不显示标注内容。</li>
            <li>修改标签、起止时间、参与状态或观察状态，然后保存。</li>
            <li>从历史标注中打开以前的记录，或删除某一条。历史记录每页显示 8 条。</li>
        </ol>
        <p>保存的标注写在 <code>output/annotations/</code>。同一打手、视频、事件、手和通道再次出现时，系统会自动带回上次的人工标签。删除某一条历史记录，删掉的是那一份保存文件，不会清空用于以后自动恢复的修订索引。</p>
        <h2>可以导出什么</h2>
        <ul>
            <li><strong>规范 v3.0 JSON。</strong>按切分段整理，分别给出 H1 和 H2 的四层标注。</li>
            <li><strong>CSV。</strong>每个通道一行，方便在表格里检查。</li>
            <li><strong>EAF。</strong>ELAN 3.0 文件。每一层都直接对齐时间，Simple-ELAN 和 ELAN 都可以打开。请把 EAF 和视频放在同一目录；如果媒体没有放在一起，软件可能提示找不到视频，标注本身仍然在文件里。</li>
        </ul>
        <h2>作者名单</h2>
        <p>本版本的导师和作者是：</p>
        <p class="credit-line">{escaped_display}</p>
        <h2>版权声明</h2>
        <p>SignPanv3.0 程序、界面和文档的著作权人，以上面的作者名单为准。在写明开源许可证之前，这些内容保留全部权利。</p>
        <p>手语视频、标注、派生数据、模型权重、训练日志，以及 <code>output/</code> 里的运行结果和上传视频，不因这段声明而获得再分发许可。Python、Flask、PyTorch、OpenCV、MediaPipe 和 ELAN 的 EAF 3.0 格式仍使用它们各自的许可证。使用 EAF 格式并不表示本项目与 Max Planck Institute 存在隶属关系。</p>
        <p><a href="/">返回标注界面</a> · <a href="/en">English</a> · <a href="/about/en">English about</a></p>
        """
        page_title = "关于 SignPan"
    return f"""<!DOCTYPE html>
<html lang="{'en' if language == 'en' else 'zh-CN'}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page_title}</title>
<link rel="stylesheet" href="/static/gui.css">
</head>
<body>
<main class="about-page">
{body}
</main>
</body>
</html>
"""


@app.route("/about")
def about():
    return _render_about("zh")


@app.route("/about/en")
def about_english():
    return _render_about("en")


@app.route("/save/<task_id>", methods=["POST"])
def save_annotation(task_id: str):
    """保存人工修改后的完整标注结果。"""
    payload = request.get_json(silent=True) or {}
    try:
        result = _validate_result(payload.get("result"))
        annotation_meta = payload.get("annotation_meta")
        if not isinstance(annotation_meta, dict):
            annotation_meta = {}
        _require_standard_manual_labels(result, annotation_meta)
        path = _annotation_path(task_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    task = tasks.setdefault(task_id, {
        "status": "done",
        "progress": 100,
        "video_path": None,
        "annotation_meta": {},
        "error": None,
    })
    task["result"] = result
    task["annotation_meta"] = annotation_meta
    task["status"] = "done"
    _remember_corrections(result)

    video_path = task.get("video_path")
    saved = {
        "schema": "sign_annotation_session_v1",
        "task_id": task_id,
        "saved_at": _utc_now(),
        "video_path": video_path,
        "video_name": Path(video_path).name if video_path else None,
        "annotation_meta": annotation_meta,
        "result": result,
    }
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return jsonify({"ok": True, "path": str(path), "saved_at": saved["saved_at"]})


@app.route("/export/<task_id>", methods=["GET"])
def export_annotation(task_id: str):
    """导出当前标注为参考 v3.0 的项目 JSON、CSV 或 EAF。"""
    task = tasks.get(task_id)
    if not task or not isinstance(task.get("result"), list):
        return jsonify({"error": "任务结果不存在，请先保存"}), 404
    result = task["result"]
    annotation_meta = task.get("annotation_meta") or {}
    export_format = request.args.get("format", "standard_json").lower()

    if export_format in {"standard_json", "standard-json"}:
        standard = result_to_standard_json(
            result,
            video_path=task.get("video_path"),
            annotation_meta=annotation_meta,
        )
        content = json.dumps(
            standard,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        return send_file(
            io.BytesIO(content),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"SignPanv3.0_{standard['record_id']}.json",
        )
    if export_format in {"xml", "eaf"}:
        video_path = task.get("video_path")
        content = result_to_eaf(
            result,
            video_path=video_path,
            annotation_meta=annotation_meta,
        )
        download_name = f"SignPanv3.0_{task_id}.eaf"
        if video_path:
            video = Path(video_path)
            download_name = f"SignPanv3.0_{video.stem}.eaf"
            sidecar = video.with_suffix(".eaf")
            try:
                sidecar.write_bytes(content)
            except OSError:
                pass
        return send_file(
            io.BytesIO(content),
            mimetype="application/xml",
            as_attachment=True,
            download_name=download_name,
        )
    if export_format != "csv":
        return jsonify({
            "error": "仅支持 standard_json、csv 或 eaf"
        }), 400

    stream = io.StringIO()
    fields = [
        "sample_id", "event_id", "signer_id", "video_id", "hand",
        "channel", "label", "model_label", "confidence", "status",
        "human_modified", "source",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for event_index, event in enumerate(result):
        if event.get("standard"):
            start_ms, end_ms = _standard_event_interval(event, event_index)
            if _segment_too_short(start_ms, end_ms):
                continue
        for channel_name, channel in event.get("channels", {}).items():
            prediction = channel.get("prediction", {})
            metadata = channel.get("metadata", {})
            writer.writerow({
                "sample_id": event.get("sample_id") or channel.get("sample_id"),
                "event_id": event.get("event_id") or channel.get("event_id"),
                "signer_id": event.get("signer_id") or metadata.get("signer_id"),
                "video_id": event.get("video_id") or metadata.get("video_id"),
                "hand": event.get("hand") or metadata.get("hand"),
                "channel": channel_name,
                "label": prediction.get("label"),
                "model_label": metadata.get("model_label", prediction.get("label")),
                "confidence": prediction.get("conf"),
                "status": channel.get("status", {}).get("code"),
                "human_modified": metadata.get("human_modified", False),
                "source": metadata.get("source"),
            })
    content = ("\ufeff" + stream.getvalue()).encode("utf-8")
    return send_file(
        io.BytesIO(content),
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=f"SignPanv3.0_{task_id}.csv",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """直接传视频路径（不用上传，适合服务器上已有视频）"""
    data = request.json
    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "视频路径无效"}), 400
    segment_mode = data.get("segment_mode", "auto")
    if segment_mode not in {"auto", "model", "annotation"}:
        return jsonify({"error": "无效切分模式"}), 400

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "video_path": video_path,
        "segment_mode": segment_mode,
        "annotation_meta": {},
        "result": None,
        "error": None,
    }

    thread = threading.Thread(target=run_prediction, args=(task_id, video_path), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


def repredict_events(
    video_path: str,
    events: list[dict],
    target_indices: list[int] | None = None,
    dominant_hand: str = "right",
    overwrite_human: bool = False,
) -> tuple[list[dict], int]:
    """对手动新增、拆分或调整过边界的切分段局部重算四通道特征与预测。"""
    from run_all import run_all
    from video2json import _parse_video_uid, deploy_to_channels, process_event

    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError("当前会话未关联本地视频文件，无法重算模型特征；仍可直接手工标注")

    _, default_signer, default_video = _parse_video_uid(video_file)
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = copy.deepcopy(events)
    if target_indices is None:
        indices = list(range(len(updated)))
    else:
        indices = [i for i in target_indices if 0 <= i < len(updated)]

    repredicted = 0
    for idx in indices:
        event = updated[idx]
        if not isinstance(event, dict):
            continue
        fps = _event_fps(event, 30.0)
        start_ms, end_ms = _standard_event_interval(event, idx)
        if end_ms <= start_ms:
            continue
        start_frame = max(0, int(round(start_ms * fps / 1000.0)))
        end_frame = max(start_frame + 1, int(round(end_ms * fps / 1000.0)))
        span = {"start_frame": start_frame, "end_frame": end_frame}

        hand = str(event.get("hand") or "right").lower()
        if hand not in {"left", "right"}:
            hand = "right"
        prefix = "L" if hand == "left" else "R"
        signer_id = str(event.get("signer_id") or default_signer)
        video_id = str(event.get("video_id") or default_video)
        original_event_id = str(event.get("event_id") or f"{prefix}_Event_{idx + 1}")
        infer_event_id = f"{prefix}_Manual_{idx + 1}_{start_frame}_{end_frame}"

        ev_req = {
            "event_id": infer_event_id,
            "hand": hand,
            "span": span,
            "signer_id": signer_id,
            "video_id": video_id,
        }
        paths = process_event(video_file, ev_req, out_dir, fast_body=None)
        deploy_to_channels(
            paths["handshape"],
            paths["location"],
            paths["orientation"],
            paths["movement"],
        )
        infer_sample_id = f"{signer_id}_{video_id}_{infer_event_id}_{hand.upper()}"
        fresh = run_all(
            split="test",
            sample_id=infer_sample_id,
            event_id=infer_event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
            span=span,
            video_path=str(video_file),
            allow_unnumbered_inference=True,
            channel_overrides={
                "orientation": paths["orientation_prediction"],
                "movement": paths["movement_prediction"],
            },
        )
        old_channels = event.get("channels") or {}
        new_channels = fresh.get("channels") or {}
        merged_channels = {}
        for ch_name in MANUAL_CHANNELS:
            old_ch = old_channels.get(ch_name) or {}
            new_ch = copy.deepcopy(new_channels.get(ch_name) or {})
            old_meta = old_ch.get("metadata") or {}
            if not overwrite_human and bool(old_meta.get("human_modified")):
                merged_channels[ch_name] = old_ch
                continue
            new_ch["event_id"] = original_event_id
            new_ch["sample_id"] = f"{signer_id}_{video_id}_{original_event_id}_{hand.upper()}"
            merged_channels[ch_name] = new_ch

        event["signer_id"] = signer_id
        event["video_id"] = video_id
        event["event_id"] = original_event_id
        event["sample_id"] = f"{signer_id}_{video_id}_{original_event_id}_{hand.upper()}"
        event["hand"] = hand
        event["channels"] = merged_channels
        segmentation = dict(event.get("segmentation") or {})
        segmentation.update({
            "source": "manual_repredicted",
            "span": span,
            "fps": fps,
        })
        event["segmentation"] = segmentation
        repredicted += 1

    aligned = align_events_to_segments(updated, dominant_hand)
    return aligned, repredicted


@app.route("/api/repredict/<task_id>", methods=["POST"])
def api_repredict(task_id: str):
    """对手动调整或新增的切分段按需重算四通道预测。"""
    payload = request.get_json(silent=True) or {}
    try:
        result = _validate_result(payload.get("result"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    task = tasks.get(task_id)
    video_path = (task or {}).get("video_path") or payload.get("video_path")
    if not video_path or not Path(video_path).is_file():
        return jsonify({
            "error": "当前任务没有可用的本地视频文件，无法重算模型预测；仍可直接手工标注"
        }), 400

    annotation_meta = payload.get("annotation_meta")
    if not isinstance(annotation_meta, dict):
        annotation_meta = (task or {}).get("annotation_meta") or {}
    dominant_hand = str(annotation_meta.get("dominant_hand") or "right").lower()

    segment_id = payload.get("segment_id")
    event_indices = payload.get("event_indices")
    if isinstance(event_indices, list):
        target_indices = [int(i) for i in event_indices if isinstance(i, int)]
    elif segment_id:
        target_indices = [
            idx
            for idx, ev in enumerate(result)
            if (ev.get("standard") or {}).get("segment_id") == segment_id
        ]
    else:
        target_indices = list(range(len(result)))

    try:
        updated, count = repredict_events(
            str(video_path),
            result,
            target_indices=target_indices,
            dominant_hand=dominant_hand,
            overwrite_human=bool(payload.get("overwrite_human", False)),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if task is not None:
        task["result"] = updated
        task["annotation_meta"] = annotation_meta
    return jsonify({"ok": True, "result": updated, "repredicted": count})


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)