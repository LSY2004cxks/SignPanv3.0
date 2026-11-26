import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema_parser import ParsedEvent, ParsedLabel, parse_label_file

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class ClipRecord:
    clip_id: str
    signer_id: str
    schema_type: str
    video_id: str
    label_path: str
    video_path: Optional[str]
    pose_path: Optional[str]
    fps: Optional[float]
    n_frames: Optional[int]
    events_by_hand: Dict[str, List[ParsedEvent]]
    parse_errors: List[str]
    match_video: str
    match_pose: str


@dataclass
class Manifest:
    records: List[ClipRecord]
    schema_report: Dict


def _infer_signer_from_path(path: str) -> Optional[str]:
    p = Path(path)
    for part in p.parts:
        if re.fullmatch(r"S\d+", part):
            return part
    return None


def _list_files(root: str, suffixes: Optional[set] = None, name_suffix: Optional[str] = None) -> List[str]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    out: List[str] = []
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        if suffixes is not None and p.suffix.lower() not in suffixes:
            continue
        if name_suffix is not None and not p.name.endswith(name_suffix):
            continue
        out.append(str(p))
    return sorted(out)


def _stem(path: str) -> str:
    return Path(path).stem


def _id_tokens(stem: str) -> List[str]:
    tokens = set()
    s = stem.lower()
    tokens.add(s)
    m = re.fullmatch(r"v(\d+)", s)
    if m:
        tokens.add(m.group(1))
    nums = re.findall(r"\d+", s)
    for n in nums:
        tokens.add(n)
    return sorted(tokens)


def _build_maps(paths: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    exact: Dict[str, List[str]] = {}
    by_token: Dict[str, List[str]] = {}
    exact_by_signer: Dict[str, Dict[str, List[str]]] = {}
    token_by_signer: Dict[str, Dict[str, List[str]]] = {}
    for p in paths:
        st = _stem(p).lower()
        signer = _infer_signer_from_path(p) or ""
        exact.setdefault(st, []).append(p)
        exact_by_signer.setdefault(signer, {}).setdefault(st, []).append(p)
        for t in _id_tokens(st):
            by_token.setdefault(t, []).append(p)
            token_by_signer.setdefault(signer, {}).setdefault(t, []).append(p)
    return exact, by_token, exact_by_signer, token_by_signer


def _pick_best(cands: List[str], signer: Optional[str]) -> str:
    if not cands:
        raise ValueError("empty candidates")
    if signer:
        for p in cands:
            if f"/{signer}/" in p:
                return p
    return sorted(cands)[0]


def _match_path(
    label_stem: str,
    signer: Optional[str],
    exact: Dict[str, List[str]],
    by_token: Dict[str, List[str]],
    exact_by_signer: Dict[str, Dict[str, List[str]]],
    token_by_signer: Dict[str, Dict[str, List[str]]],
) -> Tuple[Optional[str], str]:
    st = label_stem.lower()
    signer_key = signer or ""

    if signer_key in exact_by_signer and st in exact_by_signer[signer_key]:
        return _pick_best(exact_by_signer[signer_key][st], signer), "exact_signer"
    if st in exact:
        return _pick_best(exact[st], signer), "exact_global"

    tokens = sorted(_id_tokens(st), key=lambda x: (-len(x), x))
    if signer_key in token_by_signer:
        for t in tokens:
            cands = token_by_signer[signer_key].get(t, [])
            if len(t) >= 3 and cands:
                return _pick_best(cands, signer), f"token_signer:{t}"

    for t in tokens:
        cands = by_token.get(t, [])
        if len(t) >= 3 and cands:
            return _pick_best(cands, signer), f"token_global:{t}"

    return None, "missing"


def _filter_signers(records: List[ClipRecord], signers: Optional[List[str]]) -> List[ClipRecord]:
    if not signers:
        return records
    sset = set(signers)
    return [r for r in records if r.signer_id in sset]


def _event_to_dict(ev: ParsedEvent) -> Dict:
    return asdict(ev)


def build_manifest(
    video_root: str,
    label_root: str,
    pose_root: Optional[str],
    signers: Optional[List[str]] = None,
    allow_pose_missing: bool = True,
) -> Manifest:
    video_files = _list_files(video_root, suffixes=VIDEO_EXTS)
    label_files = _list_files(label_root, suffixes={".json"})
    pose_files = _list_files(pose_root, suffixes={".json", ".npz"}) if pose_root else []

    v_exact, v_token, v_exact_s, v_token_s = _build_maps(video_files)
    p_exact, p_token, p_exact_s, p_token_s = _build_maps(pose_files)

    records: List[ClipRecord] = []
    schema_hits: Dict[str, int] = {}
    schema_errors: List[Dict] = []
    match_stats = {
        "video": {},
        "pose": {},
    }

    for lp in label_files:
        parsed: ParsedLabel = parse_label_file(lp)
        schema_hits[parsed.schema_type] = schema_hits.get(parsed.schema_type, 0) + 1

        signer = parsed.signer_id or _infer_signer_from_path(lp) or ""
        if signer and not str(signer).startswith("S"):
            if str(signer).isdigit():
                signer = f"S{signer}"
        if signers and signer not in set(signers):
            continue

        if not parsed.ok:
            schema_errors.append({
                "label_path": lp,
                "clip_id": parsed.clip_id,
                "schema_type": parsed.schema_type,
                "reason": parsed.parse_errors,
            })
            continue

        video_path, video_match = _match_path(parsed.clip_id, signer, v_exact, v_token, v_exact_s, v_token_s)
        pose_path, pose_match = _match_path(parsed.clip_id, signer, p_exact, p_token, p_exact_s, p_token_s)

        match_stats["video"][video_match] = match_stats["video"].get(video_match, 0) + 1
        match_stats["pose"][pose_match] = match_stats["pose"].get(pose_match, 0) + 1

        if video_path is None:
            schema_errors.append({
                "label_path": lp,
                "clip_id": parsed.clip_id,
                "schema_type": parsed.schema_type,
                "reason": ["matched label but missing video path"],
            })
            continue

        if pose_path is None and not allow_pose_missing:
            schema_errors.append({
                "label_path": lp,
                "clip_id": parsed.clip_id,
                "schema_type": parsed.schema_type,
                "reason": ["missing pose path"],
            })
            continue

        video_id = parsed.video_id
        if video_id.endswith(".mp4") or video_id.endswith(".avi") or video_id.endswith(".mov") or video_id.endswith(".mkv"):
            video_id = Path(video_id).stem
        if not video_id:
            video_id = Path(video_path).stem

        rec = ClipRecord(
            clip_id=parsed.clip_id,
            signer_id=signer,
            schema_type=parsed.schema_type,
            video_id=video_id,
            label_path=lp,
            video_path=video_path,
            pose_path=pose_path,
            fps=parsed.fps,
            n_frames=parsed.n_frames,
            events_by_hand=parsed.events_by_hand,
            parse_errors=parsed.parse_errors,
            match_video=video_match,
            match_pose=pose_match,
        )
        records.append(rec)

    records = _filter_signers(records, signers)
    records = sorted(records, key=lambda r: (r.signer_id, r.clip_id))

    schema_report = {
        "n_video_files": len(video_files),
        "n_label_files": len(label_files),
        "n_pose_files": len(pose_files),
        "n_records": len(records),
        "schema_hits": schema_hits,
        "match_stats": match_stats,
        "schema_errors": schema_errors,
        "records_preview": [
            {
                "clip_id": r.clip_id,
                "signer_id": r.signer_id,
                "video_id": r.video_id,
                "video_path": r.video_path,
                "pose_path": r.pose_path,
                "schema_type": r.schema_type,
                "events_L": len(r.events_by_hand.get("L", [])),
                "events_R": len(r.events_by_hand.get("R", [])),
            }
            for r in records[:20]
        ],
    }
    return Manifest(records=records, schema_report=schema_report)
