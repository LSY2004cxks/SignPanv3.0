from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHANNELS = ("location", "handshape", "orientation", "movement")


def _event_from_legacy_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id and (
        sample_id.startswith("L_Event_") or sample_id.startswith("R_Event_")
    ):
        return sample_id
    return None


def _parse_full_sample_id(sample_id: Optional[str]) -> Dict[str, str]:
    if not sample_id:
        return {}
    match = re.search(
        r"^(?P<signer>S\d+)_(?P<video>V\d+)_.*?"
        r"(?P<event>[LR]_Event_\d+)(?:_(?P<hand>LEFT|RIGHT))?$",
        sample_id,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    hand = match.group("hand")
    if hand is None:
        hand = "left" if match.group("event").upper().startswith("L_") else "right"
    return {
        "signer_id": match.group("signer"),
        "video_id": match.group("video"),
        "event_id": match.group("event"),
        "hand": hand.lower(),
    }


def _normalise_hand(value: Any) -> Optional[str]:
    hand = str(value or "").lower()
    return {"l": "left", "r": "right"}.get(hand, hand or None)


def _item_matches(
    item: Dict[str, Any],
    *,
    sample_id: Optional[str],
    event_id: Optional[str],
    signer_id: Optional[str],
    video_id: Optional[str],
    hand: Optional[str],
) -> bool:
    item_sample = str(item.get("sample_id", ""))
    item_signer = item.get("signer_id")
    if item_signer is None and item_sample.startswith("S"):
        item_signer = item_sample.split("_", 1)[0]
    item_video = item.get("video_id")
    if item_video is None and "_V" in item_sample:
        item_video = item_sample.split("_", 2)[1]
    item_hand = _normalise_hand(item.get("hand"))
    if item_hand is None and item_sample:
        item_hand = _normalise_hand(item_sample.rsplit("_", 1)[-1])

    return (
        (sample_id is None or item_sample == sample_id)
        and (event_id is None or item.get("event_id") == event_id)
        and (signer_id is None or item_signer == signer_id)
        and (video_id is None or item_video == video_id)
        and (hand is None or item_hand == _normalise_hand(hand))
    )


def _select_item(
    raw_result: Dict[str, Any],
    *,
    sample_id: Optional[str],
    event_id: Optional[str],
    signer_id: Optional[str],
    video_id: Optional[str],
    hand: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    items = raw_result.get("items") or []
    matches = [
        item
        for item in items
        if _item_matches(
            item,
            sample_id=sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
        )
    ]
    if len(matches) == 1:
        return matches[0], "OK", ""
    if len(matches) > 1:
        return None, "AMBIGUOUS_SAMPLE", (
            f"{len(matches)} predictions match the supplied identifiers; "
            "provide signer_id, video_id and hand"
        )

    source = raw_result.get("source")
    raw_status = str(raw_result.get("status", ""))
    if source in {"unavailable", "ready_for_inference"} or raw_status in {
        "run_dir_not_found",
        "missing_features_and_predictions",
        "features_available_but_prediction_not_generated",
        "prediction_file_not_found",
        "no_models_found",
        "all_models_failed",
        "config_not_found",
    }:
        message = (
            raw_result.get("error")
            or raw_result.get("message")
            or raw_status
            or "Model or prediction asset is unavailable"
        )
        return None, "MODEL_UNAVAILABLE", str(message)
    return None, "SAMPLE_NOT_FOUND", "No prediction matches the supplied identifiers"


def _normalise_topk(raw: Dict[str, Any], channel: str) -> List[Dict[str, Any]]:
    if channel == "handshape":
        return [
            {"label": item.get("label"), "prob": float(item.get("prob", 0.0))}
            for item in raw.get("topk", [])
        ]

    labels = raw.get("top3") or raw.get("top3_labels") or []
    probs = raw.get("prob_top3") or raw.get("top3_probs") or []
    return [
        {"label": label, "prob": float(prob)}
        for label, prob in zip(labels, probs)
    ]


def normalise_channel_output(
    channel: str,
    raw_result: Dict[str, Any],
    *,
    sample_id: Optional[str] = None,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
) -> Dict[str, Any]:
    item, code, message = _select_item(
        raw_result,
        sample_id=sample_id,
        event_id=event_id,
        signer_id=signer_id,
        video_id=video_id,
        hand=hand,
    )
    if item is None:
        return {
            "schema": "sign_phonology_v1",
            "sample_id": sample_id,
            "event_id": event_id,
            "channel": channel,
            "prediction": {},
            "metadata": {
                "signer_id": signer_id,
                "video_id": video_id,
                "hand": _normalise_hand(hand),
                "source": raw_result.get("source"),
            },
            "status": {"code": code, "message": message},
        }

    topk = _normalise_topk(item, channel)
    label = (
        item.get("y_pred_top1")
        or item.get("orientation")
        or item.get("movement")
        or (topk[0].get("label") if topk else None)
    )
    confidence = float(topk[0].get("prob", 0.0)) if topk else float(
        item.get("conf", 0.0)
    )
    resolved_sample_id = item.get("sample_id")
    if channel == "orientation" and sample_id:
        resolved_sample_id = sample_id

    metadata: Dict[str, Any] = {
        "gt": item.get("y_true") or item.get("gt"),
        "span": item.get("span"),
        "signer_id": item.get("signer_id") or signer_id,
        "video_id": item.get("video_id") or video_id,
        "hand": _normalise_hand(item.get("hand") or hand),
        "source": raw_result.get("source"),
        "reference_count": item.get("reference_count"),
    }
    if channel == "handshape":
        metadata["entropy"] = item.get("entropy")
        metadata["autofill_top1"] = item.get("autofill_top1")
        metadata["tier"] = item.get("tier")
        metadata["raw_conf"] = item.get("raw_conf")
        metadata["calibration"] = item.get("calibration")
    if channel == "orientation":
        metadata["hitl_gate"] = item.get("hitl_gate")
        metadata["quality_flags"] = item.get("quality_flags", [])
    if channel == "movement":
        metadata["local_label"] = item.get("local_label")
        metadata["track_point"] = item.get("track_point")
        metadata["repeat"] = item.get("repeat")
        metadata["geometry"] = item.get("geometry")
        metadata["corpus_label"] = item.get("corpus_label")

    return {
        "schema": "sign_phonology_v1",
        "sample_id": resolved_sample_id or sample_id or item.get("event_id"),
        "event_id": item.get("event_id") or event_id,
        "channel": channel,
        "prediction": {
            "label": label,
            "topk": topk,
            "conf": confidence,
            "margin": item.get("margin"),
        },
        "metadata": metadata,
        "status": {"code": "OK", "message": ""},
    }


def _run_channels(
    *,
    split: str,
    sample_id: Optional[str],
    event_id: Optional[str],
    signer_id: Optional[str],
    video_id: Optional[str],
    hand: Optional[str],
    exp_dir: str,
    span: Any = None,
    video_path: Optional[str] = None,
    allow_unnumbered_inference: bool = False,
) -> Dict[str, Dict[str, Any]]:
    from location_channel.scripts.run_location import run_location
    from handshape_channel_data28.scripts.run_handshape import run_handshape
    from orientation_channel.scripts.run_orientation import run_orientation
    from movement_channel.scripts.run_movement import run_movement

    return {
        "location": run_location(
            split=split,
            sample_id=sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
            allow_unnumbered_inference=allow_unnumbered_inference,
        ),
        "handshape": run_handshape(
            split=split,
            sample_id=sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
            video_path=video_path,
            span=span,
            allow_unnumbered_inference=allow_unnumbered_inference,
        ),
        "orientation": run_orientation(
            split=split,
            exp_dir=exp_dir,
            sample_id=event_id,
            event_id=event_id,
            video_id=video_id,
            hand=hand,
            signer_id=signer_id,
            video_path=video_path,
            span=span,
            allow_unnumbered_inference=allow_unnumbered_inference,
        ),
        "movement": run_movement(
            split=split,
            sample_id=sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
            span=span,
        ),
    }


def assemble_result(
    channel_results: Dict[str, Dict[str, Any]],
    *,
    split: str = "test",
    sample_id: Optional[str] = None,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    exp_dir: str = "../outputs_ablation/modal6_20260305_201215/exp_modal_roi",
) -> Dict[str, Any]:
    requested_sample_id = sample_id
    parsed = _parse_full_sample_id(sample_id)
    event_id = event_id or parsed.get("event_id") or _event_from_legacy_sample_id(sample_id)
    signer_id = signer_id or parsed.get("signer_id")
    video_id = video_id or parsed.get("video_id")
    hand = hand or parsed.get("hand")
    match_sample_id = None if parsed else sample_id
    if event_id == sample_id:
        match_sample_id = None

    channels = {
        channel: normalise_channel_output(
            channel,
            channel_results.get(
                channel,
                {"source": "unavailable", "status": "channel_result_missing"},
            ),
            sample_id=match_sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
        )
        for channel in CHANNELS
    }
    available = sum(
        value["status"]["code"] == "OK" for value in channels.values()
    )
    return {
        "schema": "sign_phonology_v1",
        "split": split,
        "sample_id": requested_sample_id,
        "event_id": event_id,
        "signer_id": signer_id,
        "video_id": video_id,
        "hand": _normalise_hand(hand),
        "exp_dir": {"orientation": exp_dir},
        "channels": channels,
        "summary": {
            f"{channel}_available": channels[channel]["status"]["code"] == "OK"
            for channel in CHANNELS
        }
        | {"total_channels": len(CHANNELS), "available_channels": available},
    }


def run_all(
    split: str = "test",
    sample_id: Optional[str] = None,
    exp_dir: str = "../outputs_ablation/modal6_20260305_201215/exp_modal_roi",
    *,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    channel_results: Optional[Dict[str, Dict[str, Any]]] = None,
    channel_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    span: Any = None,
    video_path: Optional[str] = None,
    allow_unnumbered_inference: bool = False,
) -> Dict[str, Any]:
    requested_sample_id = sample_id
    parsed = _parse_full_sample_id(sample_id)
    event_id = event_id or parsed.get("event_id") or _event_from_legacy_sample_id(sample_id)
    signer_id = signer_id or parsed.get("signer_id")
    video_id = video_id or parsed.get("video_id")
    hand = hand or parsed.get("hand")
    match_sample_id = None if parsed or event_id == sample_id else sample_id
    if channel_results is not None:
        raw = dict(channel_results)
    else:
        raw = _run_channels(
            split=split,
            sample_id=match_sample_id,
            event_id=event_id,
            signer_id=signer_id,
            video_id=video_id,
            hand=hand,
            exp_dir=exp_dir,
            span=span,
            video_path=video_path,
            allow_unnumbered_inference=allow_unnumbered_inference,
        )
    if channel_overrides:
        for channel, override in channel_overrides.items():
            _, base_code, _ = _select_item(
                raw.get(channel, {}),
                sample_id=match_sample_id,
                event_id=event_id,
                signer_id=signer_id,
                video_id=video_id,
                hand=hand,
            )
            if base_code != "OK" or (
                channel == "movement"
                and override.get("source") == "geometric_fallback"
            ):
                if (
                    channel == "movement"
                    and base_code == "OK"
                    and override.get("items")
                    and (raw.get(channel) or {}).get("items")
                ):
                    gold_item = raw[channel]["items"][0]
                    corpus_label = gold_item.get("gt") or gold_item.get("y_pred_top1")
                    for item in override["items"]:
                        item["gt"] = corpus_label
                        item["corpus_label"] = corpus_label
                raw[channel] = override
    return assemble_result(
        raw,
        split=split,
        sample_id=requested_sample_id,
        event_id=event_id,
        signer_id=signer_id,
        video_id=video_id,
        hand=hand,
        exp_dir=exp_dir,
    )


def run_all_batch(
    requests: Iterable[Any],
    split: str = "test",
    exp_dir: str = "../outputs_ablation/modal6_20260305_201215/exp_modal_roi",
) -> List[Dict[str, Any]]:
    results = []
    for request in requests:
        if isinstance(request, str):
            request = {"event_id": request}
        results.append(run_all(split=split, exp_dir=exp_dir, **request))
    return results


def run_preprocessed(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(input_path.read_text(encoding="utf-8"))
        payloads = loaded if isinstance(loaded, list) else [loaded]

    results = []
    for payload in payloads:
        if (
            payload.get("schema") == "sign_phonology_v1"
            and isinstance(payload.get("channels"), dict)
            and all(
                isinstance(value, dict) and "prediction" in value and "status" in value
                for value in payload["channels"].values()
            )
        ):
            results.append(payload)
            continue
        context = payload.get("context", payload)
        channel_results = payload.get("channel_results") or payload.get("channels")
        if not isinstance(channel_results, dict):
            raise ValueError(
                "Preprocessed input must contain channel_results/channels"
            )
        results.append(
            run_all(
                split=context.get("split", "test"),
                sample_id=context.get("sample_id"),
                event_id=context.get("event_id"),
                signer_id=context.get("signer_id"),
                video_id=context.get("video_id"),
                hand=context.get("hand"),
                exp_dir=context.get("exp_dir", "../outputs_ablation/modal6_20260305_201215/exp_modal_roi"),
                channel_results=channel_results,
            )
        )
    return results


def _write_or_print(payload: Any, output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="手语音系四通道统一入口（视频/预处理结果/事件）"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="输入视频；自动提取事件特征并输出统一 JSON")
    source.add_argument(
        "--preprocessed",
        help="包含 context 与 channel_results/channels 的 JSON/JSONL",
    )
    source.add_argument("--batch_file", help="事件请求列表；每行可为 event_id 或 JSON")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_id", help="完整 sample_id")
    parser.add_argument("--event_id")
    parser.add_argument("--signer_id")
    parser.add_argument("--video_id")
    parser.add_argument("--hand", choices=["left", "right"])
    parser.add_argument("--exp_dir", default="../outputs_ablation/modal6_20260305_201215/exp_modal_roi")
    parser.add_argument("--events", default="data28/annotations/mhsl_events.jsonl")
    parser.add_argument("--work_dir", default="output")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    if args.video:
        from video2json import video2json

        payload = video2json(
            video_path=args.video,
            events_jsonl=args.events,
            out_dir=args.work_dir,
            run_inference=True,
        )
    elif args.preprocessed:
        payload = run_preprocessed(args.preprocessed)
    elif args.batch_file:
        requests: List[Any] = []
        for line in Path(args.batch_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                requests.append(json.loads(line) if line.startswith("{") else line)
        payload = run_all_batch(requests, split=args.split, exp_dir=args.exp_dir)
    else:
        payload = run_all(
            split=args.split,
            sample_id=args.sample_id,
            event_id=args.event_id,
            signer_id=args.signer_id,
            video_id=args.video_id,
            hand=args.hand,
            exp_dir=args.exp_dir,
        )
    _write_or_print(payload, args.output)


if __name__ == "__main__":
    main()
