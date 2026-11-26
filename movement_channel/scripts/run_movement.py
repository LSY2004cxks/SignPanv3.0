from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from movement_channel.src.gold import (
    aggregate_reference_items,
    annotation_path,
    cross_signer_items,
    lookup_gold,
)
from movement_channel.src.geometry import movement_from_wrist


def run_movement(
    split: str = "test",
    sample_id: Optional[str] = None,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    span: Any = None,
    ann_root: Optional[str] = None,
    features: Optional[str] = None,
    wrist_points: Any = None,
    wrist_scores: Any = None,
) -> Dict[str, Any]:
    """
    Movement 通道入口。

    处理当前视频时，手腕轨迹优先于语料金标，避免新视频因文件名碰巧对上
    data28 就被写成 gold_lookup。
    无视频轨迹时，才用金标查询或跨 signer 投票。
    """
    annotations = Path(ann_root) if ann_root else PROJECT_ROOT / "data28"
    features_path = Path(features) if features else ROOT / "data" / "features"

    gold = lookup_gold(
        ann_root=annotations,
        signer_id=signer_id,
        video_id=video_id,
        event_id=event_id,
        hand=hand,
        span=span,
    )

    if wrist_points is not None:
        geometric = movement_from_wrist(
            wrist_points,
            wrist_scores,
            hand=hand or "right",
            event_id=event_id or sample_id or "unknown",
            signer_id=signer_id or "",
            video_id=video_id or "",
            span=span,
        )
        if geometric.get("items"):
            if gold is not None:
                corpus_label = gold.get("y_pred_top1") or gold.get("gt")
                for item in geometric["items"]:
                    item["gt"] = corpus_label
                    item["corpus_label"] = corpus_label
            return geometric

    if gold is not None:
        if sample_id:
            gold["sample_id"] = sample_id
        return {
            "channel": "movement",
            "source": "gold_lookup",
            "schema": "movement_pred_v1",
            "split": split,
            "status": "ok",
            "num_items": 1,
            "items": [gold],
            "ann_path": str(
                annotation_path(annotations, str(signer_id), str(video_id))
            ),
        }

    if signer_id and video_id and hand:
        references = cross_signer_items(
            ann_root=annotations,
            signer_id=signer_id,
            video_id=video_id,
            event_id=event_id,
            hand=hand,
            span=span,
        )
        if references:
            item = aggregate_reference_items(
                references,
                signer_id=signer_id,
                video_id=video_id,
                event_id=event_id or "unknown",
                hand=hand,
                span=span,
            )
            if sample_id:
                item["sample_id"] = sample_id
            return {
                "channel": "movement",
                "source": "cross_signer_fallback",
                "schema": "movement_pred_v1",
                "split": split,
                "status": "ok",
                "num_items": 1,
                "items": [item],
            }

    if features_path.exists() and any(features_path.glob("*.npz")):
        return {
            "channel": "movement",
            "source": "ready_for_inference",
            "schema": "movement_pred_v1",
            "status": "features_available_but_prediction_not_generated",
            "features_dir": str(features_path),
            "split": split,
            "message": "Movement 尚无训练模型；请提供金标或手腕轨迹。",
        }

    return {
        "channel": "movement",
        "source": "unavailable",
        "schema": "movement_pred_v1",
        "status": "missing_features_and_predictions",
        "error": (
            "No gold movement label, no cross-signer label, "
            "and no wrist trajectory for geometric fallback"
        ),
        "items": [],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Movement 推理接口")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_id")
    parser.add_argument("--event_id")
    parser.add_argument("--signer_id")
    parser.add_argument("--video_id")
    parser.add_argument("--hand", choices=["left", "right"])
    parser.add_argument("--ann_root")
    args = parser.parse_args()
    print(
        json.dumps(
            run_movement(
                split=args.split,
                sample_id=args.sample_id,
                event_id=args.event_id,
                signer_id=args.signer_id,
                video_id=args.video_id,
                hand=args.hand,
                ann_root=args.ann_root,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
