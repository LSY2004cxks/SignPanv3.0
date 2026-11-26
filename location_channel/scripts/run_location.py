from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.io import load_yaml_with_base, write_jsonl


def _aggregate_reference_items(
    items: List[Dict[str, Any]],
    *,
    signer_id: str,
    video_id: str,
    event_id: str,
    hand: str,
) -> Dict[str, Any]:
    totals: Dict[str, float] = {}
    for item in items:
        for label, probability in zip(
            item.get("top3", []),
            item.get("prob_top3", []),
        ):
            totals[label] = totals.get(label, 0.0) + float(probability)
    count = max(len(items), 1)
    ranked = sorted(
        ((label, total / count) for label, total in totals.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]
    mass = sum(probability for _, probability in ranked) or 1.0
    ranked = [(label, probability / mass) for label, probability in ranked]
    suffix = hand.upper()
    return {
        "event_id": event_id,
        "signer_id": signer_id,
        "sample_id": f"{signer_id}_{video_id}_{event_id}_{suffix}",
        "y_true": None,
        "y_pred_top1": ranked[0][0] if ranked else None,
        "top3": [label for label, _ in ranked],
        "prob_top3": [probability for _, probability in ranked],
        "margin": (
            ranked[0][1] - ranked[1][1]
            if len(ranked) > 1
            else (ranked[0][1] if ranked else 0.0)
        ),
        "reference_count": len(items),
    }


_LOCATION_MODEL = None


def _location_feature_path(signer_id, video_id, event_id, hand) -> Optional[Path]:
    if not (signer_id and video_id and event_id and hand):
        return None
    tag = "RIGHT" if str(hand).lower() == "right" else "LEFT"
    path = ROOT / "data" / "features" / f"{signer_id}_{video_id}_{event_id}_{tag}.npz"
    return path if path.is_file() else None


def _load_location_model(run_dir_path: Path):
    global _LOCATION_MODEL
    if _LOCATION_MODEL is not None:
        return _LOCATION_MODEL
    weight = run_dir_path / "best.pt"
    config_path = run_dir_path / "config_resolved.yaml"
    if not weight.is_file() or not config_path.is_file():
        return None
    import torch
    from src.kpts.geometry import compute_frame_features, root_scale_normalize
    from src.models.frame_mlp import FrameMLP
    from src.utils.io import read_json

    cfg = load_yaml_with_base(config_path)
    labels_path = ROOT / str(cfg["paths"]["location_labels"])
    if not labels_path.is_file():
        labels_path = ROOT / "configs" / "location_labels.json"
    codes = read_json(labels_path).get("codes", [])
    ckpt = torch.load(weight, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    input_dim = int(state["mlp.0.weight"].shape[1])
    model = FrameMLP(
        input_dim,
        len(codes),
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        emb_dim=int(cfg["model"].get("emb_dim", 256)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pool_mode=cfg.get("training", {}).get("pool", "logsumexp"),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    _LOCATION_MODEL = (model, codes, cfg, root_scale_normalize, compute_frame_features)
    print(f"[INFO] Location 无编号视频权重: {weight.parent.name}")
    return _LOCATION_MODEL


def _score_location_npz(loaded, npz_path, sample_id, event_id, signer_id, video_id, hand):
    model, codes, cfg, root_scale_normalize, compute_frame_features = loaded
    data = np.load(npz_path)
    pose_xy = data["pose_kpts"].astype(np.float32)
    pose_score = data["pose_score"].astype(np.float32) if "pose_score" in data else None
    hand_xy = data["hand_point"].astype(np.float32)
    pose_cfg = cfg.get("pose", {})
    feature_cfg = cfg.get("features", {})
    pose_norm, root, scale = root_scale_normalize(
        pose_xy,
        pose_score,
        joint_names=pose_cfg.get("joint_names", []),
        left_shoulder=pose_cfg.get("left_shoulder", "left_shoulder"),
        right_shoulder=pose_cfg.get("right_shoulder", "right_shoulder"),
        left_hip=pose_cfg.get("left_hip", "left_hip"),
        right_hip=pose_cfg.get("right_hip", "right_hip"),
    )
    hand_norm = (hand_xy - root) / np.maximum(scale, 1e-6)
    distances = feature_cfg.get("joint_distances", [])
    if not bool(feature_cfg.get("include_dists", True)):
        distances = []
    feat = compute_frame_features(
        hand_norm,
        pose_norm,
        joint_names=pose_cfg.get("joint_names", []),
        joint_distances=distances,
        include_dxy=feature_cfg.get("include_dxy", True),
        include_velocity=feature_cfg.get("include_velocity", False),
        pose_scores=pose_score,
    )
    if int(feat.shape[-1]) != int(model.mlp[0].in_features):
        print(
            f"[WARN] Location 特征维度 {feat.shape[-1]} "
            f"与权重 {model.mlp[0].in_features} 不一致，跳过推理"
        )
        return None
    import torch
    mask = np.ones((feat.shape[0],), dtype=np.float32)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(feat).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )["logits_clip"]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1][:3]
    top3 = [codes[int(index)] for index in order]
    top3_probs = [float(probs[int(index)]) for index in order]
    print(f"[INFO] Location 无编号视频推理: {event_id} -> {top3[0]}")
    return {
        "channel": "location",
        "source": "neural_inference",
        "schema": "location_pred_v1",
        "split": "test",
        "num_items": 1,
        "items": [{
            "sample_id": sample_id,
            "event_id": event_id,
            "signer_id": signer_id,
            "video_id": video_id,
            "hand": hand,
            "y_pred_top1": top3[0],
            "top3": top3,
            "prob_top3": top3_probs,
            "margin": top3_probs[0] - top3_probs[1] if len(top3_probs) > 1 else top3_probs[0],
        }],
        "status": "ok",
    }


def _infer_unseen_location(
    run_dir_path: Path,
    sample_id,
    event_id,
    signer_id,
    video_id,
    hand,
):
    """查表没有这条无编号视频时，用已有位置权重对当前片段推理。不改写语料预测表。"""
    npz_path = _location_feature_path(signer_id, video_id, event_id, hand)
    loaded = _load_location_model(run_dir_path)
    if npz_path is None or loaded is None:
        return None
    try:
        return _score_location_npz(
            loaded, npz_path, sample_id, event_id, signer_id, video_id, hand
        )
    except Exception as exc:
        print(f"[WARN] Location 无编号视频推理失败: {exc}")
        return None


def run_location(
    run_dir: str = "runs/loc_mlp_mil_pxv3",
    split: str = "test",
    features: Optional[str] = None,
    sample_id: Optional[str] = None,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    config: Optional[str] = None,
    allow_unnumbered_inference: bool = False,
) -> Dict[str, Any]:
    """
    Location 推理接口（工程级，run_all.py 友好）

    行为逻辑：
    1. 优先使用已有 predictions_{split}.jsonl
    2. 有 .npz 特征但无预测 → 声明 ready_for_inference
    3. 两者皆无 → 明确 unavailable
    """
    ROOT = Path(__file__).resolve().parents[1]

    run_dir_path = ROOT / run_dir
    if not run_dir_path.exists():
        return {
            "channel": "location",
            "source": "unavailable",
            "schema": "location_pred_v1",
            "status": "run_dir_not_found",
            "error": f"Run directory not found: {run_dir_path}",
        }

    #  自动选择特征目录（优先级 px_v3 > px_v2）
    if features is None:
        for cand in ["data/features_px_v3", "data/features_px_v2"]:
            p = ROOT / cand
            if p.exists():
                features = str(p)
                break

    features_path = ROOT / features if features else None
    existing_pred = run_dir_path / f"predictions_{split}.jsonl"

    # 已有预测结果（最常见、最稳定）

    if existing_pred.exists():
        items: List[Dict[str, Any]] = []
        reference_items: List[Dict[str, Any]] = []
        with existing_pred.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_sample_id = str(item.get("sample_id", ""))
                item_hand = item_sample_id.rsplit("_", 1)[-1].lower()
                matches = (
                    (sample_id is None or item_sample_id == sample_id)
                    and (event_id is None or item.get("event_id") == event_id)
                    and (signer_id is None or item.get("signer_id") == signer_id)
                    and (video_id is None or f"_{video_id}_" in item_sample_id)
                    and (hand is None or item_hand == hand.lower())
                )
                if matches:
                    items.append(item)
                reference_matches = (
                    event_id is not None
                    and video_id is not None
                    and hand is not None
                    and item.get("event_id") == event_id
                    and f"_{video_id}_" in item_sample_id
                    and item_hand == hand.lower()
                )
                if reference_matches:
                    reference_items.append(item)

        source = "existing_prediction"
        if (
            not items
            and signer_id
            and video_id
            and event_id
            and hand
            and reference_items
        ):
            items = [
                _aggregate_reference_items(
                    reference_items,
                    signer_id=signer_id,
                    video_id=video_id,
                    event_id=event_id,
                    hand=hand,
                )
            ]
            source = "cross_signer_fallback"

        if not items and allow_unnumbered_inference:
            inferred = _infer_unseen_location(
                run_dir_path,
                sample_id,
                event_id,
                signer_id,
                video_id,
                hand,
            )
            if inferred is not None:
                return inferred

        return {
            "channel": "location",
            "source": source,
            "schema": "location_pred_v1",       # ✅ 新增
            "split": split,
            "num_items": len(items),
            "items": items,
            "features_dir": str(features_path) if features_path else None,
            "status": "ok" if items else "sample_not_found",
        }


    #有 .npz 特征，但没有 predictions → 应触发推理

    if features_path and features_path.exists():
        cfg_path = ROOT / (config or "configs/location_config.yaml")
        cfg = load_yaml_with_base(cfg_path) if cfg_path.exists() else {}

        return {
            "channel": "location",
            "source": "ready_for_inference",
            "schema": "location_pred_v1",       #  新增
            "status": "features_available_but_prediction_not_generated",
            "features_dir": str(features_path),
            "run_dir": str(run_dir_path),
            "config": str(cfg_path),
            "split": split,
            "message": "Call location_main() / inference entry here.",
        }

    # ------------------------------------------------------------
    # 3️⃣ 什么都缺 → 明确失败（不假装成功）
    # ------------------------------------------------------------
    return {
        "channel": "location",
        "source": "unavailable",
        "schema": "location_pred_v1",           #  新增
        "status": "missing_features_and_predictions",
        "error": (
            f"No predictions_{split}.jsonl found, "
            f"and no .npz features dir found. "
            f"Checked run_dir={run_dir_path}, features={features_path}"
        ),
    }


# ============================================================
# CLI 入口（给 run_all.py / 手动调试用）
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Location 推理接口（工程版）")
    parser.add_argument("--run_dir", default="runs/loc_mlp_mil_pxv3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--features")
    parser.add_argument("--sample_id", help="指定完整 sample_id")
    parser.add_argument("--event_id")
    parser.add_argument("--signer_id")
    parser.add_argument("--video_id")
    parser.add_argument("--hand", choices=["left", "right"])
    parser.add_argument("--config", default="configs/location_config.yaml")

    args = parser.parse_args()

    result = run_location(
        run_dir=args.run_dir,
        split=args.split,
        features=args.features,
        sample_id=args.sample_id,
        event_id=args.event_id,
        signer_id=args.signer_id,
        video_id=args.video_id,
        hand=args.hand,
        config=args.config,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))