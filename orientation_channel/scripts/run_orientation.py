from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ✅ 自举路径（与 Location / Handshape 一致）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2026-03-05 exp_modal_roi。相对 orientation_channel/outputs。
DEFAULT_EXP_DIR = "../outputs_ablation/modal6_20260305_201215/exp_modal_roi"
_ORIENTATION_MODEL = None


def _experiment_dir(exp_dir: str) -> Path:
    return ROOT / "outputs" / exp_dir


def _import_orientation_model():
    """用手语通道自己的包名加载，避免和位置通道的 src 撞名。"""
    import importlib.util
    import types

    base = "signpan_orientation"
    packages = {
        base: ROOT / "src",
        f"{base}.models": ROOT / "src" / "models",
        f"{base}.data": ROOT / "src" / "data",
        f"{base}.utils": ROOT / "src" / "utils",
    }
    for name, path in packages.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    model_name = f"{base}.models.orientation_model"
    if model_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            model_name,
            ROOT / "src" / "models" / "orientation_model.py",
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = f"{base}.models"
        sys.modules[model_name] = module
        spec.loader.exec_module(module)
    return sys.modules[model_name].OrientationModel


def palm_normal_from_landmarks(landmarks, hand: str):
    _import_orientation_model()
    from signpan_orientation.data.dataset import compute_palm_normal, normalize_landmarks

    xyz, _mask = normalize_landmarks(landmarks)
    return compute_palm_normal(xyz, hand)


def _load_orientation_model(exp_dir: str):
    """加载朝向权重。失败时返回 None，调用方继续使用几何估计。"""
    global _ORIENTATION_MODEL
    if _ORIENTATION_MODEL is not None:
        return _ORIENTATION_MODEL
    weight = _experiment_dir(exp_dir) / "checkpoints" / "best.pt"
    if not weight.is_file():
        print(f"[WARN] Orientation 权重不存在: {weight}")
        return None
    import torch
    OrientationModel = _import_orientation_model()

    ckpt = torch.load(weight, map_location="cpu")
    labels = list(ckpt.get("labels") or [])
    model_cfg = (ckpt.get("config") or {}).get("model") or {}
    if not labels:
        return None
    model = OrientationModel(
        num_classes=len(labels),
        roi_backbone=model_cfg.get("roi_backbone", "resnet18"),
        roi_feat_dim=int(model_cfg.get("roi_feat_dim", 512)),
        geom_encoder_type=model_cfg.get("geom_encoder_type", "stgcn"),
        geom_dim=int(model_cfg.get("geom_dim", 256)),
        normal_dim=int(model_cfg.get("normal_dim", 128)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        cross_attn_heads=int(model_cfg.get("cross_attn_heads", 4)),
        use_roi=bool(model_cfg.get("use_roi", True)),
        use_geom=bool(model_cfg.get("use_geom", False)),
        use_normal=bool(model_cfg.get("use_normal", False)),
        use_text=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    params = (ckpt.get("config") or {}).get("params") or {}
    _ORIENTATION_MODEL = (model, labels, int(params.get("roi_size", 224)))
    print(f"[INFO] Orientation 无编号视频权重: {weight.parent.parent.name}")
    return _ORIENTATION_MODEL


def _orientation_feature_path(event_id: Optional[str]) -> Optional[Path]:
    if not event_id:
        return None
    path = ROOT / "data" / "precomputed" / f"{event_id}.npz"
    return path if path.is_file() else None


def _score_orientation_npz(loaded, npz_path: Path, sample_id, event_id, signer_id, video_id, hand):
    import cv2
    import numpy as np
    import torch

    model, labels, roi_size = loaded
    with np.load(npz_path) as data:
        frames = np.array(data["roi"] if "roi" in data.files else data["frames"])
    if frames.size == 0:
        return None
    resized = np.stack(
        [cv2.resize(frames[index], (roi_size, roi_size)) for index in range(frames.shape[0])]
    )
    images = resized.astype(np.float32) / 255.0
    images = np.transpose(images, (0, 3, 1, 2))
    batch = {"roi_img": torch.from_numpy(images).unsqueeze(0)}
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=-1)[0].cpu().numpy()
    order = np.argsort(-probabilities)[:3]
    top_labels = [labels[int(index)] for index in order]
    top_probs = [float(probabilities[int(index)]) for index in order]
    print(f"[INFO] Orientation 无编号视频推理: {event_id} -> {top_labels[0]}")
    return {
        "channel": "orientation",
        "source": "neural_inference",
        "schema": "orientation_pred_v1",
        "split": "test",
        "num_items": 1,
        "items": [{
            "sample_id": sample_id or event_id,
            "event_id": event_id,
            "signer_id": signer_id,
            "video_id": video_id,
            "hand": hand,
            "orientation": top_labels[0],
            "top3_labels": top_labels,
            "top3_probs": top_probs,
            "conf": top_probs[0],
            "margin": top_probs[0] - top_probs[1] if len(top_probs) > 1 else top_probs[0],
            "hitl_gate": "recommend_top3",
            "quality_flags": [],
        }],
        "status": "ok",
    }


def _read_orientation_items(pred_file: Path, sample_id, event_id, video_id, hand):
    items: List[Dict[str, Any]] = []
    with pred_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["sample_id"] = item.get("event_id")
            requested_event = event_id or sample_id
            item_hand = str(item.get("hand", "")).lower()
            if (
                (requested_event is None or item.get("event_id") == requested_event)
                and (video_id is None or item.get("video_id") == video_id)
                and (hand is None or item_hand == hand.lower())
            ):
                item["orientation"] = item.get("top3_labels", [None])[0]
                item["top3"] = item.get("top3_labels")
                item["prob"] = item.get("top3_probs")
                items.append(item)
    return items


def _infer_unseen_orientation(exp_dir, sample_id, event_id, signer_id, video_id, hand):
    """结果表没有这条新视频时，用已有朝向权重对当前片段推理。"""
    npz_path = _orientation_feature_path(event_id)
    loaded = _load_orientation_model(exp_dir)
    if npz_path is None or loaded is None:
        return None
    try:
        return _score_orientation_npz(
            loaded, npz_path, sample_id, event_id, signer_id, video_id, hand
        )
    except Exception as exc:
        print(f"[WARN] Orientation 无编号视频推理失败: {exc}")
        return None


def run_orientation(
    split: str = "test",
    exp_dir: str = DEFAULT_EXP_DIR,
    sample_id: Optional[str] = None,          # ✅ 对外统一叫 sample_id
    event_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_path: Optional[str] = None,
    span: Optional[dict] = None,
    allow_unnumbered_inference: bool = False,
) -> Dict[str, Any]:
    """
    Orientation 推理接口（三通道统一版）

    注意：
    ◦ 原始 jsonl 中主键为 event_id

    ◦ 本接口在【读取后】将其映射为 sample_id

    ◦ 原始字段 event_id 仍保留，便于溯源

    """

    def find_pred_file() -> Path:
        # exp_dir 为空时，直接找 outputs 下所有 results 目录
        if exp_dir:
            p = ROOT / "outputs" / exp_dir / "results" / f"orientation_top3_{split}.jsonl"
            if p.exists():
                return p
        candidates = list(
            ROOT.glob(f"outputs/*/results/orientation_top3_{split}.jsonl")
        )
        if candidates:
            return candidates[0]
        return None

    pred_file = find_pred_file()
    items: List[Dict[str, Any]] = []
    if pred_file is not None and pred_file.is_file():
        items.extend(_read_orientation_items(
            pred_file, sample_id, event_id, video_id, hand
        ))

    if not items and allow_unnumbered_inference:
        inferred = _infer_unseen_orientation(
            exp_dir, sample_id, event_id, signer_id, video_id, hand
        )
        if inferred is not None:
            return inferred

    if pred_file is None or not pred_file.is_file():
        return {
            "channel": "orientation",
            "source": "unavailable",
            "schema": "orientation_pred_v1",
            "status": "prediction_file_not_found",
            "split": split,
            "exp_dir": exp_dir,
            "error": f"Orientation prediction file not found for split={split}",
        }

    if not items:
        return {
            "channel": "orientation",
            "source": "existing_prediction",
            "schema": "orientation_pred_v1",
            "status": "sample_id_not_found",
            "split": split,
            "sample_id": sample_id,
            "error": f"Sample {sample_id} not found in {pred_file}",
        }

    return {
        "channel": "orientation",
        "source": "existing_prediction",
        "schema": "orientation_pred_v1",
        "split": split,
        "exp_dir": exp_dir,
        "num_items": len(items),
        "items": items,
        "result": items[0] if sample_id else None,
        "pred_file": str(pred_file.resolve().relative_to(ROOT)),
        # 明确声明：接口层已完成 ID 统一
        "id_mapping": "event_id -> sample_id (interface only)",
    }


# ============================================================
# CLI 入口（统一使用 sample_id）
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orientation 推理接口（三通道统一版）")
    parser.add_argument("--split", default="test")
    parser.add_argument("--exp_dir", default=DEFAULT_EXP_DIR)
    parser.add_argument("--sample_id", help="指定 sample_id（原始为 event_id）")
    parser.add_argument("--event_id")
    parser.add_argument("--video_id")
    parser.add_argument("--hand", choices=["left", "right"])
    args = parser.parse_args()

    result = run_orientation(
        split=args.split,
        exp_dir=args.exp_dir,
        sample_id=args.sample_id,
        event_id=args.event_id,
        video_id=args.video_id,
        hand=args.hand,
        allow_unnumbered_inference=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))