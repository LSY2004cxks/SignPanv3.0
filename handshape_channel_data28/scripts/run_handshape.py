from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
# 路径设置
ROOT = Path(__file__).resolve().parents[1]   # handshape_channel_data28/
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# =======================
# 常量路径
# =======================
RUNS_DIR = ROOT / "runs_schemeA"
CONFIG = ROOT / "configs" / "handshape_data28.yaml"
OUT_JSON = ROOT / "test_pred.json"
OUT_CSV = ROOT / "test_predictions.csv"
CALIBRATION_TEMPERATURE = 5.8
NUM_CLASSES = 30

# =======================
# 多模型融合配置
# =======================
# ✅ 排除出问题的 deform_nce_seed43，用其余 5 个模型做集成
EXCLUDE_MODELS = ["deform_nce_seed43"]


def _calibrate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """对部署模型的 Top-K softmax 做验证集温度校准，并保留原始分数。"""
    topk = item.get("topk", [])
    if not topk or item.get("calibration", {}).get("applied"):
        return item

    calibrated = dict(item)
    calibrated["topk"] = [dict(candidate) for candidate in topk]
    probabilities = [
        max(float(candidate.get("prob", 0.0)), 1e-12)
        for candidate in topk
    ]
    tail_mass = max(1.0 - sum(probabilities), 1e-12)
    remaining_classes = max(NUM_CLASSES - len(probabilities), 1)
    exponent = 1.0 / CALIBRATION_TEMPERATURE
    top_weights = [probability ** exponent for probability in probabilities]
    tail_weight = (tail_mass / remaining_classes) ** exponent
    normalizer = sum(top_weights) + remaining_classes * tail_weight
    calibrated_probs = [weight / normalizer for weight in top_weights]

    calibrated["raw_topk"] = [dict(candidate) for candidate in topk]
    calibrated["raw_conf"] = probabilities[0]
    for candidate, probability in zip(calibrated["topk"], calibrated_probs):
        candidate["prob"] = float(probability)
    calibrated["margin"] = (
        calibrated_probs[0] - calibrated_probs[1]
        if len(calibrated_probs) > 1
        else calibrated_probs[0]
    )
    calibrated["calibration"] = {
        "applied": True,
        "method": "temperature_scaling",
        "temperature": CALIBRATION_TEMPERATURE,
        "num_classes": NUM_CLASSES,
        "fit_split": "val",
        "fit_samples": 1233,
        "approximation": "top3_with_uniform_tail",
    }
    tier = str(calibrated.get("tier", "")).upper()
    calibrated["autofill_top1"] = bool(
        tier in {"A", "B"}
        and calibrated_probs[0] >= 0.7
        and calibrated["margin"] >= 0.2
    )
    return calibrated


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
        for candidate in item.get("topk", []):
            label = candidate.get("label")
            if label:
                totals[label] = totals.get(label, 0.0) + float(
                    candidate.get("prob", 0.0)
                )
    count = max(len(items), 1)
    ranked = sorted(
        ((label, total / count) for label, total in totals.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]
    mass = sum(probability for _, probability in ranked) or 1.0
    ranked = [(label, probability / mass) for label, probability in ranked]
    hand_code = "L" if hand.lower() == "left" else "R"
    return {
        "sample_id": f"{signer_id}_{video_id}_{hand_code}_{event_id}",
        "video_id": video_id,
        "hand": hand_code,
        "event_id": event_id,
        "gt": None,
        "tier": "REFERENCE",
        "topk": [
            {"label": label, "prob": probability}
            for label, probability in ranked
        ],
        "margin": (
            ranked[0][1] - ranked[1][1]
            if len(ranked) > 1
            else (ranked[0][1] if ranked else 0.0)
        ),
        "entropy": None,
        "autofill_top1": False,
        "reference_count": len(items),
    }


def _get_available_models() -> List[Path]:
    """获取所有可用的模型目录（排除黑名单）"""
    models = []
    if not RUNS_DIR.exists():
        return models
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        # 排除黑名单
        excluded = False
        for ex in EXCLUDE_MODELS:
            if ex in d.name:
                excluded = True
                break
        if excluded:
            continue
        # 必须有 best.ckpt
        if (d / "best.ckpt").exists():
            models.append(d)
    return models


# =======================
# 核心函数
# =======================

_HANDSHAPE_MODEL = None


def _handshape_checkpoint():
    preferred = []
    others = []
    for model_dir in _get_available_models():
        name = model_dir.name
        if any(token in name for token in EXCLUDE_MODELS):
            continue
        if "deform_nce" in name:
            preferred.append(model_dir)
        else:
            others.append(model_dir)
    chosen = preferred or others
    if not chosen:
        return None
    return sorted(chosen)[-1] / "best.ckpt"


def _load_handshape_model():
    global _HANDSHAPE_MODEL
    if _HANDSHAPE_MODEL is not None:
        return _HANDSHAPE_MODEL
    ckpt_path = _handshape_checkpoint()
    if ckpt_path is None:
        return None
    import torch
    from data.dataset import roi_to_tensor
    from models import HandshapeNet

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_to_idx = ckpt.get("label_to_idx") or {}
    idx_to_label = ckpt.get("idx_to_label")
    if isinstance(idx_to_label, list):
        idx_to_label = {index: label for index, label in enumerate(idx_to_label)}
    elif isinstance(idx_to_label, dict):
        idx_to_label = {int(index): label for index, label in idx_to_label.items()}
    elif label_to_idx:
        idx_to_label = {int(index): label for label, index in label_to_idx.items()}
    else:
        return None

    model = HandshapeNet(
        mode="roi_skel_deform_nce",
        num_classes=len(idx_to_label),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    _HANDSHAPE_MODEL = (model, idx_to_label, roi_to_tensor)
    print(f"[INFO] Handshape 无编号视频权重: {ckpt_path.parent.name}")
    return _HANDSHAPE_MODEL


def _crop_clip_from_cached_pose(video_path, span, hand, frames=24):
    """切分时已经有关键点，手形只按这些点裁图，不再重新检测。"""
    import cv2
    from segment_video import cached_hand_pose

    packed = cached_hand_pose(video_path, hand)
    if not packed or packed["n_detected"] <= 0:
        return None
    points = packed["xyz_raw"]
    detected = packed["detected"]
    start = int(span.get("start_frame", span.get("start", 0)))
    end = int(span.get("end_frame", span.get("end", start + 1)))
    total = int(points.shape[0])
    end = min(end, total)
    if end <= start:
        return None
    indices = np.linspace(start, max(start, end - 1), frames, dtype=int)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    rois = []
    skeletons = []
    try:
        for index in indices.tolist():
            if index >= total or not bool(detected[index]):
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
                skeletons.append(np.zeros((21, 3), dtype=np.float32))
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
                skeletons.append(np.zeros((21, 3), dtype=np.float32))
                continue
            height, width = frame.shape[:2]
            skeleton = points[index]
            skeletons.append(skeleton)
            xs = skeleton[:, 0] * width
            ys = skeleton[:, 1] * height
            side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0) * 1.6
            center_x = float((xs.min() + xs.max()) / 2)
            center_y = float((ys.min() + ys.max()) / 2)
            x1 = max(0, int(center_x - side / 2))
            y1 = max(0, int(center_y - side / 2))
            x2 = min(width, int(center_x + side / 2))
            y2 = min(height, int(center_y + side / 2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                rois.append(cv2.resize(crop, (224, 224)))
    finally:
        cap.release()
    if not rois:
        return None
    return np.stack(rois), np.stack(skeletons)


def _sample_hand_clip(video_path, span, hand, frames=24):
    cached = _crop_clip_from_cached_pose(video_path, span, hand, frames)
    if cached is not None:
        return cached
    import cv2
    import mediapipe as mp

    start = int(span.get("start_frame", span.get("start", 0)))
    end = int(span.get("end_frame", span.get("end", start + 1)))
    if end <= start:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end, total if total > 0 else end)
    indices = np.linspace(start, max(start, end - 1), frames, dtype=int)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from segment_video import choose_signer_hand_landmarks
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )
    rois = []
    skeletons = []
    try:
        for index in indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
                skeletons.append(np.zeros((21, 3), dtype=np.float32))
                continue
            height, width = frame.shape[:2]
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            chosen = choose_signer_hand_landmarks(
                result.multi_hand_landmarks, result.multi_handedness, hand
            )
            if chosen is None and result.multi_hand_landmarks:
                chosen = result.multi_hand_landmarks[0]
            if chosen is None:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
                skeletons.append(np.zeros((21, 3), dtype=np.float32))
                continue
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in chosen.landmark],
                dtype=np.float32,
            )
            skeletons.append(points)
            xs = points[:, 0] * width
            ys = points[:, 1] * height
            side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0) * 1.6
            center_x = float((xs.min() + xs.max()) / 2)
            center_y = float((ys.min() + ys.max()) / 2)
            x1 = max(0, int(center_x - side / 2))
            y1 = max(0, int(center_y - side / 2))
            x2 = min(width, int(center_x + side / 2))
            y2 = min(height, int(center_y + side / 2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                rois.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                rois.append(cv2.resize(crop, (224, 224)))
    finally:
        cap.release()
        hands.close()
    return np.stack(rois), np.stack(skeletons)


def _infer_unseen_handshape(
    video_path,
    span,
    sample_id,
    event_id,
    signer_id,
    video_id,
    hand,
):
    """查表没有这条无编号视频时，用已有手形权重对当前片段推理。不改写语料预测表。"""
    loaded = _load_handshape_model()
    if loaded is None:
        return None
    clip = _sample_hand_clip(video_path, span, hand)
    if clip is None:
        return None
    rois, skeletons = clip
    if not np.any(skeletons):
        return None
    model, idx_to_label, roi_to_tensor = loaded
    import torch
    from runtime_utils import probs_topk

    batch = {
        "roi": roi_to_tensor(rois).unsqueeze(0),
        "skeleton": torch.from_numpy(skeletons.astype(np.float32)).unsqueeze(0),
    }
    with torch.no_grad():
        logits = model(batch)["logits"].detach().cpu()
    topk_idx, topk_prob = probs_topk(logits, k=3, temperature=CALIBRATION_TEMPERATURE)
    topk = []
    for column in range(topk_idx.shape[1]):
        label = idx_to_label.get(int(topk_idx[0, column]))
        if label:
            topk.append({"label": label, "prob": float(topk_prob[0, column])})
    if not topk:
        return None
    item = {
        "sample_id": sample_id,
        "event_id": event_id,
        "signer_id": signer_id,
        "video_id": video_id,
        "hand": hand,
        "span": span,
        "prediction": {"label": topk[0]["label"], "conf": topk[0]["prob"], "topk": topk},
        "topk": topk,
    }
    print(f"[INFO] Handshape 无编号视频推理: {event_id} -> {topk[0]['label']}")
    return {
        "channel": "handshape",
        "source": "neural_inference",
        "split": "test",
        "schema": "handshape_pred_v1",
        "items": [item],
        "status": "ok",
    }


def run_handshape(
    split: str = "test",
    sample_id: Optional[str] = None,
    event_id: Optional[str] = None,
    signer_id: Optional[str] = None,
    video_id: Optional[str] = None,
    hand: Optional[str] = None,
    force_inference: bool = False,
    video_path: Optional[str] = None,
    span: Optional[Dict[str, Any]] = None,
    allow_unnumbered_inference: bool = False,
) -> Dict[str, Any]:
    """
    多模型融合版：
    1. 如果有已存在的融合结果，直接返回
    2. 否则循环推理所有可用模型，融合后保存
    """

    # ---------- 辅助：从 JSON 文件加载结果 ----------
    def _load_existing(json_path: Path) -> Optional[Dict[str, Any]]:
        if not json_path.exists():
            return None
        try:
            with json_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            with json_path.open("r", encoding="gbk", errors="replace") as f:
                raw = json.load(f)

        if isinstance(raw, dict) and "items" in raw:
            items = raw["items"]
            schema = raw.get("schema", "handshape_pred_v1")
        elif isinstance(raw, dict) and "result" in raw:
            items = raw["result"].get("items", [])
            schema = raw["result"].get("schema", "handshape_pred_v1")
        elif isinstance(raw, list):
            items = raw
            schema = "handshape_pred_v1"
        else:
            return None

        items = [_calibrate_item(item) for item in items]
        base = {
            "channel": "handshape",
            "source": "existing_prediction",
            "split": split,
            "schema": schema,
            "items": items,
            "pred_json": str(json_path),
        }

        def _matches(it: Dict[str, Any]) -> bool:
            item_sample_id = str(it.get("sample_id", ""))
            item_hand = str(it.get("hand", "")).lower()
            wanted_hand = hand.lower() if hand else None
            if item_hand == "l":
                item_hand = "left"
            elif item_hand == "r":
                item_hand = "right"
            return (
                (sample_id is None or item_sample_id == sample_id)
                and (event_id is None or it.get("event_id") == event_id)
                and (signer_id is None or item_sample_id.startswith(f"{signer_id}_"))
                and (video_id is None or it.get("video_id") == video_id)
                and (wanted_hand is None or item_hand == wanted_hand)
            )

        filtered = [it for it in items if _matches(it)]
        if (
            not filtered
            and signer_id
            and video_id
            and event_id
            and hand
        ):
            wanted_hand = hand.lower()
            references = []
            for item in items:
                item_hand = str(item.get("hand", "")).lower()
                item_hand = {"l": "left", "r": "right"}.get(
                    item_hand, item_hand
                )
                if (
                    item.get("video_id") == video_id
                    and item.get("event_id") == event_id
                    and item_hand == wanted_hand
                ):
                    references.append(item)
            if references:
                filtered = [
                    _aggregate_reference_items(
                        references,
                        signer_id=signer_id,
                        video_id=video_id,
                        event_id=event_id,
                        hand=hand,
                    )
                ]
                base["source"] = "cross_signer_fallback"
        return {
            **base,
            "items": filtered,
            "status": "ok" if filtered else "sample_not_found",
        }

    # ---------- 先尝试读已有融合结果 ----------
    existing = _load_existing(OUT_JSON)
    if allow_unnumbered_inference:
        if existing is not None and existing.get("status") == "ok":
            return existing
        inferred = None
        if video_path and span:
            try:
                inferred = _infer_unseen_handshape(
                    video_path=video_path,
                    span=span,
                    sample_id=sample_id,
                    event_id=event_id,
                    signer_id=signer_id,
                    video_id=video_id,
                    hand=hand,
                )
            except Exception as exc:
                print(f"[WARN] Handshape 无编号视频推理失败: {exc}")
        if inferred is not None:
            return inferred
        if existing is not None:
            return existing
        return {
            "channel": "handshape",
            "source": "unavailable",
            "status": "sample_not_found",
            "items": [],
        }
    if existing is not None and not force_inference:
        return existing

    # ---------- 获取可用模型 ----------
    models = _get_available_models()
    if len(models) == 0:
        return {
            "channel": "handshape",
            "source": "unavailable",
            "status": "no_models_found",
            "runs_dir": str(RUNS_DIR),
        }

    print(f"[INFO] Handshape 融合 {len(models)} 个模型:")
    for m in models:
        print(f"  - {m.name}")

    if not CONFIG.exists():
        return {
            "channel": "handshape",
            "source": "unavailable",
            "status": "config_not_found",
            "config_path": str(CONFIG),
        }

    # 仅在确实需要神经网络推理时加载 torch 及模型依赖。
    # 读取已有预测不应要求本机 CUDA/PyTorch DLL 可用。
    from infer import main as handshape_infer_main

    # ---------- 逐个模型推理 ----------
    all_model_results = []  # 每个元素是一个 {sample_id: item} 的 dict

    for model_dir in models:
        ckpt = model_dir / "best.ckpt"
        if not ckpt.exists():
            print(f"  [WARN] 跳过 {model_dir.name}（无 best.ckpt）")
            continue

        # 临时输出路径
        tmp_json = ROOT / f"test_pred_{model_dir.name}.json"
        tmp_csv = ROOT / f"test_predictions_{model_dir.name}.csv"

        # 保存旧 sys.argv
        _old_argv = sys.argv.copy()
        sys.argv = [
            "infer.py",
            "--config", str(CONFIG),
            "--ckpt", str(ckpt),
            "--split", split,
            "--out_csv", str(tmp_csv),
            "--out_predjson", str(tmp_json),
        ]

        try:
            print(f"  [RUN] {model_dir.name} ...")
            handshape_infer_main()
        except SystemExit:
            pass
        except Exception as e:
            print(f"  [ERR] {model_dir.name} 推理失败: {e}")
        finally:
            sys.argv = _old_argv

        # 读取该模型的结果
        result = _load_existing(tmp_json)
        if result is not None and "items" in result:
            model_items = {}
            for item in result["items"]:
                sid = item.get("sample_id")
                if sid:
                    model_items[sid] = item
            all_model_results.append(model_items)
            print(f"  [OK] {model_dir.name}: {len(model_items)} 个样本")
        else:
            print(f"  [WARN] {model_dir.name}: 未生成有效结果")

    # ---------- 融合 ----------
    if len(all_model_results) == 0:
        return {
            "channel": "handshape",
            "source": "unavailable",
            "status": "all_models_failed",
        }

    # 收集所有 sample_id
    all_sample_ids = set()
    for mr in all_model_results:
        all_sample_ids.update(mr.keys())

    fused_items = []

    for sid in sorted(all_sample_ids):
        # 收集该 sample 在所有模型中的 topk 概率
        # 结构: {label: [prob_from_model_1, prob_from_model_2, ...]}
        label_probs = {}

        for mr in all_model_results:
            if sid not in mr:
                continue
            item = mr[sid]
            # 获取 topk 概率列表
            topk = item.get("topk", [])
            if not topk:
                # 如果没有 topk，尝试从 prediction 字段获取
                pred = item.get("prediction", {})
                if "label" in pred and "conf" in pred:
                    label = pred["label"]
                    conf = pred["conf"]
                    label_probs.setdefault(label, []).append(conf)

            for tk in topk:
                label = tk.get("label", "")
                prob = tk.get("prob", 0.0)
                if label:
                    label_probs.setdefault(label, []).append(prob)

        if not label_probs:
            # 跳过无法融合的样本
            continue

        # 平均每个 label 的概率
        avg_probs = {}
        for label, probs in label_probs.items():
            avg_probs[label] = sum(probs) / len(probs)

        # 排序取 top3
        sorted_labels = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_labels[:3]

        # 归一化（确保加起来=1）
        total = sum(p for _, p in top3)
        if total > 0:
            top3_normalized = [(l, p / total) for l, p in top3]
        else:
            top3_normalized = top3

        # 构建融合后的 item（以第一个模型的结果为基础模板）
        base_item = None
        for mr in all_model_results:
            if sid in mr:
                base_item = mr[sid].copy()
                break

        if base_item is None:
            continue

        # 更新 prediction 字段
        fused_pred = {
            "label": top3_normalized[0][0] if top3_normalized else "?",
            "conf": top3_normalized[0][1] if top3_normalized else 0.0,
            "topk": [
                {"label": l, "prob": p}
                for l, p in top3_normalized
            ],
        }
        base_item["prediction"] = fused_pred
        base_item["topk"] = fused_pred["topk"]

        fused_items.append(base_item)

    # ---------- 保存融合结果 ----------
    fused_output = {
        "schema": "handshape_pred_v1",
        "channel": "handshape",
        "source": "ensemble_fusion",
        "models_used": [m.name for m in models],
        "num_models": len(all_model_results),
        "items": fused_items,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(fused_output, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] 融合完成: {len(fused_items)} 个样本 → {OUT_JSON}")

    # ---------- 返回结果 ----------
    base = {
        "channel": "handshape",
        "source": "ensemble_fusion",
        "split": split,
        "schema": "handshape_pred_v1",
        "items": fused_items,
        "pred_json": str(OUT_JSON),
        "models_used": [m.name for m in models],
        "num_models": len(all_model_results),
    }

    filtered = _load_existing(OUT_JSON)
    return filtered if filtered is not None else base


# =======================
# CLI 入口
# =======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Handshape 多模型融合推理接口")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_id", help="指定 sample_id（可选）")
    parser.add_argument("--event_id")
    parser.add_argument("--signer_id")
    parser.add_argument("--video_id")
    parser.add_argument("--hand", choices=["left", "right"])
    parser.add_argument("--force_inference", action="store_true")
    args = parser.parse_args()

    result = run_handshape(
        split=args.split,
        sample_id=args.sample_id,
        event_id=args.event_id,
        signer_id=args.signer_id,
        video_id=args.video_id,
        hand=args.hand,
        force_inference=args.force_inference,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))