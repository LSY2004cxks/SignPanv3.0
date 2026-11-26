#!/usr/bin/env python3
"""
segment_video.py — 基于 BoundaryTCN 的单视频切分模块（最终修正版）

✅ 动态导入 boundary_prefill_B_final 的真实 BoundaryTCN
✅ 用完立即还原 sys.path，不污染其他通道
用法：
    python segment_video.py --video test.mp4 --checkpoint boundary_prefill_B_final/logs/run_20260228_210024/best.pt
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# ============================================================
# 路径配置
# ============================================================
DEFAULT_CHECKPOINT = Path(
    "boundary_prefill_B_final/logs/run_20260228_210024/best.pt"
)
_BOUNDARY_ROOT = Path(__file__).resolve().parent / "boundary_prefill_B_final"

# 延迟加载的模型引用
_BOUNDARY: Optional[Tuple[type, object]] = None


def mediapipe_label_for_signer_hand(hand: str) -> str:
    """MediaPipe 按自拍镜像分左右。正面机位对调后，才是打字人自己的手。"""
    if str(hand or "").lower().startswith("l"):
        return "Right"
    return "Left"


def choose_signer_hand_landmarks(multi_hand_landmarks, multi_handedness, signer_hand: str):
    """同一侧有多只候选时，留下置信度更高的那只。"""
    target = mediapipe_label_for_signer_hand(signer_hand)
    best = None
    best_score = -1.0
    if not multi_hand_landmarks or not multi_handedness:
        return None
    for landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        classification = handedness.classification[0]
        if str(classification.label) != target:
            continue
        score = float(classification.score)
        if score > best_score:
            best = landmarks
            best_score = score
    return best


def _load_boundary():
    r"""
    动态导入 BoundaryTCN + decode_boundaries。
    关键：把 BOUNDARY_ROOT 的父目录（项目根目录）加入 sys.path，
    使其可被解析为 boundary_prefill_B_final.src.models.tcn
    """
    global _BOUNDARY
    if _BOUNDARY is not None:
        return _BOUNDARY

    parent = str(_BOUNDARY_ROOT.parent)
    added = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added = True

    try:
        from boundary_prefill_B_final.src.models.tcn import BoundaryTCN
        from boundary_prefill_B_final.src.infer.postprocess import decode_boundaries
        _BOUNDARY = (BoundaryTCN, decode_boundaries)
    finally:
        # 用完立即清理，避免遮蔽其他通道的 src
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent)

    return _BOUNDARY


# ============================================================
# 特征构建：23 维（基于 FeatureBuilder 构建）
# ============================================================
def build_pose_features(xyz: np.ndarray, median_k: int = 5) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim == 2 and xyz.shape[1] == 63:
        xyz = xyz.reshape(-1, 21, 3)
    if xyz.ndim != 3 or xyz.shape[1:] != (21, 3):
        raise ValueError(f"期望 (T, 21, 3)，得到 {xyz.shape}")

    T = xyz.shape[0]
    if T == 0:
        return np.zeros((0, 23), dtype=np.float32)

    feat = np.zeros((T, 23), dtype=np.float32)

    # 1-3: 质心
    feat[:, 0:3] = xyz.mean(axis=1)
    # 4-6: 标准差
    feat[:, 3:6] = xyz.std(axis=1) + 1e-8
    # 7-11: 指尖到手腕距离 (5个)
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        feat[:, 6 + i] = np.linalg.norm(xyz[:, tip, :] - xyz[:, 0, :], axis=1)
    # 12-13: 手掌宽高比
    palm_w = np.linalg.norm(xyz[:, 5, :2] - xyz[:, 17, :2], axis=1)
    palm_h = np.linalg.norm(xyz[:, 0, :2] - xyz[:, 9, :2], axis=1)
    feat[:, 11] = palm_w / (palm_h + 1e-8)
    feat[:, 12] = palm_h / (palm_w + 1e-8)

    # 14-16: 速度 (一阶差分)
    if T > 1:
        feat[1:, 13:16] = np.diff(xyz, axis=0).mean(axis=1)
    feat[0, 13:16] = feat[1, 13:16] if T > 1 else 0

    # 17-19: 加速度 (速度的差分) ← 修复这里
    if T > 2:
        feat[2:, 16:19] = np.diff(feat[1:, 13:16], axis=0)  # 输出 (T-2, 3)，刚好匹配
    feat[:2, 16:19] = feat[2, 16:19] if T > 2 else 0

    # 20-21: 手掌方向向量 (归一化)
    vec = xyz[:, 9, :2] - xyz[:, 0, :2]
    n = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    feat[:, 19:21] = vec / n
    # 22-23: 手腕坐标
    feat[:, 21:23] = xyz[:, 0, :2]

    if median_k > 1:
        try:
            from scipy.ndimage import median_filter
            for d in range(feat.shape[1]):
                feat[:, d] = median_filter(feat[:, d], size=median_k)
        except ImportError:
            pass
    return feat

# ============================================================
# 模型加载（基于 BoundaryTCN + model_cfg）
# ============================================================
def load_boundary_model(checkpoint_path: str, device: str = "cpu"):
    BoundaryTCN, _ = _load_boundary()

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        model_cfg = ckpt.get("model_cfg", {})
        input_dim = int(ckpt.get("input_dim", model_cfg.get("pose_dim", 23)))
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        model_cfg = ckpt.get("config", ckpt.get("model_cfg", {}))
        input_dim = int(ckpt.get("input_dim", model_cfg.get("pose_dim", 23)))
    else:
        raise RuntimeError(f"无法解析 checkpoint: {checkpoint_path}")

    # 用 model_cfg 里的字段名（pose_dim / fusion_mode 等）构建模型
    model = BoundaryTCN(
        input_dim=input_dim,
        channels=int(model_cfg.get("channels", 128)),
        layers=int(model_cfg.get("layers", 4)),
        dilations=list(model_cfg.get("dilations", [1, 2, 4, 8])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        fusion_mode=str(model_cfg.get("fusion_mode", "early")),
        pose_dim=int(model_cfg.get("pose_dim", 23)),
        flow_dim=int(model_cfg.get("flow_dim", 6)),
        fusion_dim=int(model_cfg.get("fusion_dim", 64)),
        gate_hidden=int(model_cfg.get("gate_hidden", 64)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device


# ============================================================
# 核心切分函数
# ============================================================
def segment_video(pose_xyz, checkpoint_path=DEFAULT_CHECKPOINT,
                  tau=0.5, device="cpu"):
    _, decode_boundaries = _load_boundary()
    checkpoint_path = str(checkpoint_path)
    model, device = load_boundary_model(checkpoint_path, device)

    features = build_pose_features(pose_xyz)
    if features.shape[0] == 0:
        return []

    x = np.expand_dims(features.astype(np.float32), axis=0)  # (1, T, 23)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))[0].cpu().numpy()

    p = 1.0 / (1.0 + np.exp(-logits))
    decoded = decode_boundaries(
        p_start=p[:, 0], p_end=p[:, 1], p_inside=p[:, 2],
        tau=tau, peak_radius=7, nms_window=7,
        Lmin=6, Lmax=80, inside_recommend_thr=0.5,
        pair_mode="score_then_nearest",
    )
    return decoded.get("spans_recommended", [])


# ============================================================
# 视频文件 -> 关键点 -> 切分
# ============================================================
def _fill_nearest(xyz, detected):
    T = xyz.shape[0]
    valid = np.where(detected)[0]
    if len(valid) == 0:
        return xyz
    out = xyz.copy()
    for t in range(T):
        if not detected[t]:
            out[t] = out[int(valid[np.argmin(np.abs(valid - t))])]
    return out


_POSE_CACHE = {}


def _hand_pose_from_video(video_path, hand="right", fill_missing=True):
    """提取单手关键点。同一进程里同一文件的第二次切分直接复用。"""
    import mediapipe as mp

    path = Path(video_path)
    key = (str(path.resolve()), str(hand).lower(), path.stat().st_mtime_ns, bool(fill_missing))
    cached = _POSE_CACHE.get(key)
    if cached is not None:
        print(f"[INFO] 复用已提取的{hand}手关键点")
        return cached

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 视频总帧数: {n_frames}, FPS: {fps}")

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    all_xyz, detected_flags = [], []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            all_xyz.append(np.zeros((21, 3), dtype=np.float32))
            detected_flags.append(False)
            continue
        res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        chosen = choose_signer_hand_landmarks(
            res.multi_hand_landmarks, res.multi_handedness, hand
        )
        if res.multi_hand_landmarks:
            if chosen is None:
                all_xyz.append(np.zeros((21, 3), dtype=np.float32))
                detected_flags.append(False)
                continue
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in chosen.landmark], dtype=np.float32)
            all_xyz.append(xyz)
            detected_flags.append(True)
        else:
            all_xyz.append(np.zeros((21, 3), dtype=np.float32))
            detected_flags.append(False)

    cap.release()
    mp_hands.close()

    raw_xyz = np.stack(all_xyz, axis=0)
    detected = np.array(detected_flags, dtype=bool)
    xyz_seq = _fill_nearest(raw_xyz, detected) if fill_missing else raw_xyz

    n_detected = int(detected.sum())
    print(f"[INFO] 关键点序列: {xyz_seq.shape}, 有效帧: {n_detected}/{n_frames}")
    packed = {
        "xyz": xyz_seq,
        "xyz_raw": raw_xyz,
        "detected": detected,
        "n_frames": n_frames,
        "fps": fps,
        "n_detected": n_detected,
    }
    _POSE_CACHE[key] = packed
    return packed


def cached_hand_pose(video_path, hand="right"):
    """返回这一进程里已经抽过的手部关键点。没有抽过就是 None。"""
    path = Path(video_path)
    key = (str(path.resolve()), str(hand).lower(), path.stat().st_mtime_ns, True)
    return _POSE_CACHE.get(key)


def preload_unnumbered_poses(video_path, max_width=960):
    """无编号视频只读一遍，左右手一起检测。画面先缩小，坐标仍是 0–1。"""
    path = Path(video_path)
    if cached_hand_pose(path, "left") is not None and cached_hand_pose(path, "right") is not None:
        print("[INFO] 复用已提取的左右手关键点")
        return

    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 无编号视频一次抽取左右手: {n_frames} 帧, FPS: {fps}")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.35, min_tracking_confidence=0.35,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    raw = {
        "left": [],
        "right": [],
    }
    flags = {"left": [], "right": []}
    wrist_pose = {
        "left": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "right": np.full((n_frames, 2), np.nan, dtype=np.float32),
    }
    shoulder_y = np.full(n_frames, np.nan, dtype=np.float32)
    hip_y = np.full(n_frames, np.nan, dtype=np.float32)
    body_kpts = []
    body_scores = []
    orig_w = orig_h = 0

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            for hand in ("left", "right"):
                raw[hand].append(np.zeros((21, 3), dtype=np.float32))
                flags[hand].append(False)
            body_kpts.append(np.zeros((17, 2), dtype=np.float32))
            body_scores.append(np.zeros(17, dtype=np.float32))
            continue
        orig_h, orig_w = frame.shape[:2]
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(frame.shape[0] * scale))),
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pres = pose.process(rgb)
        result = hands.process(rgb)

        pw = {}
        if pres.pose_landmarks:
            plm = pres.pose_landmarks.landmark
            body_kpts.append(np.array(
                [[p.x * orig_w, p.y * orig_h] for p in plm[:17]],
                dtype=np.float32,
            ))
            body_scores.append(np.array(
                [p.visibility for p in plm[:17]],
                dtype=np.float32,
            ))
            if plm[11].visibility > 0.3 and plm[12].visibility > 0.3:
                shoulder_y[frame_idx] = 0.5 * (plm[11].y + plm[12].y)
            if plm[23].visibility > 0.3 and plm[24].visibility > 0.3:
                hip_y[frame_idx] = 0.5 * (plm[23].y + plm[24].y)
            if plm[15].visibility > 0.35:
                pw["left"] = np.array([plm[15].x, plm[15].y], dtype=np.float32)
                wrist_pose["left"][frame_idx] = pw["left"]
            if plm[16].visibility > 0.35:
                pw["right"] = np.array([plm[16].x, plm[16].y], dtype=np.float32)
                wrist_pose["right"][frame_idx] = pw["right"]
        else:
            body_kpts.append(np.zeros((17, 2), dtype=np.float32))
            body_scores.append(np.zeros(17, dtype=np.float32))

        if result.multi_hand_landmarks:
            cands = []
            handedness_list = result.multi_handedness or [None] * len(result.multi_hand_landmarks)
            for lm, hd in zip(result.multi_hand_landmarks, handedness_list):
                arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                lbl = hd.classification[0].label if hd and getattr(hd, "classification", None) else ""
                sc = float(hd.classification[0].score) if hd and getattr(hd, "classification", None) else 0.5
                cands.append((arr, lbl, sc))
            used = set()
            for hand in ("right", "left"):
                target_lbl = mediapipe_label_for_signer_hand(hand)
                best_i = None
                best_cost = 999.0
                for idx_c, (arr, lbl, sc) in enumerate(cands):
                    if idx_c in used:
                        continue
                    w_xy = arr[0, :2]
                    if hand in pw:
                        d_own = float(np.linalg.norm(w_xy - pw[hand]))
                        other = "left" if hand == "right" else "right"
                        d_other = float(np.linalg.norm(w_xy - pw[other])) if other in pw else 0.25
                        if d_other < 0.08 and d_own > d_other + 0.08:
                            continue
                        if d_own > 0.22:
                            continue
                        lbl_penalty = 0.0 if lbl.lower() == target_lbl.lower() else 0.08
                        cost = d_own + lbl_penalty - 0.02 * sc
                    else:
                        if lbl.lower() != target_lbl.lower():
                            continue
                        cost = 0.15 - 0.02 * sc
                    if cost < best_cost:
                        best_cost = cost
                        best_i = idx_c
                if best_i is not None:
                    used.add(best_i)
                    raw[hand].append(cands[best_i][0])
                    flags[hand].append(True)
                else:
                    raw[hand].append(np.zeros((21, 3), dtype=np.float32))
                    flags[hand].append(False)
        else:
            for hand in ("left", "right"):
                raw[hand].append(np.zeros((21, 3), dtype=np.float32))
                flags[hand].append(False)
    cap.release()
    hands.close()
    pose.close()

    fast_body = {
        "pose_kpts": np.stack(body_kpts) if body_kpts else np.zeros((0, 17, 2), np.float32),
        "pose_scores": np.stack(body_scores) if body_scores else np.zeros((0, 17), np.float32),
        "width": orig_w,
        "height": orig_h,
    }
    for hand in ("left", "right"):
        raw_xyz = np.stack(raw[hand], axis=0) if raw[hand] else np.zeros((0, 21, 3), np.float32)
        detected = np.array(flags[hand], dtype=bool)
        xyz_seq = _fill_nearest(raw_xyz, detected) if len(detected) else raw_xyz
        key = (str(path.resolve()), hand, path.stat().st_mtime_ns, True)
        _POSE_CACHE[key] = {
            "xyz": xyz_seq,
            "xyz_raw": raw_xyz,
            "detected": detected,
            "wrist_pose": wrist_pose[hand],
            "shoulder_y": shoulder_y,
            "hip_y": hip_y,
            "fast_body": fast_body,
            "n_frames": n_frames,
            "fps": fps,
            "n_detected": int(detected.sum()),
        }
        print(f"[INFO] {hand}手关键点: {xyz_seq.shape}, 有效帧: {int(detected.sum())}/{n_frames}")


def segment_from_video_file(video_path, checkpoint_path=DEFAULT_CHECKPOINT,
                            hand="right", tau=0.5, device="cpu",
                            fill_missing=True, span_fn=None):
    packed = _hand_pose_from_video(video_path, hand=hand, fill_missing=fill_missing)
    decode = span_fn or segment_video
    if packed["n_detected"] <= 0:
        segments = []
    elif span_fn is None:
        segments = decode(packed["xyz"], checkpoint_path, tau, device)
    else:
        import inspect
        extra = {}
        try:
            sig = inspect.signature(decode)
            for k in ("wrist_pose", "shoulder_y", "hip_y"):
                if k in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                ):
                    extra[k] = packed.get(k)
        except (TypeError, ValueError):
            pass
        segments = decode(
            packed["xyz"], checkpoint_path, tau, device,
            pose_raw=packed["xyz_raw"], detected=packed["detected"],
            **extra,
        )
    n_frames = packed["n_frames"]
    n_detected = packed["n_detected"]
    return {
        "video_path": str(video_path), "n_frames": n_frames, "fps": packed["fps"],
        "n_detected": n_detected, "detect_ratio": n_detected / max(n_frames, 1),
        "segments": segments,
    }


# ============================================================
# CLI 入口
# ============================================================
def main():
    p = argparse.ArgumentParser(description="BoundaryTCN 视频切分")
    p.add_argument("--video", required=True, help="视频路径")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="权重路径")
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--tau", type=float, default=0.5, help="阈值")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", default=None, help="输出 JSON")
    args = p.parse_args()

    result = segment_from_video_file(
        video_path=args.video, checkpoint_path=args.checkpoint,
        hand=args.hand, tau=args.tau, device=args.device,
    )

    print("\n" + "=" * 50)
    print(f"视频: {result['video_path']}")
    print(f"总帧数: {result['n_frames']}  有效帧: {result['n_detected']} ({result['detect_ratio']:.1%})")
    print(f"切分出 {len(result['segments'])} 个片段:")
    for i, seg in enumerate(result["segments"], 1):
        print(f"  [{i}] 帧 {seg['start']:>4} ~ {seg['end']:<4}  长度={seg['length']}  得分={seg['score']:.3f}")
    print("=" * 50)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"→ 已保存: {args.output}")


if __name__ == "__main__":
    main()