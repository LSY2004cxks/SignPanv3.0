from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================
# 1. MediaPipe 初始化
# ============================================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)


# ============================================================
# 2. 事件加载（只取当前视频的 event）
# ============================================================
import re

def _is_corpus_filename(video_path) -> bool:
    """语料文件名以打字员编号和视频编号结尾。0158.mp4 这类名字不是。"""
    parts = Path(video_path).stem.split("-")
    return (
        len(parts) >= 2
        and re.fullmatch(r"\d+", parts[-2]) is not None
        and re.fullmatch(r"\d+", parts[-1]) is not None
    )


def _parse_video_uid(video_path):
    """解析语料文件名；普通文件名使用稳定的合成身份，不再要求固定命名。"""
    import hashlib

    stem = Path(video_path).stem
    parts = stem.split("-")

    # 兼容原语料的 ...-signer-video 格式。
    has_corpus_suffix = (
        len(parts) >= 2
        and re.fullmatch(r"\d+", parts[-2]) is not None
        and re.fullmatch(r"\d+", parts[-1]) is not None
    )
    if has_corpus_suffix:
        signer_id = f"S{parts[-2]}"
        video_id = f"V{parts[-1]}"
    else:
        # 新视频可以使用任意名称。数字哈希保持下游 S\d+ / V\d+ 接口兼容，
        # 同名文件也会得到相同身份，便于重新上传后关联人工修订。
        digest = hashlib.sha256(stem.encode("utf-8")).digest()
        synthetic_number = int.from_bytes(digest[:8], "big")
        signer_id = "S000"
        video_id = f"V{synthetic_number}"
        print(
            f"[INFO] 视频 {stem} 使用合成身份 "
            f"(signer={signer_id}, video={video_id})"
        )

    video_uid = f"{signer_id}_{video_id}"  # S142_V6326
    return video_uid, signer_id, video_id


def load_video_events(video_path, events_jsonl):
    """加载当前视频对应的 event 标注"""
    video_path = Path(video_path)

    # [OK] 用统一解析函数
    video_uid, signer_id, video_id = _parse_video_uid(video_path)

    print(f"[INFO] 解析结果: video_uid={video_uid} (signer={signer_id}, video={video_id})")

    events_path = Path(events_jsonl)
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    if not events_path.exists():
        print(f"[WARN] 标注文件不存在，跳过标注回退: {events_path}")
        return [], video_uid

    # [OK] 读取标注库
    with open(events_path, "r", encoding="utf-8") as f:
        all_events = [json.loads(line) for line in f if line.strip()]

    # [OK] 匹配策略：先按 video_uid，再按 video_id
    events = [ev for ev in all_events if ev.get("video_uid") == video_uid]
    if not events:
        events = [ev for ev in all_events if ev.get("video_id") == video_id]

    print(f"[INFO] 匹配到 {len(events)} 个 event")

    if not events:
        print(f"[WARN] 未找到任何 event！请确认标注库中有 {video_uid} 的记录")

    return events, video_uid

# ============================================================
# 3. 抽帧
# ============================================================
def extract_frames(video_path: str, start: int, end: int, num: int) -> tuple[list, list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end, total)
    indices = np.linspace(start, end - 1, num, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, indices


def orientation_from_landmarks(
    landmarks: list[np.ndarray],
    hand: str,
    event_id: str,
    signer_id: str,
    video_id: str,
    span: dict,
) -> dict:
    """无模型权重时，以 MediaPipe 手掌法向量给出可审计的几何回退结果。"""
    if not landmarks:
        return {
            "channel": "orientation",
            "source": "unavailable",
            "status": "no_hand_landmarks",
            "error": "MediaPipe 未检测到足够的手部关键点",
            "items": [],
        }

    xyz = np.stack(landmarks).astype(np.float32)
    v1 = xyz[:, 5] - xyz[:, 0]
    v2 = xyz[:, 17] - xyz[:, 0]
    normals = np.cross(v1, v2)
    if hand == "left":
        normals = -normals
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return {
            "channel": "orientation",
            "source": "unavailable",
            "status": "invalid_palm_normal",
            "error": "手掌法向量无法计算",
            "items": [],
        }
    normals = normals[valid] / lengths[valid, None]
    normal = np.median(normals, axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)

    diagonal = float(1.0 / np.sqrt(2.0))
    prototypes = {
        "TU": np.array([0.0, -1.0, 0.0]),
        "TD": np.array([0.0, 1.0, 0.0]),
        "TL": np.array([-1.0, 0.0, 0.0]),
        "TR": np.array([1.0, 0.0, 0.0]),
        "TLU": np.array([-diagonal, -diagonal, 0.0]),
        "TRU": np.array([diagonal, -diagonal, 0.0]),
        "TLD": np.array([-diagonal, diagonal, 0.0]),
        "TRD": np.array([diagonal, diagonal, 0.0]),
        "AFB": np.array([0.0, 0.0, 1.0]),
        "TB": np.array([0.0, 0.0, -1.0]),
    }
    labels = list(prototypes)
    scores = np.array([float(normal @ prototypes[label]) for label in labels])
    probabilities = np.exp((scores - scores.max()) * 4.0)
    probabilities /= probabilities.sum()
    order = np.argsort(-probabilities)[:3]
    top_labels = [labels[index] for index in order]
    top_probs = [float(probabilities[index]) for index in order]
    return {
        "channel": "orientation",
        "source": "geometric_fallback",
        "schema": "orientation_pred_v1",
        "items": [
            {
                "sample_id": event_id,
                "signer_id": signer_id,
                "video_id": video_id,
                "hand": hand,
                "event_id": event_id,
                "span": span,
                "orientation": top_labels[0],
                "top3_labels": top_labels,
                "top3_probs": top_probs,
                "conf": top_probs[0],
                "margin": top_probs[0] - top_probs[1],
                "hitl_gate": "recommend_top3",
                "quality_flags": ["GEOMETRIC_FALLBACK"],
            }
        ],
    }


# ============================================================
# 4. 提取四通道特征并保存 npz
# ============================================================
def _scan_body_pose(video_path, max_width=960):
    """身体关键点只扫一遍。若 preload_unnumbered_poses 已提取则直接复用。"""
    from segment_video import cached_hand_pose
    cached = cached_hand_pose(video_path, "right") or cached_hand_pose(video_path, "left")
    if cached and cached.get("fast_body") is not None:
        print(f"[INFO] 复用已提取的身体关键点: {cached.get('n_frames')} 帧")
        return cached["fast_body"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_kpts, pose_scores = [], []
    width = height = 0
    print(f"[INFO] 无编号视频一次抽取身体关键点: {n_frames} 帧")
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            pose_kpts.append(np.zeros((17, 2), dtype=np.float32))
            pose_scores.append(np.zeros(17, dtype=np.float32))
            continue
        height, width = frame.shape[:2]
        small = frame
        if width > max_width:
            scale = max_width / width
            small = cv2.resize(frame, (max_width, max(1, int(height * scale))))
        pres = mp_pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if pres.pose_landmarks:
            landmarks = pres.pose_landmarks.landmark[:17]
            pose_kpts.append(np.array(
                [[point.x * width, point.y * height] for point in landmarks],
                dtype=np.float32,
            ))
            pose_scores.append(np.array(
                [point.visibility for point in landmarks],
                dtype=np.float32,
            ))
        else:
            pose_kpts.append(np.zeros((17, 2), dtype=np.float32))
            pose_scores.append(np.zeros(17, dtype=np.float32))
    cap.release()
    return {
        "pose_kpts": np.stack(pose_kpts),
        "pose_scores": np.stack(pose_scores),
        "width": width,
        "height": height,
    }


ORIENTATION_FRAMES = 24


def _palm_normals(landmarks: np.ndarray, hand: str) -> np.ndarray:
    channel_root = ROOT / "orientation_channel"
    if str(channel_root) not in sys.path:
        sys.path.insert(0, str(channel_root))
    from src.data.dataset import compute_palm_normal, normalize_landmarks

    xyz, _mask = normalize_landmarks(landmarks)
    return compute_palm_normal(xyz, hand)


def _save_orientation_clip(out_dir: Path, eid: str, hand: str, roi_list, landmark_list, frame_indices) -> Path:
    landmarks = np.stack(landmark_list).astype(np.float32)
    ori_dir = out_dir / "orientation"
    ori_dir.mkdir(parents=True, exist_ok=True)
    ori_path = ori_dir / f"{eid}.npz"
    np.savez(
        ori_path,
        frames=np.stack(roi_list),
        roi=np.stack(roi_list),
        landmarks=landmarks,
        palm_normal=_palm_normals(landmarks, hand),
        frame_indices=np.array(frame_indices, dtype=np.int64),
        event_id=np.array(eid, dtype="<U20"),
        hand=np.array(hand, dtype="<U10"),
    )
    return ori_path


def _sample_indices(start: int, end: int, count: int, total: int) -> list[int]:
    end = min(end, total)
    if end <= start:
        return []
    return np.linspace(start, end - 1, count, dtype=int).tolist()


ORIENTATION_FRAMES = 24


def _palm_normals(landmarks: np.ndarray, hand: str) -> np.ndarray:
    from orientation_channel.scripts.run_orientation import palm_normal_from_landmarks

    return palm_normal_from_landmarks(landmarks, hand)


def _save_orientation_npz(out_dir: Path, eid: str, hand: str, roi_list, landmark_list, frame_indices) -> Path:
    landmarks = np.stack(landmark_list).astype(np.float32)
    frames = np.stack(roi_list)
    ori_dir = out_dir / "orientation"
    ori_dir.mkdir(parents=True, exist_ok=True)
    ori_path = ori_dir / f"{eid}.npz"
    np.savez(
        ori_path,
        frames=frames,
        roi=frames,
        landmarks=landmarks,
        palm_normal=_palm_normals(landmarks, hand),
        frame_indices=np.array(frame_indices, dtype=np.int64),
        event_id=np.array(eid, dtype="<U20"),
        hand=np.array(hand, dtype="<U10"),
    )
    return ori_path


def _cached_hand_arrays(video_path, hand: str):
    from segment_video import cached_hand_pose

    packed = cached_hand_pose(video_path, hand)
    if not packed:
        return None
    return packed["xyz_raw"], packed["detected"]


def process_event(video_path: Path, ev: dict, out_dir: Path, fast_body=None) -> dict[str, object]:
    if fast_body is not None:
        return _process_event_from_cache(video_path, ev, out_dir, fast_body)
    eid = ev["event_id"]
    hand = ev.get("hand", "right")
    span = ev["span"]
    start, end = int(span["start_frame"]), int(span["end_frame"])

    _, signer_id, video_id = _parse_video_uid(video_path)
    from segment_video import choose_signer_hand_landmarks

    # --- Handshape: (8, 21, 3) ---
    hs_frames, hs_idx = extract_frames(str(video_path), start, end, 8)
    kpts_list, bboxes_list, sides_list, scores_list, lq_list = [], [], [], [], []
    for frame in hs_frames:
        res = mp_hands.process(frame)
        if res.multi_hand_landmarks:
            chosen = choose_signer_hand_landmarks(
                res.multi_hand_landmarks, res.multi_handedness, hand
            )
            if chosen is None:
                chosen = res.multi_hand_landmarks[0]
            kpts = np.array([[lm.x, lm.y, lm.z] for lm in chosen.landmark], dtype=np.float32)
            h_, w_ = frame.shape[:2]
            xs = kpts[:, 0] * w_; ys = kpts[:, 1] * h_
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            m = 0.1
            bbox = np.array([max(0, x1-(x2-x1)*m), max(0, y1-(y2-y1)*m),
                             min(w_, x2+(x2-x1)*m), min(h_, y2+(y2-y1)*m)], dtype=np.float32)
            kpts_list.append(kpts)
            bboxes_list.append(bbox)
            sides_list.append(max(x2-x1, y2-y1) * (1+2*m))
            scores_list.append(1.0)
            lq_list.append(0)
        else:
            kpts_list.append(np.zeros((21, 3), dtype=np.float32))
            bboxes_list.append(np.zeros(4, dtype=np.float32))
            sides_list.append(0.0)
            scores_list.append(0.0)
            lq_list.append(1)

    hs_dir = out_dir / "handshape"; hs_dir.mkdir(parents=True, exist_ok=True)
    hand_code = "R" if hand == "right" else "L"
    hs_name = f"{signer_id}_{video_id}_{hand_code}_{eid}.npz"
    hs_path = hs_dir / hs_name
    np.savez(hs_path,
             frame_indices=np.array(hs_idx, dtype=np.int64),
             keypoints=np.stack(kpts_list), bboxes=np.stack(bboxes_list),
             bbox_side_px=np.array(sides_list, dtype=np.float32),
             scores=np.array(scores_list, dtype=np.float32),
             low_quality=np.array(lq_list, dtype=np.uint8))

    # --- Location: (24, 17, 2) pose + (24, 2) hand ---
    loc_frames, loc_idx = extract_frames(str(video_path), start, end, 24)
    pose_kpts, pose_scores, hand_pts, hand_scores = [], [], [], []
    for frame in loc_frames:
        pres = mp_pose.process(frame)
        h_, w_ = frame.shape[:2]
        if pres.pose_landmarks:
            lm = pres.pose_landmarks.landmark[:17]
            pose_kpts.append(np.array([[p.x*w_, p.y*h_] for p in lm], dtype=np.float32))
            pose_scores.append(np.array([p.visibility for p in lm], dtype=np.float32))
        else:
            pose_kpts.append(np.zeros((17, 2), dtype=np.float32))
            pose_scores.append(np.zeros(17, dtype=np.float32))
        hres = mp_hands.process(frame)
        if hres.multi_hand_landmarks:
            chosen = hres.multi_hand_landmarks[0]
            hand_pts.append(np.array([chosen.landmark[0].x*w_, chosen.landmark[0].y*h_], dtype=np.float32))
            hand_scores.append(1.0)
        else:
            hand_pts.append(np.zeros(2, dtype=np.float32))
            hand_scores.append(0.0)

    loc_dir = out_dir / "location"; loc_dir.mkdir(parents=True, exist_ok=True)
    loc_name = f"{signer_id}_{video_id}_{eid}_RIGHT.npz" if hand == "right" else f"{signer_id}_{video_id}_{eid}_LEFT.npz"
    loc_path = loc_dir / loc_name
    np.savez(loc_path,
             pose_kpts=np.stack(pose_kpts), pose_score=np.stack(pose_scores),
             hand_point=np.stack(hand_pts), hand_score=np.array(hand_scores),
             valid_mask=np.array([1]*len(pose_kpts), dtype=np.uint8),
             frame_indices=np.array(loc_idx, dtype=np.int32),
             source_pose=np.array("mediapipe", dtype="<U20"),
             source_hand=np.array("mediapipe", dtype="<U20"))

    from movement_channel.src.geometry import movement_from_wrist

    mv_dir = out_dir / "movement"
    mv_dir.mkdir(parents=True, exist_ok=True)
    mv_name = (
        f"{signer_id}_{video_id}_{eid}_RIGHT.npz"
        if hand == "right"
        else f"{signer_id}_{video_id}_{eid}_LEFT.npz"
    )
    mv_path = mv_dir / mv_name
    np.savez(
        mv_path,
        wrist_point=np.stack(hand_pts),
        wrist_score=np.array(hand_scores, dtype=np.float32),
        frame_indices=np.array(loc_idx, dtype=np.int32),
    )
    movement_prediction = movement_from_wrist(
        np.stack(hand_pts),
        np.array(hand_scores, dtype=np.float32),
        hand=hand,
        event_id=eid,
        signer_id=signer_id,
        video_id=video_id,
        span=span,
    )

    # --- Orientation: 24-frame ROI, plus landmarks and palm normal ---
    ori_frames, ori_idx = extract_frames(str(video_path), start, end, ORIENTATION_FRAMES)
    roi_list = []
    ori_landmarks = []
    for frame in ori_frames:
        h_, w_ = frame.shape[:2]
        hres = mp_hands.process(frame)
        if hres.multi_hand_landmarks:
            chosen = choose_signer_hand_landmarks(
                hres.multi_hand_landmarks, hres.multi_handedness, hand
            )
            if chosen is None:
                chosen = hres.multi_hand_landmarks[0]
            ori_landmarks.append(
                np.array(
                    [[lm.x, lm.y, lm.z] for lm in chosen.landmark],
                    dtype=np.float32,
                )
            )
            xs = [lm.x*w_ for lm in chosen.landmark]
            ys = [lm.y*h_ for lm in chosen.landmark]
            x1, x2 = max(0, int(min(xs)-20)), min(w_, int(max(xs)+20))
            y1, y2 = max(0, int(min(ys)-20)), min(h_, int(max(ys)+20))
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi = cv2.resize(roi, (128, 128))
            else:
                roi = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            ori_landmarks.append(np.zeros((21, 3), dtype=np.float32))
            roi = np.zeros((128, 128, 3), dtype=np.uint8)
        roi_list.append(roi)

    ori_path = _save_orientation_npz(out_dir, eid, hand, roi_list, ori_landmarks, ori_idx)

    orientation_prediction = orientation_from_landmarks(
        ori_landmarks,
        hand=hand,
        event_id=eid,
        signer_id=signer_id,
        video_id=video_id,
        span=span,
    )
    return {
        "handshape": hs_path,
        "location": loc_path,
        "orientation": ori_path,
        "movement": mv_path,
        "orientation_prediction": orientation_prediction,
        "movement_prediction": movement_prediction,
    }


# ============================================================
# 5. 拷贝 npz 到目标缓存目录
# ============================================================
def _process_event_from_cache(video_path, ev, out_dir, fast_body):
    """无编号视频不再对每一小段重新检测手。手用切分时的关键点，身体用一次扫完的结果。"""
    eid = ev["event_id"]
    hand = ev.get("hand", "right")
    span = ev["span"]
    start, end = int(span["start_frame"]), int(span["end_frame"])
    _, signer_id, video_id = _parse_video_uid(video_path)
    cached = _cached_hand_arrays(video_path, hand)
    if cached is None:
        return process_event(video_path, ev, out_dir, fast_body=None)
    hand_xyz, hand_detected = cached
    total = int(hand_xyz.shape[0])
    width = int(fast_body["width"])
    height = int(fast_body["height"])

    def hand_at(index):
        if index < 0 or index >= total or not bool(hand_detected[index]):
            return None
        return hand_xyz[index]

    hs_idx = _sample_indices(start, end, 8, total)
    kpts_list, bboxes_list, sides_list, scores_list, lq_list = [], [], [], [], []
    for index in hs_idx:
        points = hand_at(index)
        if points is None:
            kpts_list.append(np.zeros((21, 3), dtype=np.float32))
            bboxes_list.append(np.zeros(4, dtype=np.float32))
            sides_list.append(0.0)
            scores_list.append(0.0)
            lq_list.append(1)
            continue
        xs = points[:, 0] * width
        ys = points[:, 1] * height
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        margin = 0.1
        bbox = np.array([
            max(0, x1 - (x2 - x1) * margin),
            max(0, y1 - (y2 - y1) * margin),
            min(width, x2 + (x2 - x1) * margin),
            min(height, y2 + (y2 - y1) * margin),
        ], dtype=np.float32)
        kpts_list.append(points)
        bboxes_list.append(bbox)
        sides_list.append(max(x2 - x1, y2 - y1) * (1 + 2 * margin))
        scores_list.append(1.0)
        lq_list.append(0)
    hs_dir = out_dir / "handshape"
    hs_dir.mkdir(parents=True, exist_ok=True)
    hand_code = "R" if hand == "right" else "L"
    hs_path = hs_dir / f"{signer_id}_{video_id}_{hand_code}_{eid}.npz"
    np.savez(
        hs_path,
        frame_indices=np.array(hs_idx, dtype=np.int64),
        keypoints=np.stack(kpts_list),
        bboxes=np.stack(bboxes_list),
        bbox_side_px=np.array(sides_list, dtype=np.float32),
        scores=np.array(scores_list, dtype=np.float32),
        low_quality=np.array(lq_list, dtype=np.uint8),
    )

    loc_idx = _sample_indices(start, end, 24, total)
    pose_kpts, pose_scores, hand_pts, hand_scores = [], [], [], []
    for index in loc_idx:
        pose_kpts.append(fast_body["pose_kpts"][index])
        pose_scores.append(fast_body["pose_scores"][index])
        points = hand_at(index)
        if points is None:
            hand_pts.append(np.zeros(2, dtype=np.float32))
            hand_scores.append(0.0)
        else:
            hand_pts.append(np.array(
                [points[0, 0] * width, points[0, 1] * height],
                dtype=np.float32,
            ))
            hand_scores.append(1.0)
    loc_dir = out_dir / "location"
    loc_dir.mkdir(parents=True, exist_ok=True)
    loc_name = (
        f"{signer_id}_{video_id}_{eid}_RIGHT.npz"
        if hand == "right"
        else f"{signer_id}_{video_id}_{eid}_LEFT.npz"
    )
    loc_path = loc_dir / loc_name
    np.savez(
        loc_path,
        pose_kpts=np.stack(pose_kpts),
        pose_score=np.stack(pose_scores),
        hand_point=np.stack(hand_pts),
        hand_score=np.array(hand_scores),
        valid_mask=np.array([1] * len(pose_kpts), dtype=np.uint8),
        frame_indices=np.array(loc_idx, dtype=np.int32),
        source_pose=np.array("mediapipe", dtype="<U20"),
        source_hand=np.array("mediapipe", dtype="<U20"),
    )

    from movement_channel.src.geometry import movement_from_wrist

    mv_dir = out_dir / "movement"
    mv_dir.mkdir(parents=True, exist_ok=True)
    mv_path = mv_dir / loc_name
    np.savez(
        mv_path,
        wrist_point=np.stack(hand_pts),
        wrist_score=np.array(hand_scores, dtype=np.float32),
        frame_indices=np.array(loc_idx, dtype=np.int32),
    )
    movement_prediction = movement_from_wrist(
        np.stack(hand_pts),
        np.array(hand_scores, dtype=np.float32),
        hand=hand,
        event_id=eid,
        signer_id=signer_id,
        video_id=video_id,
        span=span,
    )

    ori_idx = _sample_indices(start, end, ORIENTATION_FRAMES, total)
    ori_frames, _ = extract_frames(str(video_path), start, end, ORIENTATION_FRAMES) if ori_idx else ([], [])
    roi_list = []
    ori_landmarks = []
    blank_roi = np.zeros((128, 128, 3), dtype=np.uint8)
    blank_hand = np.zeros((21, 3), dtype=np.float32)
    for frame, index in zip(ori_frames, ori_idx):
        points = hand_at(index)
        if points is None:
            roi_list.append(blank_roi.copy())
            ori_landmarks.append(blank_hand.copy())
            continue
        ori_landmarks.append(np.asarray(points, dtype=np.float32))
        xs = points[:, 0] * width
        ys = points[:, 1] * height
        x1, x2 = max(0, int(xs.min()) - 20), min(width, int(xs.max()) + 20)
        y1, y2 = max(0, int(ys.min()) - 20), min(height, int(ys.max()) + 20)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi_list.append(cv2.resize(roi, (128, 128)))
        else:
            roi_list.append(blank_roi.copy())
    while len(roi_list) < len(ori_idx):
        roi_list.append(blank_roi.copy())
        ori_landmarks.append(blank_hand.copy())
    ori_path = _save_orientation_npz(out_dir, eid, hand, roi_list, ori_landmarks, ori_idx)
    orientation_prediction = orientation_from_landmarks(
        ori_landmarks,
        hand=hand,
        event_id=eid,
        signer_id=signer_id,
        video_id=video_id,
        span=span,
    )
    return {
        "handshape": hs_path,
        "location": loc_path,
        "orientation": ori_path,
        "movement": mv_path,
        "orientation_prediction": orientation_prediction,
        "movement_prediction": movement_prediction,
    }


def deploy_to_channels(hs_path: Path, loc_path: Path, ori_path: Path, mv_path: Path):
    """把 npz 拷贝到四通道的默认特征目录"""
    # Handshape
    hs_dst = ROOT / "handshape_channel_data28" / "cache" / "keypoints"
    hs_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hs_path, hs_dst / hs_path.name)

    # Location
    loc_dst = ROOT / "location_channel" / "data" / "features"
    loc_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(loc_path, loc_dst / loc_path.name)

    # Orientation
    ori_dst = ROOT / "orientation_channel" / "data" / "precomputed"
    ori_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ori_path, ori_dst / ori_path.name)

    mv_dst = ROOT / "movement_channel" / "data" / "features"
    mv_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mv_path, mv_dst / mv_path.name)

    print(f"  [DEPLOY] 已部署到四通道缓存目录")


def use_unnumbered_path(has_events: bool, segment_mode: str) -> bool:
    """无编号路径只在严格切分和标注都没有事件时启用。"""
    return (not has_events) and segment_mode == "auto"


def build_segment_events(
    video_path: Path,
    checkpoint_path: str,
    tau: float = 0.5,
    hands: tuple[str, ...] = ("left", "right"),
    span_fn=None,
    boundary_source: str = "BoundaryTCN",
) -> tuple[list[dict], str]:
    """调用 BoundaryTCN，将左右手切分结果转换成统一事件结构。"""
    from segment_video import segment_from_video_file

    video_uid, signer_id, video_id = _parse_video_uid(video_path)
    events = []
    for hand in hands:
        segmented = segment_from_video_file(
            str(video_path),
            checkpoint_path=checkpoint_path,
            hand=hand,
            tau=tau,
            device="cpu",
            span_fn=span_fn,
        )
        spans = sorted(
            segmented.get("segments", []),
            key=lambda item: (int(item["start"]), int(item["end"])),
        )
        prefix = "L" if hand == "left" else "R"
        for index, segment in enumerate(spans, 1):
            events.append({
                "video_id": video_id,
                "video_uid": video_uid,
                "signer_id": signer_id,
                "video_path": str(video_path),
                "hand": hand,
                "event_id": f"{prefix}_Event_{index}",
                "span": {
                    "start_frame": int(segment["start"]),
                    "end_frame": int(segment["end"]) + 1,
                },
                "boundary": {
                    "source": boundary_source,
                    "score": float(segment.get("score", 0.0)),
                    "inside_mean": float(segment.get("inside_mean", 0.0)),
                    "checkpoint": str(checkpoint_path),
                    "detect_ratio": float(segmented.get("detect_ratio", 0.0)),
                },
                "fps": segmented.get("fps"),
                "num_frames": segmented.get("n_frames"),
            })
    events.sort(
        key=lambda event: (
            int(event["span"]["start_frame"]),
            event["hand"],
        )
    )
    return events, video_uid


def build_unnumbered_events(
    video_path: Path,
    checkpoint_path: str,
    tau: float = 0.5,
) -> tuple[list[dict], str]:
    """只给对不上标注的视频使用，不改变语料视频的严格切分。"""
    from unnumbered_segment import align_bimanual_events, segment_pose_unnumbered

    print("[INFO] 文件名没有对上标注，改走无编号视频切分")
    events, video_uid = build_segment_events(
        video_path,
        checkpoint_path=checkpoint_path,
        tau=tau,
        span_fn=segment_pose_unnumbered,
        boundary_source="unnumbered",
    )
    return align_bimanual_events(events), video_uid


# ============================================================
# 6. 主流程
# ============================================================
def video2json(
    video_path: str,
    events_jsonl: str,
    out_dir: str,
    run_inference: bool,
    segment_mode: str = "auto",
    boundary_checkpoint: str = (
        "boundary_prefill_B_final/logs/run_20260228_210024/best.pt"
    ),
    boundary_tau: float = 0.5,
):
    import time
    import hashlib

    video_path = Path(video_path)
    out = Path(out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    # ===== 1. 获取事件边界 =====
    if segment_mode not in {"auto", "model", "annotation"}:
        raise ValueError(f"不支持的 segment_mode: {segment_mode}")

    events = []
    video_uid = _parse_video_uid(video_path)[0]
    segmentation_source = "annotation"
    run_strict = segment_mode == "model" or (
        segment_mode == "auto" and _is_corpus_filename(video_path)
    )
    if run_strict:
        checkpoint = Path(boundary_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if checkpoint.exists():
            try:
                events, video_uid = build_segment_events(
                    video_path,
                    checkpoint_path=str(checkpoint),
                    tau=boundary_tau,
                )
                segmentation_source = "BoundaryTCN"
            except Exception as exc:
                if segment_mode == "model":
                    raise
                print(f"[WARN] BoundaryTCN 切分失败，回退标注边界: {exc}")
        elif segment_mode == "model":
            raise FileNotFoundError(f"切分模型不存在: {checkpoint}")

    if not events and segment_mode in {"auto", "annotation"}:
        events, video_uid = load_video_events(video_path, events_jsonl)
        segmentation_source = (
            "annotation_fallback" if segment_mode == "auto" else "annotation"
        )

    fast_body = None
    if use_unnumbered_path(bool(events), segment_mode):
        checkpoint = Path(boundary_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if checkpoint.exists():
            try:
                from segment_video import preload_unnumbered_poses

                preload_unnumbered_poses(video_path)
                fast_body = _scan_body_pose(video_path)
                events, video_uid = build_unnumbered_events(
                    video_path,
                    checkpoint_path=str(checkpoint),
                    tau=boundary_tau,
                )
                if events:
                    segmentation_source = "unnumbered"
            except Exception as exc:
                print(f"[WARN] 无编号视频切分失败: {exc}")

    print(f"[INFO] 视频: {video_path.name}")
    print(f"[INFO] video_uid: {video_uid}, 找到 {len(events)} 个 event")
    print(f"[INFO] 事件来源: {segmentation_source}")

    # ===== 2. 如果 events 为空，直接报错返回，不跑推理 =====
    if not events:
        print(f"[ERROR] 未找到任何 event！视频 {video_path.name} 未切分出有效手语事件")
        print("[HINT] 新视频请使用 BoundaryTCN（auto 或 model）切分，并确认视频中手部清晰可见")
        return []

    # ===== 3. 每次跑前清掉三通道 npz 缓存，强制重新生成 =====
    print(f"[INFO] 清理旧缓存...")
    for cache_dir in [
        ROOT / "handshape_channel_data28" / "cache" / "keypoints",
        ROOT / "location_channel" / "data" / "features",
        ROOT / "orientation_channel" / "data" / "precomputed",
        ROOT / "movement_channel" / "data" / "features",
    ]:
        for f in cache_dir.glob("*.npz"):
            f.unlink()
            print(f"  删除: {f.name}")

    all_results = []

    for ev in events:
        eid = ev["event_id"]
        print(f"\n  -> {eid} ({ev.get('hand')}) 帧[{ev['span']['start_frame']}, {ev['span']['end_frame']}]")

        # ===== 4. 预处理（为本视频生成 npz）=====
        paths = process_event(
            video_path,
            ev,
            out,
            fast_body=fast_body if segmentation_source == "unnumbered" else None,
        )
        print(f"    [OK] npz 已生成")

        # ===== 5. 部署到三通道缓存目录 =====
        deploy_to_channels(
            paths["handshape"],
            paths["location"],
            paths["orientation"],
            paths["movement"],
        )

        # ===== 6. 校验 npz 确实部署成功了 =====

        hs_dir = Path("handshape_channel_data28/cache/keypoints")
        hs_matches = list(hs_dir.glob(f"*{eid}.npz"))
        if not hs_matches:
            print(f"    [WARN] 包含 {eid} 的 handshape npz 未找到，跳过推理")
            continue

        # ===== 7. 推理（用 event_id 作为 sample_id）=====
        if run_inference:
            from run_all import run_all
            unified_sample_id = (
                f"{ev.get('signer_id')}_{ev.get('video_id')}_{eid}_"
                f"{str(ev.get('hand', '')).upper()}"
            )
            result = run_all(
                split="test",
                sample_id=unified_sample_id,
                event_id=eid,
                signer_id=ev.get("signer_id"),
                video_id=ev.get("video_id"),
                hand=ev.get("hand"),
                span=ev.get("span"),
                video_path=(
                    str(video_path) if segmentation_source == "unnumbered" else None
                ),
                allow_unnumbered_inference=segmentation_source == "unnumbered",
                channel_overrides={
                    "orientation": paths["orientation_prediction"],
                    "movement": paths["movement_prediction"],
                },
            )
            result["segmentation"] = {
                "source": segmentation_source,
                "span": ev.get("span"),
                "fps": ev.get("fps") or 30,
                "num_frames": ev.get("num_frames"),
                **ev.get("boundary", {}),
            }
            all_results.append(result)
            ch = result["channels"]
            codes = {k: v["status"]["code"] for k, v in ch.items()}
            print(f"    [DONE] 推理完成: {codes}")

    # ===== 8. 保存 JSON =====
    if run_inference and all_results:
        json_path = out / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] 统一 JSON 已保存: {json_path}")
    elif run_inference and not all_results:
        print(f"\n[WARN] 推理完成但无结果（可能所有 event 的 npz 都缺失）")

    return all_results

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频 -> 统一 JSON")
    parser.add_argument("--video", "-v", required=True)
    parser.add_argument("--events", "-e", default="data28/annotations/mhsl_events.jsonl")
    parser.add_argument("--out", "-o", default="output")
    parser.add_argument("--run_all", action="store_true", help="预处理后自动调 run_all 推理")
    parser.add_argument(
        "--segment_mode",
        choices=["auto", "model", "annotation"],
        default="auto",
        help="auto=BoundaryTCN失败时回退标注；model=仅模型；annotation=仅标注",
    )
    parser.add_argument(
        "--boundary_checkpoint",
        default="boundary_prefill_B_final/logs/run_20260228_210024/best.pt",
    )
    parser.add_argument("--boundary_tau", type=float, default=0.5)
    args = parser.parse_args()

    video2json(
        video_path=args.video,
        events_jsonl=args.events,
        out_dir=args.out,
        run_inference=args.run_all,
        segment_mode=args.segment_mode,
        boundary_checkpoint=args.boundary_checkpoint,
        boundary_tau=args.boundary_tau,
    )
