"""无编号视频的单独切分。

只在严格 BoundaryTCN 和标注库都没有事件时调用。
语料视频继续使用 segment_video.segment_video 的原阈值。
"""

from __future__ import annotations

import numpy as np


def _split_model_run(start, end, scores, max_frames, min_frames):
    length = end - start + 1
    if length <= max_frames:
        return [(start, end)]
    midpoint = start + length // 2
    left = max(start + min_frames, midpoint - 8)
    right = min(end - min_frames, midpoint + 8)
    if left > right:
        cut = midpoint
    else:
        cut = left + int(np.argmin(scores[left:right + 1]))
    if cut <= start or cut >= end:
        cut = max(start + 1, min(end - 1, midpoint))
    return (
        _split_model_run(start, cut, scores, max_frames, min_frames)
        + _split_model_run(cut + 1, end, scores, max_frames, min_frames)
    )


def spans_from_model_inside(p_inside, min_frames=6, max_frames=80):
    """严格解码为空时，按模型自己的内部概率取出还在手语里的片段。"""
    inside = np.asarray(p_inside, dtype=np.float32).reshape(-1)
    total = int(inside.shape[0])
    if total < min_frames or float(np.max(inside)) < 0.05:
        return []
    threshold = max(0.05, float(np.max(inside)) * 0.45)
    active = inside >= threshold
    runs = []
    index = 0
    while index < total:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < total and active[index]:
            index += 1
        if index - start >= min_frames:
            runs.append((start, index - 1))

    spans = []
    for start, end in runs:
        for piece_start, piece_end in _split_model_run(
            start, end, inside, max_frames, min_frames
        ):
            piece = inside[piece_start:piece_end + 1]
            spans.append({
                "start": int(piece_start),
                "end": int(piece_end),
                "length": int(piece_end - piece_start + 1),
                "score": float(np.mean(piece)),
                "inside_mean": float(np.mean(piece)),
            })
    return spans


def _wrist_is_moving(pose_xyz, index, window=12, extent_thr=0.08, detected=None):
    """只看当前帧往前的已检出手腕。没检出，或位移还不到手语动作，都算没动。"""
    wrist = np.asarray(pose_xyz, dtype=np.float32)
    if wrist.ndim != 3 or wrist.shape[0] == 0:
        return False
    total = int(wrist.shape[0])
    index = int(index)
    if index < 0 or index >= total:
        return False
    flags = None if detected is None else np.asarray(detected, dtype=bool)
    if flags is not None and not bool(flags[index]):
        return False
    start = max(0, index - int(window))
    points = wrist[start:index + 1, 0, :2]
    if flags is not None:
        points = points[flags[start:index + 1]]
    else:
        points = points[np.linalg.norm(points, axis=1) > 1e-4]
    if len(points) < 3:
        return False
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    return extent >= float(extent_thr)


def trim_still_edges(
    pose_xyz, spans, window=12, extent_thr=0.08, min_frames=6, detected=None
):
    """去掉片段开头和结尾手腕没动的帧。整段都没动就丢掉。"""
    kept = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        original = (start, end)
        while start <= end and not _wrist_is_moving(
            pose_xyz, start, window, extent_thr, detected
        ):
            start += 1
        while end >= start and not _wrist_is_moving(
            pose_xyz, end, window, extent_thr, detected
        ):
            end -= 1
        if end - start + 1 < min_frames:
            print(
                f"[INFO] 无编号视频丢弃静止手片段 "
                f"帧 {original[0]}–{original[1]}"
            )
            continue
        if (start, end) != original:
            print(
                f"[INFO] 无编号视频裁掉静止手 "
                f"帧 {original[0]}–{original[1]} -> {start}–{end}"
            )
        updated = dict(span)
        updated["start"] = int(start)
        updated["end"] = int(end)
        updated["length"] = int(end - start + 1)
        kept.append(updated)
    return kept


def merge_overlapping_spans(spans):
    """只吞掉几乎落在已有段里的重复检测。伸到段外的下一次动作单独留下。"""
    ordered = sorted(
        spans,
        key=lambda item: (
            -(int(item["end"]) - int(item["start"])),
            int(item["start"]),
        ),
    )
    kept = []
    for span in ordered:
        start = int(span["start"])
        end = int(span["end"])
        length = end - start + 1
        host = None
        for current in kept:
            overlap = (
                min(int(current["end"]), end) - max(int(current["start"]), start) + 1
            )
            if overlap < 6:
                continue
            stick_out = (
                max(0, int(current["start"]) - start)
                + max(0, end - int(current["end"]))
            )
            if stick_out <= 15:
                host = current
                break
        if host is not None:
            previous = (int(host["start"]), int(host["end"]))
            host["start"] = min(int(host["start"]), start)
            host["end"] = max(int(host["end"]), end)
            host["length"] = int(host["end"]) - int(host["start"]) + 1
            if float(span.get("score") or 0) > float(host.get("score") or 0):
                host["score"] = span.get("score", host.get("score"))
                host["inside_mean"] = span.get(
                    "inside_mean", host.get("inside_mean")
                )
            print(
                f"[INFO] 无编号视频合并重复片段 "
                f"帧 {previous[0]}–{previous[1]} + {start}–{end} "
                f"-> {host['start']}–{host['end']}"
            )
            continue
        updated = dict(span)
        updated["start"] = start
        updated["end"] = end
        updated["length"] = length
        kept.append(updated)
    kept.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return kept


def segment_kinematic_signing_space(
    pose_xyz,
    detected=None,
    wrist_pose=None,
    shoulder_y=None,
    hip_y=None,
    p_inside=None,
    min_frames=6,
):
    """基于躯干归一化的手语空间高度、预备/回落运动学裁剪、保持段保留与多冲程转移切分。"""
    arr = np.asarray(pose_xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < min_frames:
        return []
    total = int(arr.shape[0])

    s_med = (
        float(np.nanmedian(shoulder_y))
        if shoulder_y is not None and np.any(~np.isnan(shoulder_y))
        else 0.46
    )
    h_med = (
        float(np.nanmedian(hip_y))
        if hip_y is not None and np.any(~np.isnan(hip_y))
        else 0.88
    )
    torso = max(0.25, h_med - s_med)

    w_traj = np.full((total, 2), np.nan, dtype=np.float32)
    flags = None if detected is None else np.asarray(detected, dtype=bool)
    wp = None if wrist_pose is None else np.asarray(wrist_pose, dtype=np.float32)

    for t in range(total):
        if flags is not None and bool(flags[t]):
            w_traj[t] = arr[t, 0, :2]
        elif wp is not None and wp.shape[0] == total and not np.isnan(wp[t, 0]):
            w_traj[t] = wp[t, :2]
        elif flags is None and np.linalg.norm(arr[t, 0, :2]) > 1e-4:
            w_traj[t] = arr[t, 0, :2]

    valid = np.where(~np.isnan(w_traj[:, 0]))[0]
    if len(valid) < min_frames:
        return []
    for c in range(2):
        w_traj[:, c] = np.interp(np.arange(total), valid, w_traj[valid, c])

    w_s = np.copy(w_traj)
    for _ in range(2):
        nxt = np.copy(w_s)
        nxt[1:-1] = 0.25 * w_s[:-2] + 0.5 * w_s[1:-1] + 0.25 * w_s[2:]
        w_s = nxt

    y_rel = (w_s[:, 1] - s_med) / torso
    dy = np.zeros(total, dtype=np.float32)
    dx = np.zeros(total, dtype=np.float32)
    dy[1:] = (w_s[1:, 1] - w_s[:-1, 1]) / torso
    dx[1:] = (w_s[1:, 0] - w_s[:-1, 0]) / torso
    spd = np.hypot(dx, dy)

    in_space = y_rel < 0.65
    runs = []
    idx = 0
    while idx < total:
        if not in_space[idx]:
            idx += 1
            continue
        end_idx = idx
        while end_idx < total and in_space[end_idx]:
            end_idx += 1
        pts = w_s[idx:end_idx]
        ext = float(np.max(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else 0.0
        if (
            end_idx - idx >= min_frames
            and float(np.min(y_rel[idx:end_idx])) < 0.45
            and ext >= 0.04
        ):
            runs.append((idx, end_idx - 1))
        idx = end_idx

    spans = []
    inside_arr = (
        np.asarray(p_inside, dtype=np.float32).reshape(-1)
        if p_inside is not None
        else None
    )

    for r_start, r_end in runs:
        s, e = r_start, r_end
        # 1) 裁掉抬手预备段（从腰部快速上抬至打手语高度）
        while s < e - min_frames:
            if (y_rel[s] > 0.43 and dy[s] < -0.018) or (
                dy[s] < -0.042 and abs(dy[s]) >= 1.15 * abs(dx[s])
            ):
                s += 1
            else:
                break
        # 2) 裁掉收手回落段（打完手语向腰部下落，保留胸部/面部静止 Hold 段）
        while e > s + min_frames:
            if (y_rel[e] > 0.40 and dy[e] > 0.012) or (
                dy[e] > 0.008 and y_rel[min(total - 1, e + 4)] > y_rel[e] + 0.04
            ):
                e -= 1
            else:
                break
        while (
            e > s + min_frames
            and dy[e] > 0.005
            and dy[min(total - 1, e + 1)] >= 0.008
            and y_rel[min(total - 1, e + 5)] > y_rel[e] + 0.06
        ):
            e -= 1

        if e - s + 1 < min_frames:
            continue

        # 3) 检测多冲程/多词素之间的高度下沉+重定位过渡（Transition Valley）并切分
        pieces = [(s, e)]
        if e - s + 1 >= 24:
            best_dip = None
            best_prom = 0.0
            for t in range(s + 8, e - 8):
                l_min = float(np.min(y_rel[s:t]))
                r_min = float(np.min(y_rel[t + 1:e + 1]))
                prom = min(y_rel[t] - l_min, y_rel[t] - r_min)
                if prom > best_prom:
                    best_prom = prom
                    best_dip = t
            if best_dip is not None and best_prom >= 0.12:
                e1_end = best_dip
                # 往前退掉下沉进入过渡谷的帧（dy > 0.008）
                while e1_end > s + min_frames and dy[e1_end] > 0.008:
                    e1_end -= 1
                for k in range(e1_end, s + min_frames, -1):
                    if dy[k] <= 0.010 and spd[k] <= 0.015:
                        e1_end = k
                        break
                reloc_win = spd[best_dip:min(e, best_dip + 16) + 1]
                t_peak = best_dip + int(np.argmax(reloc_win))
                e2_start = t_peak
                while e2_start < e - min_frames and (
                    spd[e2_start] > 0.018 or dy[e2_start] < -0.010
                ):
                    e2_start += 1
                if e1_end - s + 1 >= min_frames and e - e2_start + 1 >= min_frames:
                    pieces = [(s, e1_end), (e2_start, e)]

        for cs, ce in pieces:
            if ce - cs + 1 < min_frames:
                continue
            sc_val = (
                float(np.mean(inside_arr[cs:ce + 1]))
                if inside_arr is not None and inside_arr.shape[0] == total
                else 0.5
            )
            spans.append({
                "start": int(cs),
                "end": int(ce),
                "length": int(ce - cs + 1),
                "score": sc_val,
                "inside_mean": sc_val,
            })

    return merge_overlapping_spans(spans)


def align_bimanual_events(events: list[dict], sync_tol: int = 6) -> list[dict]:
    """当左右手在同一手语词段同时打手语且起止仅相差几帧时，自动协同对齐边界。"""
    left_evs = [ev for ev in events if ev.get("hand") == "left"]
    right_evs = [ev for ev in events if ev.get("hand") == "right"]
    for l_ev in left_evs:
        ls = int(l_ev["span"]["start_frame"])
        le = int(l_ev["span"]["end_frame"])
        for r_ev in right_evs:
            rs = int(r_ev["span"]["start_frame"])
            re = int(r_ev["span"]["end_frame"])
            ov = min(le, re) - max(ls, rs)
            union = max(le, re) - min(ls, rs)
            iou = ov / max(union, 1)
            if ov >= 6 and (
                (abs(ls - rs) <= sync_tol and abs(le - re) <= sync_tol)
                or iou >= 0.55
            ):
                shared_s = int(round(0.5 * (ls + rs)))
                shared_e = int(round(0.5 * (le + re)))
                if (shared_s, shared_e) != (ls, le) or (shared_s, shared_e) != (rs, re):
                    print(
                        f"[INFO] 双手协同边界对齐: "
                        f"L[{ls}, {le}] & R[{rs}, {re}] -> [{shared_s}, {shared_e}]"
                    )
                l_ev["span"]["start_frame"] = r_ev["span"]["start_frame"] = shared_s
                l_ev["span"]["end_frame"] = r_ev["span"]["end_frame"] = shared_e
    events.sort(
        key=lambda event: (
            int(event["span"]["start_frame"]),
            event["hand"],
        )
    )
    return events


def segment_pose_unnumbered(
    pose_xyz, checkpoint_path, tau=0.5, device="cpu",
    pose_raw=None, detected=None,
    wrist_pose=None, shoulder_y=None, hip_y=None,
):
    """同一套 BoundaryTCN 权重 + 躯干手语空间运动学精细切分。"""
    from segment_video import _load_boundary, build_pose_features, load_boundary_model

    _, decode_boundaries = _load_boundary()
    model, device = load_boundary_model(str(checkpoint_path), device)
    features = build_pose_features(pose_xyz)
    if features.shape[0] == 0:
        return []

    x = np.expand_dims(features.astype(np.float32), axis=0)
    import torch

    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))[0].cpu().numpy()

    p = 1.0 / (1.0 + np.exp(-logits))
    motion_pose = pose_xyz if pose_raw is None else pose_raw

    # 优先使用躯干归一化手语空间运动学切分（自动过滤抬手预备、回落静止，保留胸部保持段并切分多冲程转移）
    kin_spans = segment_kinematic_signing_space(
        motion_pose,
        detected=detected,
        wrist_pose=wrist_pose,
        shoulder_y=shoulder_y,
        hip_y=hip_y,
        p_inside=p[:, 2],
    )
    if kin_spans:
        print(f"[INFO] 无编号视频运动学精细切分出 {len(kin_spans)} 段")
        return kin_spans

    relaxed = decode_boundaries(
        p_start=p[:, 0],
        p_end=p[:, 1],
        p_inside=p[:, 2],
        tau=min(float(tau), 0.2),
        peak_radius=7,
        nms_window=7,
        Lmin=6,
        Lmax=120,
        inside_recommend_thr=0.15,
        pair_mode="score_then_nearest",
    )
    spans = merge_overlapping_spans(trim_still_edges(
        motion_pose, relaxed.get("spans_recommended", []), detected=detected
    ))
    if spans:
        print(f"[INFO] 无编号视频放宽阈值后切出 {len(spans)} 段")
        return spans

    spans = merge_overlapping_spans(trim_still_edges(
        motion_pose, spans_from_model_inside(p[:, 2]), detected=detected
    ))
    if spans:
        print(f"[INFO] 无编号视频按内部概率切出 {len(spans)} 段")
    return spans
