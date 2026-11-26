#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.indexer import build_manifest
from src.data.split import load_signer_split
from src.features.builder import FeatureBuilder
from src.infer.postprocess import decode_boundaries
from src.metrics.span import counts_to_prf, match_spans_iou
from src.models.tcn import BoundaryTCN
from src.utils.io import load_yaml, select_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pick and render Figure 3-3 shift/noise cases.")
    ap.add_argument("--config", default="configs/train_data28_poseflow_gated_farneback.yaml")
    ap.add_argument("--checkpoint", default="logs/pose_flow_gated_farneback/run_20260301_210004/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--window", type=int, default=7, help="Boundary tolerance window for shift matching.")
    ap.add_argument(
        "--prefer-hand",
        default="auto",
        choices=["auto", "L", "R"],
        help="Prefer selecting cases from a specific hand if available.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Optional filename/key suffix, e.g. 'left' to generate *_left.png and shift_left/noise_left keys.",
    )
    ap.add_argument("--out-root", default="outputs/paper/dataset_stats")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def _feature_builder_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    flow_cfg = cfg.get("features", {}).get("flow", {}) or {}
    return {
        "score_thr": float(cfg["features"]["score_thr"]),
        "median_k": int(cfg["features"]["median_k"]),
        "feature_mode": str(cfg.get("features", {}).get("mode", "auto")),
        "flow_enabled": bool(flow_cfg.get("enabled", False)),
        "flow_root": flow_cfg.get("root", None),
        "flow_feature_keys": flow_cfg.get(
            "feature_keys", ["mu", "p95", "p99", "delta_mu", "u_mean", "v_mean"]
        ),
        "flow_allow_missing": bool(flow_cfg.get("allow_missing", True)),
        "flow_feature_dim": int(flow_cfg.get("feature_dim", 6)),
    }


def _model_kwargs(model_cfg: Dict[str, Any], input_dim: int) -> Dict[str, Any]:
    return {
        "input_dim": int(input_dim),
        "channels": int(model_cfg["channels"]),
        "layers": int(model_cfg["layers"]),
        "dilations": [int(x) for x in model_cfg["dilations"]],
        "kernel_size": int(model_cfg["kernel_size"]),
        "dropout": float(model_cfg["dropout"]),
        "fusion_mode": str(model_cfg.get("fusion_mode", "early")),
        "pose_dim": int(model_cfg.get("pose_dim", 23)),
        "flow_dim": int(model_cfg.get("flow_dim", 6)),
        "fusion_dim": int(model_cfg.get("fusion_dim", 64)),
        "gate_hidden": int(model_cfg.get("gate_hidden", 64)),
    }


def _match_offsets(gt: List[int], pred: List[int], window: int) -> List[int]:
    g = sorted(int(x) for x in gt)
    p = sorted(int(x) for x in pred)
    used = set()
    offsets: List[int] = []
    for t in p:
        cand = [(abs(t - gg), i, gg) for i, gg in enumerate(g) if i not in used and abs(t - gg) <= window]
        if not cand:
            continue
        _, i, gg = min(cand, key=lambda x: x[0])
        used.add(i)
        offsets.append(int(t - gg))
    return offsets


def _series_jitter_score(x: np.ndarray, diff_thr: float = 0.015) -> int:
    d = np.diff(x.astype(np.float32))
    if d.size < 3:
        return 0
    c = 0
    for i in range(d.size - 1):
        if d[i] * d[i + 1] < 0 and (abs(float(d[i])) + abs(float(d[i + 1]))) * 0.5 >= diff_thr:
            c += 1
    return int(c)


def _plot_one(
    out_path: Path,
    p_start: np.ndarray,
    p_end: np.ndarray,
    p_inside: np.ndarray,
    gt_starts: List[int],
    gt_ends: List[int],
    decoded: Dict[str, Any],
    title: str,
) -> None:
    T = len(p_start)
    x = np.arange(T)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, p_start, label="p_start", color="#1f77b4", linewidth=1.5)
    ax.plot(x, p_end, label="p_end", color="#ff7f0e", linewidth=1.5)
    ax.plot(x, p_inside, label="p_inside", color="#2ca02c", linewidth=1.5)

    for t in gt_starts:
        ax.axvline(int(t), color="#1f77b4", alpha=0.25, linestyle="--")
    for t in gt_ends:
        ax.axvline(int(t), color="#17becf", alpha=0.25, linestyle="--")
    for p in decoded.get("start_peaks", []):
        ax.scatter([int(p["t"])], [float(p["score"])], color="#d62728", s=18, zorder=3)
    for p in decoded.get("end_peaks", []):
        ax.scatter([int(p["t"])], [float(p["score"])], color="#9467bd", s=18, zorder=3)
    for s in decoded.get("spans_proposed", [])[:20]:
        ax.axvspan(int(s["start"]), int(s["end"]), color="#d62728", alpha=0.06)

    ax.set_xlim(0, max(1, T - 1))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    cfg = load_yaml(args.config)
    if args.device:
        cfg["runtime"]["device"] = args.device

    split_obj = load_signer_split(cfg["splits"]["file"])
    signers = split_obj.get(args.split, [])
    manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )
    records = [r for r in manifest.records if r.pose_path is not None]
    if not records:
        raise SystemExit("No records found.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_cfg", cfg["model"])
    model = BoundaryTCN(**_model_kwargs(model_cfg, int(ckpt["input_dim"])))
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = select_device(cfg["runtime"].get("device", "auto"))
    model = model.to(device).eval()

    feat_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))
    infer_cfg = cfg["infer"]

    candidates: List[Dict[str, Any]] = []
    for rec in records:
        feat_data = feat_builder.build_from_pose_path(rec.pose_path)
        for hand in ("L", "R"):
            feat = feat_data[hand]["features"]
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)[0].cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            p_start, p_end, p_inside = p[:, 0], p[:, 1], p[:, 2]
            decoded = decode_boundaries(
                p_start,
                p_end,
                p_inside,
                tau=float(args.tau),
                peak_radius=int(infer_cfg["peak_radius"]),
                nms_window=int(infer_cfg["w_nms"]),
                Lmin=int(infer_cfg["Lmin"]),
                Lmax=int(infer_cfg["Lmax"]),
                inside_recommend_thr=float(infer_cfg["inside_recommend_thr"]),
                pair_mode=str(infer_cfg.get("pair_mode", "score_then_nearest")),
            )

            gt_events = rec.events_by_hand.get(hand, [])
            gt_events = [ev for ev in gt_events if ev.train_valid]
            gt_starts = [int(ev.start) for ev in gt_events]
            gt_ends = [int(ev.end) for ev in gt_events]
            gt_spans = [(int(ev.start), int(ev.end)) for ev in gt_events]
            pred_starts = [int(x["t"]) for x in decoded["start_peaks"]]
            pred_ends = [int(x["t"]) for x in decoded["end_peaks"]]
            pred_spans = [(int(s["start"]), int(s["end"])) for s in decoded["spans_proposed"]]

            start_off = _match_offsets(gt_starts, pred_starts, int(args.window))
            end_off = _match_offsets(gt_ends, pred_ends, int(args.window))
            all_off = [abs(x) for x in (start_off + end_off)]
            mean_abs_off = float(np.mean(all_off)) if all_off else 0.0

            tp, fp, fn = match_spans_iou(gt_spans, pred_spans, iou_thr=0.5)
            span_f1 = float(counts_to_prf(tp, fp, fn)["f1"])
            peaks_total = len(pred_starts) + len(pred_ends)
            jitter_score = _series_jitter_score(p_start) + _series_jitter_score(p_end)

            candidates.append(
                {
                    "clip_key": f"{rec.signer_id}_{rec.clip_id}_{hand}",
                    "clip_id": rec.clip_id,
                    "video_id": rec.video_id,
                    "signer_id": rec.signer_id,
                    "hand": hand,
                    "gt_spans": len(gt_spans),
                    "pred_spans": len(pred_spans),
                    "start_peaks": len(pred_starts),
                    "end_peaks": len(pred_ends),
                    "peaks_total": peaks_total,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "f1_iou05": span_f1,
                    "mean_abs_offset": mean_abs_off,
                    "offset_count": len(all_off),
                    "jitter_score": jitter_score,
                    "p_start": p_start,
                    "p_end": p_end,
                    "p_inside": p_inside,
                    "decoded": decoded,
                    "gt_starts": gt_starts,
                    "gt_ends": gt_ends,
                }
            )

    # shift: matched but noticeably shifted
    shift_pool = [
        c
        for c in candidates
        if c["gt_spans"] > 0
        and c["offset_count"] >= 2
        and 2.0 <= c["mean_abs_offset"] <= float(args.window)
        and c["f1_iou05"] < 0.95
        and c["tp"] >= 1
    ]
    if not shift_pool:
        shift_pool = [c for c in candidates if c["gt_spans"] > 0 and c["offset_count"] >= 1 and c["tp"] >= 1]
    if args.prefer_hand in ("L", "R"):
        shift_pool_pref = [c for c in shift_pool if c["hand"] == args.prefer_hand]
        if shift_pool_pref:
            shift_pool = shift_pool_pref
    shift_case = sorted(
        shift_pool, key=lambda x: (x["mean_abs_offset"], -x["f1_iou05"], x["peaks_total"]), reverse=True
    )[0]

    # noise: many peaks with poor span quality, usually jittery
    noise_pool = [
        c
        for c in candidates
        if c["gt_spans"] > 0 and c["peaks_total"] >= 8 and c["f1_iou05"] <= 0.2 and c["tp"] == 0
    ]
    if not noise_pool:
        noise_pool = [c for c in candidates if c["gt_spans"] > 0 and c["peaks_total"] >= 6 and c["f1_iou05"] <= 0.3]
    if args.prefer_hand in ("L", "R"):
        noise_pool_pref = [c for c in noise_pool if c["hand"] == args.prefer_hand]
        if noise_pool_pref:
            noise_pool = noise_pool_pref
    noise_case = sorted(
        noise_pool, key=lambda x: (x["peaks_total"], x["jitter_score"], -x["f1_iou05"]), reverse=True
    )[0]

    out_root = Path(args.out_root).resolve()
    raw_root = out_root / "figure_3_3_cases_raw"
    shift_raw = raw_root / "shift" / f"test_{shift_case['video_id']}_{'left' if shift_case['hand']=='L' else 'right'}.png"
    noise_raw = raw_root / "noise" / f"test_{noise_case['video_id']}_{'left' if noise_case['hand']=='L' else 'right'}.png"

    _plot_one(
        shift_raw,
        shift_case["p_start"],
        shift_case["p_end"],
        shift_case["p_inside"],
        shift_case["gt_starts"],
        shift_case["gt_ends"],
        shift_case["decoded"],
        title=f"Shift case | {shift_case['signer_id']} {shift_case['video_id']} {'left' if shift_case['hand']=='L' else 'right'}",
    )
    _plot_one(
        noise_raw,
        noise_case["p_start"],
        noise_case["p_end"],
        noise_case["p_inside"],
        noise_case["gt_starts"],
        noise_case["gt_ends"],
        noise_case["decoded"],
        title=f"Noise case | {noise_case['signer_id']} {noise_case['video_id']} {'left' if noise_case['hand']=='L' else 'right'}",
    )

    suffix = f"_{args.tag}" if args.tag else ""
    fig_e = out_root / f"figure_3_3e_shift_timeline{suffix}.png"
    fig_f = out_root / f"figure_3_3f_noise_timeline{suffix}.png"
    fig_e.write_bytes(shift_raw.read_bytes())
    fig_f.write_bytes(noise_raw.read_bytes())

    # also copy to paper figures/ch3 for direct thesis use
    figures_dir = out_root.parent / "figures" / "ch3"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / fig_e.name).write_bytes(fig_e.read_bytes())
    (figures_dir / fig_f.name).write_bytes(fig_f.read_bytes())

    def _case_brief(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "clip_id": f"{c['signer_id']}_{c['video_id']}",
            "video_id": c["video_id"],
            "signer_id": c["signer_id"],
            "hand": c["hand"],
            "gt_spans": c["gt_spans"],
            "pred_spans": c["pred_spans"],
            "start_peaks": c["start_peaks"],
            "end_peaks": c["end_peaks"],
            "f1_iou05": c["f1_iou05"],
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "mean_abs_offset": c["mean_abs_offset"],
            "jitter_score": c["jitter_score"],
        }

    selection_path = out_root / "figure_3_3_case_selection.json"
    if selection_path.exists():
        sel = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        sel = {"config": Path(args.config).name, "checkpoint": args.checkpoint, "tau": args.tau, "cases": {}}
    shift_key = "shift" if not args.tag else f"shift_{args.tag}"
    noise_key = "noise" if not args.tag else f"noise_{args.tag}"
    sel["cases"][shift_key] = _case_brief(shift_case)
    sel["cases"][noise_key] = _case_brief(noise_case)
    selection_path.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = out_root / "figure_3_3_manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        man = {"schema": "figure_3_3_cases_v1", "cases": {}, "figure_files": {}}
    man["cases"][shift_key] = _case_brief(shift_case)
    man["cases"][noise_key] = _case_brief(noise_case)
    man.setdefault("figure_files", {})
    fig_e_key = "3-3e" if not args.tag else f"3-3e-{args.tag}"
    fig_f_key = "3-3f" if not args.tag else f"3-3f-{args.tag}"
    man["figure_files"][fig_e_key] = str(fig_e)
    man["figure_files"][fig_f_key] = str(fig_f)
    manifest_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  shift_raw: {shift_raw}")
    print(f"  noise_raw: {noise_raw}")
    print(f"  figure_3_3e: {fig_e}")
    print(f"  figure_3_3f: {fig_f}")
    print(f"  case_selection: {selection_path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
