#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from src.models.tcn import BoundaryTCN
from src.utils.io import ensure_dir, load_yaml, parse_signer_list, select_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize timelines for boundary proposal.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--signers", default=None)
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--label-root", default=None)
    ap.add_argument("--pose-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--clip-ids", default=None, help="Comma-separated label clip ids (e.g. V6326,V6331)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=None)
    return ap.parse_args()


def _plot_one(
    out_path: str,
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

    for p in decoded["start_peaks"]:
        ax.scatter([p["t"]], [p["score"]], color="#d62728", s=18, zorder=3)
    for p in decoded["end_peaks"]:
        ax.scatter([p["t"]], [p["score"]], color="#9467bd", s=18, zorder=3)

    for s in decoded["spans_proposed"][:15]:
        ax.axvspan(int(s["start"]), int(s["end"]), color="#d62728", alpha=0.06)

    ax.set_xlim(0, max(1, T - 1))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _feature_builder_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    flow_cfg = cfg.get("features", {}).get("flow", {}) or {}
    return {
        "score_thr": float(cfg["features"]["score_thr"]),
        "median_k": int(cfg["features"]["median_k"]),
        "feature_mode": str(cfg.get("features", {}).get("mode", "auto")),
        "flow_enabled": bool(flow_cfg.get("enabled", False)),
        "flow_root": flow_cfg.get("root", None),
        "flow_feature_keys": flow_cfg.get("feature_keys", ["mu", "p95", "p99", "delta_mu", "u_mean", "v_mean"]),
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


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    cfg = load_yaml(args.config)
    if args.video_root:
        cfg["data"]["video_root"] = args.video_root
    if args.label_root:
        cfg["data"]["label_root"] = args.label_root
    if args.pose_root:
        cfg["data"]["pose_root"] = args.pose_root
    if args.split_file:
        cfg["splits"]["file"] = args.split_file
    if args.device:
        cfg["runtime"]["device"] = args.device

    split_obj = load_signer_split(cfg["splits"]["file"])
    signers = parse_signer_list(args.signers) or split_obj.get(args.split, [])

    manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )

    records = [r for r in manifest.records if r.pose_path is not None]
    if not records:
        raise SystemExit("No records for visualization.")

    if args.clip_ids:
        wanted = {x.strip() for x in args.clip_ids.split(",") if x.strip()}
        records = [r for r in records if r.clip_id in wanted]
        if not records:
            raise SystemExit("No records matched --clip-ids.")

    random.seed(args.seed)
    random.shuffle(records)
    records = records[: args.K]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_cfg", cfg["model"])
    model = BoundaryTCN(**_model_kwargs(model_cfg, int(ckpt["input_dim"])))
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = select_device(cfg["runtime"].get("device", "auto"))
    model = model.to(device).eval()

    feat_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))

    out_dir = Path(args.output_dir or cfg["paths"]["vis_dir"])
    ensure_dir(str(out_dir))

    n_saved = 0
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
                peak_radius=int(cfg["infer"]["peak_radius"]),
                nms_window=int(cfg["infer"]["w_nms"]),
                Lmin=int(cfg["infer"]["Lmin"]),
                Lmax=int(cfg["infer"]["Lmax"]),
                inside_recommend_thr=float(cfg["infer"]["inside_recommend_thr"]),
                pair_mode=str(cfg["infer"].get("pair_mode", "score_then_nearest")),
            )

            gt_events = rec.events_by_hand.get(hand, [])
            gt_starts = [int(ev.start) for ev in gt_events if ev.train_valid]
            gt_ends = [int(ev.end) for ev in gt_events if ev.train_valid]

            hand_name = "left" if hand == "L" else "right"
            out_path = out_dir / f"{args.split}_{rec.video_id}_{hand_name}.png"
            _plot_one(
                out_path=str(out_path),
                p_start=p_start,
                p_end=p_end,
                p_inside=p_inside,
                gt_starts=gt_starts,
                gt_ends=gt_ends,
                decoded=decoded,
                title=f"{rec.video_id} | {rec.signer_id} | {hand_name}",
            )
            n_saved += 1

    print(f"Saved {n_saved} timeline figures -> {out_dir}")


if __name__ == "__main__":
    main()
