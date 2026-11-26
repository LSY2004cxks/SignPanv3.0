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
from src.models.tcn import BoundaryTCN
from src.utils.io import load_yaml, select_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render Figure 3-3 case timelines with unified in-figure titles.")
    ap.add_argument("--config", default="configs/train_data28_poseflow_gated_farneback.yaml")
    ap.add_argument("--checkpoint", default="logs/pose_flow_gated_farneback/run_20260301_210004/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--case-json", default="outputs/paper/dataset_stats/figure_3_3_case_selection.json")
    ap.add_argument("--out-root", default="outputs/paper/dataset_stats")
    ap.add_argument("--both-hands", action="store_true", help="Render both left/right figures for each case video.")
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


def _resolve_record(records: List[Any], signer_id: str, video_id: str):
    for r in records:
        if str(r.signer_id) == str(signer_id) and str(r.video_id) == str(video_id):
            return r
    return None


def _find_record_for_hand(records: List[Any], signer_id: str, video_id: str, hand: str):
    rec = _resolve_record(records, signer_id, video_id)
    if rec is None:
        return None
    if hand in ("L", "R"):
        return rec
    return None


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

    case_data = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    cases = case_data.get("cases", {})

    # Standard mapping for thesis figures
    # (case_key, figure_code[a-f], display_name, file_slug)
    mapping: List[Tuple[str, str, str, str]] = [
        ("success", "a", "Success", "success"),
        ("burst", "b", "Burst", "burst"),
        ("miss", "c", "Miss", "miss"),
        ("mismatch", "d", "Mismatch", "mismatch"),
        ("shift", "e", "Shift", "shift"),
        ("noise", "f", "Noise", "noise"),
    ]

    out_root = Path(args.out_root).resolve()
    fig_dir = out_root.parent / "figures" / "ch3"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for case_key, fig_code, case_name, file_slug in mapping:
        if case_key not in cases:
            continue
        c = cases[case_key]
        signer_id = str(c["signer_id"])
        video_id = str(c["video_id"])
        hand = str(c["hand"]).upper()
        rec = _resolve_record(records, signer_id, video_id)
        if rec is None:
            continue

        feat_data = feat_builder.build_from_pose_path(rec.pose_path)
        hands_to_render = ["L", "R"] if args.both_hands else [hand]
        for hh in hands_to_render:
            feat = feat_data[hh]["features"]
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)[0].cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))
            p_start, p_end, p_inside = p[:, 0], p[:, 1], p[:, 2]
            dec = decode_boundaries(
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

            gt_events = [ev for ev in rec.events_by_hand.get(hh, []) if ev.train_valid]
            gt_starts = [int(ev.start) for ev in gt_events]
            gt_ends = [int(ev.end) for ev in gt_events]
            hand_name = "left" if hh == "L" else "right"
            title = f"{case_name} case | {signer_id} {video_id} {hand_name}"
            if args.both_hands:
                out_file = out_root / f"figure_3_3{fig_code}_{file_slug}_timeline_{hand_name}.png"
            else:
                out_file = out_root / f"figure_3_3{fig_code}_{file_slug}_timeline.png"
            _plot_one(out_file, p_start, p_end, p_inside, gt_starts, gt_ends, dec, title)
            (fig_dir / out_file.name).write_bytes(out_file.read_bytes())
            rendered.append((f"{case_key}_{hand_name}", out_file))

    # update manifest figure files to the new paper paths for a-f
    manifest_path = out_root / "figure_3_3_manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        man.setdefault("figure_files", {})
        for case_key, fig_code, case_name, file_slug in mapping:
            p = out_root / f"figure_3_3{fig_code}_{file_slug}_timeline.png"
            if p.exists():
                man["figure_files"][f"3-3{fig_code}"] = str(p)
        manifest_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Rendered:")
    for k, p in rendered:
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
