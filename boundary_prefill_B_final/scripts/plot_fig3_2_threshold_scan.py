#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import BoundaryDataset, collate_batch
from src.data.indexer import build_manifest
from src.data.split import load_signer_split
from src.features.builder import FeatureBuilder
from src.infer.postprocess import decode_boundaries
from src.metrics.boundary import aggregate_boundary_counts, finalize_aggregate_boundary
from src.models.tcn import BoundaryTCN
from src.utils.io import ensure_dir, load_yaml, select_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Figure 3-2 threshold scan curves on val split.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="outputs/dataset_stats")
    ap.add_argument("--prefix", default="figure_3_2")
    return ap.parse_args()


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


def _tau_values(eval_cfg: Dict[str, Any]) -> List[float]:
    t0 = float(eval_cfg["tau_scan"]["start"])
    t1 = float(eval_cfg["tau_scan"]["end"])
    dt = float(eval_cfg["tau_scan"]["step"])
    vals = []
    t = t0
    while t <= t1 + 1e-8:
        vals.append(round(t, 4))
        t += dt
    return vals


def eval_once(model, loader, device, tau: float, infer_cfg: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    acc = {}
    windows = [7]

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].cpu().numpy().astype(int)
            probs = torch.sigmoid(model(x)).cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                T = int(lengths[i])
                ps = probs[i, :T, 0]
                pe = probs[i, :T, 1]
                pi = probs[i, :T, 2]
                dec = decode_boundaries(
                    p_start=ps,
                    p_end=pe,
                    p_inside=pi,
                    tau=float(tau),
                    peak_radius=int(infer_cfg["peak_radius"]),
                    nms_window=int(infer_cfg["w_nms"]),
                    Lmin=int(infer_cfg["Lmin"]),
                    Lmax=int(infer_cfg["Lmax"]),
                    inside_recommend_thr=float(infer_cfg["inside_recommend_thr"]),
                    pair_mode=str(infer_cfg.get("pair_mode", "score_then_nearest")),
                )

                pred_s = [int(x["t"]) for x in dec["start_peaks"]]
                pred_e = [int(x["t"]) for x in dec["end_peaks"]]
                gt_s = [int(x) for x in meta["gt_start"]]
                gt_e = [int(x) for x in meta["gt_end"]]
                aggregate_boundary_counts(gt_s, gt_e, pred_s, pred_e, windows, acc)

    b = finalize_aggregate_boundary(acc)[7]
    mean_p = 0.5 * (float(b["start_p"]) + float(b["end_p"]))
    mean_r = 0.5 * (float(b["start_r"]) + float(b["end_r"]))
    return {
        "start_p": float(b["start_p"]),
        "start_r": float(b["start_r"]),
        "start_f1": float(b["start_f1"]),
        "end_p": float(b["end_p"]),
        "end_r": float(b["end_r"]),
        "end_f1": float(b["end_f1"]),
        "mean_p": mean_p,
        "mean_r": mean_r,
        "mean_f1": float(b["mean_f1"]),
    }


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    cfg = load_yaml(args.config)

    split_obj = load_signer_split(cfg["splits"]["file"])
    signers = split_obj[args.split]
    manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )

    feature_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))
    expected_dim_cfg = cfg["features"].get("expected_dim", None)
    expected_dim = int(expected_dim_cfg) if expected_dim_cfg is not None else None
    ds = BoundaryDataset(
        manifest.records,
        feature_builder,
        sigma=float(cfg["labels"]["sigma"]),
        radius=int(cfg["labels"]["radius"]),
        use_valid_only=bool(cfg["labels"]["use_valid_only"]),
        expected_dim=expected_dim,
        skip_missing_pose=True,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_batch,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_cfg", cfg["model"])
    model = BoundaryTCN(**_model_kwargs(model_cfg, int(ckpt["input_dim"])))
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = select_device(args.device)
    model = model.to(device)

    rows = []
    taus = _tau_values(cfg["eval"])
    for tau in taus:
        m = eval_once(model, loader, device, tau=float(tau), infer_cfg=cfg["infer"])
        m["tau"] = float(tau)
        rows.append(m)

    best = max(rows, key=lambda x: x["mean_f1"])

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    csv_path = out_dir / f"{args.prefix}_val_tau_scan_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau",
                "start_p",
                "start_r",
                "start_f1",
                "end_p",
                "end_r",
                "end_f1",
                "mean_p",
                "mean_r",
                "mean_f1",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Figure 3-2a: Mean F1@7 vs tau
    x = [r["tau"] for r in rows]
    y = [r["mean_f1"] for r in rows]
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(x, y, color="black", linewidth=1.2, marker="o", markersize=3, markerfacecolor="lightgray")
    plt.axvline(best["tau"], color="dimgray", linestyle="--", linewidth=1.0)
    plt.scatter([best["tau"]], [best["mean_f1"]], color="black", s=20, zorder=3)
    plt.text(
        best["tau"],
        best["mean_f1"] + 0.005,
        f"tau*={best['tau']:.2f}, F1={best['mean_f1']:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    plt.xlabel("Threshold tau")
    plt.ylabel("Mean F1@7")
    plt.title("Figure 3-2a  Val Mean F1@7 vs tau")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.prefix}a_val_mean_f1_w7_vs_tau.png", dpi=300)
    plt.savefig(out_dir / f"{args.prefix}a_val_mean_f1_w7_vs_tau.pdf")
    plt.close()

    # Figure 3-2b: PR curve (start/end/mean)
    sr = [r["start_r"] for r in rows]
    sp = [r["start_p"] for r in rows]
    er = [r["end_r"] for r in rows]
    ep = [r["end_p"] for r in rows]
    mr = [r["mean_r"] for r in rows]
    mp = [r["mean_p"] for r in rows]
    plt.figure(figsize=(6.6, 3.3))
    plt.plot(sr, sp, color="black", linewidth=1.2, marker="o", markersize=3, label="Start PR")
    plt.plot(er, ep, color="dimgray", linewidth=1.2, marker="s", markersize=3, label="End PR")
    plt.plot(mr, mp, color="gray", linewidth=1.0, linestyle="--", marker="^", markersize=3, label="Mean PR")
    # mark best tau point on mean PR
    plt.scatter([best["mean_r"]], [best["mean_p"]], color="black", s=20, zorder=3)
    plt.text(best["mean_r"], best["mean_p"] + 0.01, f"tau*={best['tau']:.2f}", fontsize=8, ha="center")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Figure 3-2b  Val Precision-Recall (tau scan)")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.prefix}b_val_pr_curve.png", dpi=300)
    plt.savefig(out_dir / f"{args.prefix}b_val_pr_curve.pdf")
    plt.close()

    print(f"saved: {csv_path}")
    print(f"saved: {out_dir / f'{args.prefix}a_val_mean_f1_w7_vs_tau.png'}")
    print(f"saved: {out_dir / f'{args.prefix}a_val_mean_f1_w7_vs_tau.pdf'}")
    print(f"saved: {out_dir / f'{args.prefix}b_val_pr_curve.png'}")
    print(f"saved: {out_dir / f'{args.prefix}b_val_pr_curve.pdf'}")
    print(f"best_tau={best['tau']:.4f}, best_mean_f1@7={best['mean_f1']:.4f}")


if __name__ == "__main__":
    main()

