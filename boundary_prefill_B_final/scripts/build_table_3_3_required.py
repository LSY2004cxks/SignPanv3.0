#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.metrics.span import counts_to_prf, match_spans_iou
from src.models.tcn import BoundaryTCN
from src.utils.io import load_yaml, select_device


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _fmt_or_dash(x):
    if isinstance(x, str):
        return x
    return f"{float(x):.4f}"


def compute_after_metrics(
    config_path: str,
    checkpoint_path: str,
    tau: float,
    device_str: str,
    split: str = "test",
) -> Tuple[float, float, float]:
    cfg = load_yaml(config_path)
    split_obj = load_signer_split(cfg["splits"]["file"])
    signers = split_obj[split]

    manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )

    flow_cfg = cfg.get("features", {}).get("flow", {}) or {}
    fb = FeatureBuilder(
        score_thr=float(cfg["features"]["score_thr"]),
        median_k=int(cfg["features"]["median_k"]),
        feature_mode=str(cfg.get("features", {}).get("mode", "auto")),
        flow_enabled=bool(flow_cfg.get("enabled", False)),
        flow_root=flow_cfg.get("root", None),
        flow_feature_keys=flow_cfg.get("feature_keys", ["mu", "p95", "p99", "delta_mu", "u_mean", "v_mean"]),
        flow_allow_missing=bool(flow_cfg.get("allow_missing", True)),
        flow_feature_dim=int(flow_cfg.get("feature_dim", 6)),
    )
    expected_dim_cfg = cfg["features"].get("expected_dim", None)
    expected_dim = int(expected_dim_cfg) if expected_dim_cfg is not None else None

    ds = BoundaryDataset(
        manifest.records,
        fb,
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

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    mcfg = ckpt.get("model_cfg", cfg["model"])
    model = BoundaryTCN(
        input_dim=int(ckpt["input_dim"]),
        channels=int(mcfg["channels"]),
        layers=int(mcfg["layers"]),
        dilations=[int(x) for x in mcfg["dilations"]],
        kernel_size=int(mcfg["kernel_size"]),
        dropout=float(mcfg["dropout"]),
        fusion_mode=str(mcfg.get("fusion_mode", "early")),
        pose_dim=int(mcfg.get("pose_dim", 23)),
        flow_dim=int(mcfg.get("flow_dim", 6)),
        fusion_dim=int(mcfg.get("fusion_dim", 64)),
        gate_hidden=int(mcfg.get("gate_hidden", 64)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = select_device(device_str)
    model = model.to(device).eval()

    infer_cfg = cfg["infer"]
    tp = fp = fn = 0
    props_before: Dict[str, int] = {}
    props_after: Dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].cpu().numpy().astype(int)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            for i, meta in enumerate(batch["meta"]):
                T = int(lengths[i])
                ps, pe, pi = probs[i, :T, 0], probs[i, :T, 1], probs[i, :T, 2]
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

                pred_before = [(int(s["start"]), int(s["end"])) for s in dec["spans_proposed"]]
                pred_after = [(int(s["start"]), int(s["end"])) for s in dec["spans_recommended"]]
                gt = [(int(a), int(b)) for a, b in meta["gt_spans"]]
                a, b, c = match_spans_iou(gt, pred_after, iou_thr=0.5)
                tp += a
                fp += b
                fn += c

                vkey = f"{meta.get('signer_id','')}|{meta.get('video_id','')}"
                props_before[vkey] = props_before.get(vkey, 0) + len(pred_before)
                props_after[vkey] = props_after.get(vkey, 0) + len(pred_after)

    f1_after = float(counts_to_prf(tp, fp, fn)["f1"])
    avg_before = float(np.mean(list(props_before.values()))) if props_before else 0.0
    avg_after = float(np.mean(list(props_after.values()))) if props_after else 0.0
    return f1_after, avg_before, avg_after


def main():
    ap = argparse.ArgumentParser(description="Build required Table 3-3 format.")
    ap.add_argument("--pose-report", required=True)
    ap.add_argument("--inside-report", required=True)
    ap.add_argument("--soft-report", required=True)
    ap.add_argument("--soft-config", required=True)
    ap.add_argument("--soft-checkpoint", required=True)
    ap.add_argument("--soft-tau", type=float, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-prefix", default="outputs/dataset_stats/table_3_3_span_f1_required_format")
    args = ap.parse_args()

    pose = _load_json(args.pose_report)
    ins = _load_json(args.inside_report)
    soft = _load_json(args.soft_report)

    pose_f1 = float(pose["fixed_metrics"]["span_by_iou"]["0.5"]["f1"])
    pose_before = float(pose["fixed_metrics"]["avg_proposals_per_video"])
    ins_f1 = float(ins["fixed_metrics"]["span_by_iou"]["0.5"]["f1"])
    ins_before = float(ins["fixed_metrics"]["avg_proposals_per_video"])
    soft_f1 = float(soft["fixed_metrics"]["span_by_iou"]["0.5"]["f1"])
    soft_before = float(soft["fixed_metrics"]["avg_proposals_per_video"])

    val_f1_after, val_before, val_after = compute_after_metrics(
        config_path=args.soft_config,
        checkpoint_path=args.soft_checkpoint,
        tau=float(args.soft_tau),
        device_str=args.device,
        split="test",
    )

    rows: List[Dict] = [
        {
            "Method": "Pose-only",
            "Event F1 (IoU>=0.5)": pose_f1,
            "Avg proposals/video (before)": pose_before,
            "Avg proposals/video (after)": "—",
        },
        {
            "Method": "+ inside head",
            "Event F1 (IoU>=0.5)": ins_f1,
            "Avg proposals/video (before)": ins_before,
            "Avg proposals/video (after)": "—",
        },
        {
            "Method": "+ soft boundary",
            "Event F1 (IoU>=0.5)": soft_f1,
            "Avg proposals/video (before)": soft_before,
            "Avg proposals/video (after)": "—",
        },
        {
            "Method": "+ validity filter",
            "Event F1 (IoU>=0.5)": val_f1_after,
            "Avg proposals/video (before)": val_before,
            "Avg proposals/video (after)": val_after,
        },
    ]

    out_prefix = Path(args.out_prefix)
    md = out_prefix.with_suffix(".md")
    csvp = out_prefix.with_suffix(".csv")
    jsonp = out_prefix.with_suffix(".json")

    md.parent.mkdir(parents=True, exist_ok=True)
    with open(md, "w", encoding="utf-8") as f:
        f.write("| Method | Event F1 (IoU>=0.5) | Avg proposals/video (before) | Avg proposals/video (after) |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['Method']} | {_fmt_or_dash(r['Event F1 (IoU>=0.5)'])} | "
                f"{_fmt_or_dash(r['Avg proposals/video (before)'])} | {_fmt_or_dash(r['Avg proposals/video (after)'])} |\n"
            )

    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Method",
                "Event F1 (IoU>=0.5)",
                "Avg proposals/video (before)",
                "Avg proposals/video (after)",
                "note",
            ]
        )
        note = "tau_policy=val_best_from_training(test); +validity row uses spans_recommended"
        for r in rows:
            w.writerow(
                [
                    r["Method"],
                    _fmt_or_dash(r["Event F1 (IoU>=0.5)"]),
                    _fmt_or_dash(r["Avg proposals/video (before)"]),
                    _fmt_or_dash(r["Avg proposals/video (after)"]),
                    note,
                ]
            )

    _save_json(
        str(jsonp),
        {
            "schema": "table_3_3_required_format_v2",
            "rows": rows,
            "sources": {
                "pose_report": args.pose_report,
                "inside_report": args.inside_report,
                "soft_report": args.soft_report,
                "soft_config": args.soft_config,
                "soft_checkpoint": args.soft_checkpoint,
                "soft_tau": args.soft_tau,
            },
        },
    )
    print("written", md)
    print("written", csvp)
    print("written", jsonp)


if __name__ == "__main__":
    main()

