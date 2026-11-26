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
from src.metrics.span import counts_to_prf, match_spans_iou
from src.models.tcn import BoundaryTCN
from src.utils.io import (
    ensure_dir,
    load_yaml,
    make_run_dir,
    parse_signer_list,
    save_json,
    select_device,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate boundary proposal model.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", help="train|val|test")
    ap.add_argument("--signers", default=None)
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--label-root", default=None)
    ap.add_argument("--pose-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method-name", default="current", help="Row label for exported metric tables")
    return ap.parse_args()


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


def evaluate_once(model, loader, device, tau, infer_cfg, eval_cfg):
    model.eval()
    windows = [int(x) for x in eval_cfg["windows"]]
    acc = {}
    iou_thrs = [float(x) for x in eval_cfg.get("iou_thresholds", [float(eval_cfg.get("iou_thr", 0.5))])]
    span_acc = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in iou_thrs}
    proposals_by_video = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].cpu().numpy().astype(int)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            for i, meta in enumerate(batch["meta"]):
                T = int(lengths[i])
                ps, pe, pi = probs[i, :T, 0], probs[i, :T, 1], probs[i, :T, 2]
                dec = decode_boundaries(
                    ps,
                    pe,
                    pi,
                    tau=tau,
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

                pred_spans = [(int(s["start"]), int(s["end"])) for s in dec["spans_proposed"]]
                gt_spans = [(int(a), int(b)) for a, b in meta["gt_spans"]]
                for thr in iou_thrs:
                    tp, fp, fn = match_spans_iou(gt_spans, pred_spans, iou_thr=float(thr))
                    span_acc[thr]["tp"] += tp
                    span_acc[thr]["fp"] += fp
                    span_acc[thr]["fn"] += fn
                vkey = f"{meta.get('signer_id','')}|{meta.get('video_id','')}"
                proposals_by_video[vkey] = proposals_by_video.get(vkey, 0) + len(pred_spans)

    boundary = finalize_aggregate_boundary(acc)
    span_by_iou = {
        str(thr): counts_to_prf(v["tp"], v["fp"], v["fn"])
        for thr, v in span_acc.items()
    }
    primary_iou = float(eval_cfg.get("iou_thr", 0.5))
    span = span_by_iou.get(str(primary_iou), counts_to_prf(0, 0, 0))
    avg_props = float(np.mean(list(proposals_by_video.values()))) if proposals_by_video else 0.0
    return {
        "boundary": boundary,
        "span": span,
        "span_by_iou": span_by_iou,
        "avg_proposals_per_video": avg_props,
    }


def _mean_f1_for_window(res: Dict[str, Any], window: int) -> float:
    b = res.get("boundary", {})
    m = b.get(int(window), {})
    return float(m.get("mean_f1", 0.0))


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
    records = manifest.records[: args.limit] if args.limit else manifest.records
    if not records:
        raise SystemExit("No records for evaluation.")

    feature_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))
    expected_dim_cfg = cfg["features"].get("expected_dim", None)
    expected_dim = int(expected_dim_cfg) if expected_dim_cfg is not None else None
    ds = BoundaryDataset(
        records,
        feature_builder,
        sigma=float(cfg["labels"]["sigma"]),
        radius=int(cfg["labels"]["radius"]),
        use_valid_only=bool(cfg["labels"]["use_valid_only"]),
        expected_dim=expected_dim,
        skip_missing_pose=True,
    )
    if len(ds) == 0:
        raise SystemExit("Dataset is empty after pose filtering. Provide pose cache first.")
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
    device = select_device(cfg["runtime"].get("device", "auto"))
    model = model.to(device)

    fixed = evaluate_once(model, loader, device, tau=float(args.tau), infer_cfg=cfg["infer"], eval_cfg=cfg["eval"])

    tau_rows = []
    eval_windows = [int(x) for x in cfg["eval"].get("windows", [7])]
    primary_window = int(cfg["eval"].get("primary_window", eval_windows[0]))
    if primary_window not in eval_windows:
        eval_windows.append(primary_window)
    best_tau = None
    best_f1 = -1.0
    for tau in _tau_values(cfg["eval"]):
        res = evaluate_once(model, loader, device, tau=float(tau), infer_cfg=cfg["infer"], eval_cfg=cfg["eval"])
        f1 = _mean_f1_for_window(res, primary_window)
        tau_rows.append(
            {
                "tau": tau,
                "f1_w7": _mean_f1_for_window(res, 7),
                "f1_w5": _mean_f1_for_window(res, 5),
                "f1_w3": _mean_f1_for_window(res, 3),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    out_dir = make_run_dir(cfg["paths"]["logs_dir"])
    save_yaml(str(Path(out_dir) / "config.yaml"), cfg)

    tau_csv = Path(out_dir) / "tau_scan.csv"
    with open(tau_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "f1_w7", "f1_w5", "f1_w3"])
        writer.writeheader()
        for r in tau_rows:
            writer.writerow(r)

    plt.figure(figsize=(6, 4))
    plt.plot([r["tau"] for r in tau_rows], [r["f1_w7"] for r in tau_rows], marker="o")
    plt.xlabel("tau")
    plt.ylabel("F1@7")
    plt.title(f"tau scan ({args.split})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "tau_scan.png", dpi=150)
    plt.close()

    report = {
        "split": args.split,
        "signers": signers,
        "fixed_tau": float(args.tau),
        "fixed_metrics": fixed,
        "best_tau": best_tau,
        "best_f1_w7": best_f1,
        "schema_report": manifest.schema_report,
        "feature_mapping": feature_builder.get_report(),
    }
    save_json(str(Path(out_dir) / "evaluation_report.json"), report)

    # Export compact tables for thesis sections.
    b3 = fixed["boundary"].get(3, {})
    b5 = fixed["boundary"].get(5, {})
    b7 = fixed["boundary"].get(7, {})
    table32 = Path(out_dir) / "table_3_2_frame_f1.csv"
    with open(table32, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Start F1@3",
                "End F1@3",
                "Mean@3",
                "Mean@5",
                "Mean@7",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "Method": args.method_name,
                "Start F1@3": b3.get("start_f1", 0.0),
                "End F1@3": b3.get("end_f1", 0.0),
                "Mean@3": b3.get("mean_f1", 0.0),
                "Mean@5": b5.get("mean_f1", 0.0),
                "Mean@7": b7.get("mean_f1", 0.0),
            }
        )

    span_iou = fixed.get("span_by_iou", {})
    table33 = Path(out_dir) / "table_3_3_span_f1.csv"
    with open(table33, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "IoU@0.3 F1",
                "IoU@0.5 F1",
                "IoU@0.7 F1",
                "Avg proposals/video",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "Method": args.method_name,
                "IoU@0.3 F1": (span_iou.get("0.3") or {}).get("f1", 0.0),
                "IoU@0.5 F1": (span_iou.get("0.5") or {}).get("f1", 0.0),
                "IoU@0.7 F1": (span_iou.get("0.7") or {}).get("f1", 0.0),
                "Avg proposals/video": fixed.get("avg_proposals_per_video", 0.0),
            }
        )

    print(f"Evaluation output -> {out_dir}")
    print(f"Best tau={best_tau} F1@7={best_f1:.4f}")


if __name__ == "__main__":
    main()
