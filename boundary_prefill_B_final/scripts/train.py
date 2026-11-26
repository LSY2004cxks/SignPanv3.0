#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import BoundaryDataset, collate_batch
from src.data.indexer import build_manifest
from src.data.split import get_split_signers, load_signer_split
from src.features.builder import FeatureBuilder
from src.infer.postprocess import decode_boundaries
from src.metrics.boundary import aggregate_boundary_counts, finalize_aggregate_boundary
from src.metrics.span import counts_to_prf, match_spans_iou
from src.models.losses import boundary_loss
from src.models.tcn import BoundaryTCN
from src.utils.io import (
    append_csv,
    ensure_dir,
    load_yaml,
    make_run_dir,
    parse_signer_list,
    save_json,
    save_yaml,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train single-hand boundary proposal model (Stage A).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--label-root", default=None)
    ap.add_argument("--pose-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--train-signers", default=None, help="Comma-separated override")
    ap.add_argument("--val-signers", default=None, help="Comma-separated override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-clips", type=int, default=None, help="Debug limit on number of clips per split")
    ap.add_argument("--log-interval", type=int, default=None, help="Print training progress every N steps")
    return ap.parse_args()


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.video_root:
        cfg["data"]["video_root"] = args.video_root
    if args.label_root:
        cfg["data"]["label_root"] = args.label_root
    if args.pose_root:
        cfg["data"]["pose_root"] = args.pose_root
    if args.split_file:
        cfg["splits"]["file"] = args.split_file
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.log_interval is not None:
        cfg.setdefault("train", {})
        cfg["train"]["log_interval"] = int(args.log_interval)
    return cfg


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


def _build_loader(dataset: BoundaryDataset, batch_size: int, shuffle: bool, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def _evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    tau: float,
    infer_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
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
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                T = int(lengths[i])
                ps = probs[i, :T, 0]
                pe = probs[i, :T, 1]
                pi = probs[i, :T, 2]

                decoded = decode_boundaries(
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

                pred_s = [int(x["t"]) for x in decoded["start_peaks"]]
                pred_e = [int(x["t"]) for x in decoded["end_peaks"]]
                gt_s = [int(x) for x in meta["gt_start"]]
                gt_e = [int(x) for x in meta["gt_end"]]

                aggregate_boundary_counts(gt_s, gt_e, pred_s, pred_e, windows, acc)

                pred_spans = [(int(s["start"]), int(s["end"])) for s in decoded["spans_proposed"]]
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


def _build_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "AdamW")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unsupported optimizer={train_cfg.get('optimizer')}, expected AdamW or Adam")


def _collect_code_version() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        info["git_hash"] = git_hash
    except Exception:
        info["git_hash"] = "not_available"
    return info


def _fps_report(records: List[Any], expected_fps: Optional[float]) -> Dict[str, Any]:
    fps_values = [float(r.fps) for r in records if r.fps is not None]
    out = {
        "expected_fps": expected_fps,
        "num_records": len(records),
        "num_with_fps": len(fps_values),
    }
    if fps_values:
        uniq = sorted(set(round(x, 3) for x in fps_values))
        out["unique_fps"] = uniq
        out["mean_fps"] = float(np.mean(fps_values))
        if expected_fps is not None:
            bad = [x for x in fps_values if abs(float(x) - float(expected_fps)) > 1e-3]
            out["num_mismatch_expected"] = len(bad)
    return out


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    cfg = load_yaml(args.config)
    cfg = _apply_overrides(cfg, args)

    seed = int(cfg["project"].get("seed", 7))
    set_seed(seed)

    run_dir = make_run_dir(cfg["paths"]["logs_dir"])
    ensure_dir(run_dir)
    save_yaml(str(Path(run_dir) / "config.yaml"), cfg)
    save_json(str(Path(run_dir) / "code_version.json"), _collect_code_version())

    device = select_device(cfg["runtime"].get("device", "auto"))

    split_obj = load_signer_split(cfg["splits"]["file"])
    train_signers = get_split_signers(split_obj, "train", parse_signer_list(args.train_signers))
    val_signers = get_split_signers(split_obj, "val", parse_signer_list(args.val_signers))

    train_manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=train_signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )
    val_manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=val_signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )

    if args.max_clips is not None:
        random.seed(seed)
        random.shuffle(train_manifest.records)
        random.shuffle(val_manifest.records)
        train_manifest.records = train_manifest.records[: args.max_clips]
        val_manifest.records = val_manifest.records[: max(1, args.max_clips // 2)]

    if not train_manifest.records:
        raise SystemExit("No training records found. Check data roots and split signers.")
    if not val_manifest.records:
        raise SystemExit("No validation records found. Check data roots and split signers.")

    expected_fps = cfg.get("data", {}).get("expected_fps", None)
    fps_info = {
        "train": _fps_report(train_manifest.records, expected_fps),
        "val": _fps_report(val_manifest.records, expected_fps),
    }
    save_json(str(Path(run_dir) / "fps_report.json"), fps_info)

    feature_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))
    expected_dim_cfg = cfg["features"].get("expected_dim", None)
    expected_dim = int(expected_dim_cfg) if expected_dim_cfg is not None else None

    train_ds = BoundaryDataset(
        train_manifest.records,
        feature_builder,
        sigma=float(cfg["labels"]["sigma"]),
        radius=int(cfg["labels"]["radius"]),
        use_valid_only=bool(cfg["labels"]["use_valid_only"]),
        expected_dim=expected_dim,
        skip_missing_pose=True,
    )
    val_ds = BoundaryDataset(
        val_manifest.records,
        feature_builder,
        sigma=float(cfg["labels"]["sigma"]),
        radius=int(cfg["labels"]["radius"]),
        use_valid_only=bool(cfg["labels"]["use_valid_only"]),
        expected_dim=expected_dim,
        skip_missing_pose=True,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Dataset became empty after pose filtering.")

    train_loader = _build_loader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    sample = train_ds[0]
    input_dim = int(sample["x"].shape[1])

    model = BoundaryTCN(**_model_kwargs(cfg["model"], input_dim)).to(device)

    optim = _build_optimizer(model, cfg["train"])

    metrics_csv = str(Path(run_dir) / "metrics.csv")
    metric_fields = [
        "epoch",
        "train_loss",
        "val_mean_f1_w3",
        "val_mean_f1_w5",
        "val_mean_f1_w7",
        "val_span_f1",
    ]

    best_f1 = -1.0
    best_path = str(Path(run_dir) / "best.pt")
    es_cfg = cfg.get("train", {}).get("early_stop", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_bad_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0
        n_batches = 0
        epoch_t0 = time.time()
        total_steps = max(1, len(train_loader))
        log_interval = int(cfg.get("train", {}).get("log_interval", 0))

        for step_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device)
            y_start = batch["y_start"].to(device)
            y_end = batch["y_end"].to(device)
            y_inside = batch["y_inside"].to(device)
            mask = batch["mask"].to(device)

            optim.zero_grad()
            logits = model(x)
            losses = boundary_loss(
                logits,
                y_start,
                y_end,
                y_inside,
                mask,
                focal_alpha=float(cfg["loss"]["focal_alpha"]),
                focal_gamma=float(cfg["loss"]["focal_gamma"]),
                inside_weight=float(cfg["loss"]["inside_weight"]),
            )
            losses["loss"].backward()
            clip = float(cfg["train"].get("clip_grad_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

            running += float(losses["loss"].item())
            n_batches += 1

            if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == total_steps):
                elapsed = max(1e-6, time.time() - epoch_t0)
                sec_per_step = elapsed / float(step_idx)
                eta_sec = max(0.0, (total_steps - step_idx) * sec_per_step)
                print(
                    f"[Epoch {epoch:03d}] step {step_idx}/{total_steps} "
                    f"loss={running / max(1, n_batches):.4f} "
                    f"time={sec_per_step:.2f}s/step ETA={eta_sec/60.0:.1f}m",
                    flush=True,
                )

        val_res = _evaluate(
            model,
            val_loader,
            device,
            tau=float(cfg["infer"]["tau"]),
            infer_cfg=cfg["infer"],
            eval_cfg=cfg["eval"],
        )

        val_w3 = val_res["boundary"].get(3, {}).get("mean_f1", 0.0)
        val_w5 = val_res["boundary"].get(5, {}).get("mean_f1", 0.0)
        val_w7 = val_res["boundary"].get(7, {}).get("mean_f1", 0.0)

        row = {
            "epoch": epoch,
            "train_loss": running / max(1, n_batches),
            "val_mean_f1_w3": val_w3,
            "val_mean_f1_w5": val_w5,
            "val_mean_f1_w7": val_w7,
            "val_span_f1": val_res["span"]["f1"],
        }
        append_csv(metrics_csv, row, metric_fields)

        print(
            f"Epoch {epoch:03d} | loss={row['train_loss']:.4f} | "
            f"F1@3={val_w3:.4f} F1@5={val_w5:.4f} F1@7={val_w7:.4f} spanF1={row['val_span_f1']:.4f}"
        )

        if val_w7 > best_f1:
            best_f1 = val_w7
            es_bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "model_cfg": cfg["model"],
                "feature_dim": input_dim,
                "seed": seed,
            }
            torch.save(ckpt, best_path)
        else:
            if (val_w7 + es_min_delta) <= best_f1:
                es_bad_epochs += 1

        if es_enabled and es_bad_epochs >= es_patience:
            print(
                f"Early stop at epoch {epoch}: no improvement on val_mean_f1_w7 "
                f"for {es_bad_epochs} epochs (patience={es_patience})."
            )
            break

    if not Path(best_path).exists():
        raise SystemExit("Training ended without checkpoint.")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    tau_rows: List[Dict[str, Any]] = []
    best_tau = float(cfg["infer"]["tau"])
    best_tau_f1 = -1.0
    for tau in _tau_values(cfg["eval"]):
        val_res = _evaluate(
            model,
            val_loader,
            device,
            tau=float(tau),
            infer_cfg=cfg["infer"],
            eval_cfg=cfg["eval"],
        )
        f1 = val_res["boundary"][int(cfg["eval"]["primary_window"])]["mean_f1"]
        tau_rows.append(
            {
                "tau": tau,
                "f1_w7": f1,
                "f1_w5": val_res["boundary"][5]["mean_f1"],
                "f1_w3": val_res["boundary"][3]["mean_f1"],
            }
        )
        if f1 > best_tau_f1:
            best_tau_f1 = f1
            best_tau = tau

    tau_csv = str(Path(run_dir) / "tau_scan.csv")
    with open(tau_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "f1_w7", "f1_w5", "f1_w3"])
        writer.writeheader()
        for r in tau_rows:
            writer.writerow(r)

    plt.figure(figsize=(6, 4))
    plt.plot([r["tau"] for r in tau_rows], [r["f1_w7"] for r in tau_rows], marker="o")
    plt.xlabel("tau")
    plt.ylabel("F1@7")
    plt.title("Val tau scan")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(Path(run_dir) / "tau_scan.png"), dpi=150)
    plt.close()

    save_json(str(Path(run_dir) / "best_tau.json"), {"best_tau": best_tau, "best_f1_w7": best_tau_f1})

    # schema + mapping report
    schema_report = {
        "train": train_manifest.schema_report,
        "val": val_manifest.schema_report,
        "feature_mapping": feature_builder.get_report(),
        "device": str(device),
    }
    save_json(str(Path(run_dir) / "schema_report.json"), schema_report)
    save_json(str(Path(cfg["paths"]["logs_dir"]) / "schema_report.json"), schema_report)

    print(f"Best checkpoint: {best_path}")
    print(f"Best tau on val (F1@7): {best_tau:.3f}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
