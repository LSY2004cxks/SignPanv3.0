#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from data.dataset import HandshapeEventDataset, collate_fn
from metrics import aggregate_metrics, margin_and_entropy, tierwise_metrics
from models import HandshapeNet
from runtime_utils import load_config, probs_topk


def enforce_roi_skel_only(cfg: Dict[str, Any]) -> None:
    mcfg = cfg.get("model", {})
    if not bool(mcfg.get("enforce_roi_skel_only", False)):
        return
    mode = str(mcfg.get("mode", ""))
    if mode not in {"roi_skel_fusion", "roi_skel_deform_nce"}:
        raise ValueError(
            "model.enforce_roi_skel_only=true requires model.mode in "
            "{roi_skel_fusion, roi_skel_deform_nce}"
        )


def write_predictions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "sample_id",
        "gt",
        "top1_label",
        "top1_prob",
        "top2_label",
        "top2_prob",
        "top3_label",
        "top3_prob",
        "margin",
        "entropy",
        "tier",
        "autofill_top1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["label_idx"] = out["label_idx"].to(device, non_blocking=True)
    if "roi" in out:
        out["roi"] = out["roi"].to(device, non_blocking=True)
    if "skeleton" in out:
        out["skeleton"] = out["skeleton"].to(device, non_blocking=True)
    return out


def build_dataset(cfg: Dict[str, Any], split: str, mode: str, label_vocab: Dict[str, int] | None = None):
    data_cfg = cfg["data"]
    ds = HandshapeEventDataset(
        manifest_path=data_cfg["manifest_path"],
        split_dir=data_cfg["split_dir"],
        split=split,
        mode=mode,
        k=int(data_cfg["k"]),
        cache_roi_dir=data_cfg["cache_roi_dir"],
        cache_skeleton_dir=data_cfg["cache_skeleton_dir"],
        cache_meta_dir=data_cfg["cache_meta_dir"],
        tier_weights=data_cfg["tier_weights"],
        on_the_fly_fallback=bool(data_cfg.get("on_the_fly_fallback", True)),
    )

    if label_vocab is not None:
        # enforce training label order for compatibility
        ds.label_to_idx = dict(label_vocab)
        ds.idx_to_label = {i: lb for lb, i in ds.label_to_idx.items()}
        for s in ds.samples:
            s["label_idx"] = ds.label_to_idx[s["label"]]
        ds.num_classes = len(ds.label_to_idx)

    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_metrics", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    enforce_roi_skel_only(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    label_to_idx = ckpt.get("label_to_idx")
    idx_to_label = ckpt.get("idx_to_label")
    if idx_to_label is None and label_to_idx is not None:
        idx_to_label = {i: lb for lb, i in label_to_idx.items()}

    model_cfg = cfg["model"]
    mode = model_cfg["mode"]

    ds = build_dataset(cfg, split=args.split, mode=mode, label_vocab=label_to_idx)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = HandshapeNet(
        mode=mode,
        num_classes=ds.num_classes,
        roi_backbone=model_cfg.get("roi_backbone", "convnext_tiny"),
        roi_dim=int(model_cfg.get("roi_dim", 512)),
        skel_dim=int(model_cfg.get("skel_dim", 256)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logits_all = []
    targets_all = []
    tiers: List[str] = []
    sample_ids: List[str] = []
    gt_names: List[str] = []

    with torch.no_grad():
        for batch in dl:
            b = batch_to_device(batch, device)
            out = model(b)
            logits = out["logits"]
            logits_all.append(logits.detach().cpu())
            targets_all.append(b["label_idx"].detach().cpu())
            tiers.extend([str(t) for t in batch["tier"]])
            sample_ids.extend(batch["sample_id"])
            gt_names.extend(batch["gt"])

    if not logits_all:
        raise RuntimeError("No samples in split")

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    metrics = aggregate_metrics(logits_all, targets_all)
    metrics["tierwise"] = tierwise_metrics(logits_all, targets_all, tiers)

    topk_idx, topk_prob = probs_topk(logits_all, k=3)
    margins, entropy = margin_and_entropy(logits_all)

    gate_cfg = cfg.get("inference", {}).get("gating", {})
    gate_margin = float(gate_cfg.get("margin_min", 0.2))
    gate_p1 = float(gate_cfg.get("p1_min", 0.7))

    rows: List[Dict[str, Any]] = []
    for i in range(logits_all.shape[0]):
        p1 = float(topk_prob[i, 0]) if topk_prob.shape[1] > 0 else 0.0
        p2 = float(topk_prob[i, 1]) if topk_prob.shape[1] > 1 else 0.0
        p3 = float(topk_prob[i, 2]) if topk_prob.shape[1] > 2 else 0.0
        l1 = idx_to_label[int(topk_idx[i, 0])]
        l2 = idx_to_label[int(topk_idx[i, 1])] if topk_idx.shape[1] > 1 else ""
        l3 = idx_to_label[int(topk_idx[i, 2])] if topk_idx.shape[1] > 2 else ""
        tier = str(tiers[i]).upper()
        m = float(margins[i])
        e = float(entropy[i])
        autofill = bool((tier in {"A", "B"}) and (m >= gate_margin) and (p1 >= gate_p1))

        rows.append(
            {
                "sample_id": sample_ids[i],
                "gt": gt_names[i],
                "top1_label": l1,
                "top1_prob": f"{p1:.6f}",
                "top2_label": l2,
                "top2_prob": f"{p2:.6f}",
                "top3_label": l3,
                "top3_prob": f"{p3:.6f}",
                "margin": f"{m:.6f}",
                "entropy": f"{e:.6f}",
                "tier": tier,
                "autofill_top1": str(autofill).lower(),
            }
        )

    ckpt_dir = Path(args.ckpt).resolve().parent
    out_csv = Path(args.out_csv).resolve() if args.out_csv else ckpt_dir / f"{args.split}_predictions.csv"
    out_metrics = Path(args.out_metrics).resolve() if args.out_metrics else ckpt_dir / f"metrics_{args.split}.json"

    write_predictions_csv(out_csv, rows)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"split": args.split, "out_csv": str(out_csv), "out_metrics": str(out_metrics), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
