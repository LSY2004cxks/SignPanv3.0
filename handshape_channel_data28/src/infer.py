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
from metrics import margin_and_entropy
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_predjson", type=str, default=None)
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
    data_cfg = cfg["data"]

    ds = HandshapeEventDataset(
        manifest_path=data_cfg["manifest_path"],
        split_dir=data_cfg["split_dir"],
        split=args.split,
        mode=mode,
        k=int(data_cfg["k"]),
        cache_roi_dir=data_cfg["cache_roi_dir"],
        cache_skeleton_dir=data_cfg["cache_skeleton_dir"],
        cache_meta_dir=data_cfg["cache_meta_dir"],
        tier_weights=data_cfg["tier_weights"],
        on_the_fly_fallback=bool(data_cfg.get("on_the_fly_fallback", True)),
    )

    if label_to_idx is not None:
        ds.label_to_idx = dict(label_to_idx)
        ds.idx_to_label = {i: lb for lb, i in ds.label_to_idx.items()}
        for s in ds.samples:
            s["label_idx"] = ds.label_to_idx[s["label"]]
        ds.num_classes = len(ds.label_to_idx)

    dl = DataLoader(
        ds,
        batch_size=1,  # 强制改为 1
        num_workers=0,  # 强制改为 0
        pin_memory=False,  # 强制改为 False
       #           batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        shuffle=False,
        #         num_workers=int(data_cfg.get("num_workers", 4)),
        #         pin_memory=True,
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

    gate_cfg = cfg.get("inference", {}).get("gating", {})
    gate_margin = float(gate_cfg.get("margin_min", 0.2))
    gate_p1 = float(gate_cfg.get("p1_min", 0.7))
    calibration_cfg = cfg.get("inference", {}).get("calibration", {})
    temperature = (
        float(calibration_cfg.get("temperature", 1.0))
        if calibration_cfg.get("enabled", False)
        else 1.0
    )

    rows: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []

    with torch.no_grad():
         for  batch in dl:

            b = batch_to_device(batch, device)
            out = model(b)
            logits = out["logits"].detach().cpu()

            topk_idx, topk_prob = probs_topk(
                logits,
                k=3,
                temperature=temperature,
            )
            margins, entropy = margin_and_entropy(logits / temperature)

            for i in range(logits.shape[0]):
                tier = str(batch["tier"][i]).upper()
                p1 = float(topk_prob[i, 0]) if topk_prob.shape[1] > 0 else 0.0
                p2 = float(topk_prob[i, 1]) if topk_prob.shape[1] > 1 else 0.0
                p3 = float(topk_prob[i, 2]) if topk_prob.shape[1] > 2 else 0.0

                l1 = idx_to_label[int(topk_idx[i, 0])] if topk_idx.shape[1] > 0 else ""
                l2 = idx_to_label[int(topk_idx[i, 1])] if topk_idx.shape[1] > 1 else ""
                l3 = idx_to_label[int(topk_idx[i, 2])] if topk_idx.shape[1] > 2 else ""

                margin = float(margins[i])
                ent = float(entropy[i])
                autofill = bool((tier in {"A", "B"}) and (margin >= gate_margin) and (p1 >= gate_p1))

                row = {
                    "sample_id": batch["sample_id"][i],
                    "gt": batch["gt"][i],
                    "top1_label": l1,
                    "top1_prob": f"{p1:.6f}",
                    "top2_label": l2,
                    "top2_prob": f"{p2:.6f}",
                    "top3_label": l3,
                    "top3_prob": f"{p3:.6f}",
                    "margin": f"{margin:.6f}",
                    "entropy": f"{ent:.6f}",
                    "tier": tier,
                    "autofill_top1": str(autofill).lower(),
                }
                rows.append(row)

                items.append(
                    {
                        "sample_id": batch["sample_id"][i],
                        "video_id": batch["video_id"][i],
                        "hand": batch["hand"][i],
                        "event_id": batch["event_id"][i],
                        "span": [int(batch["span_start"][i]), int(batch["span_end"][i])],
                        "gt": batch["gt"][i],
                        "tier": tier,
                        "topk": [
                            {"label": l1, "prob": p1},
                            {"label": l2, "prob": p2},
                            {"label": l3, "prob": p3},
                        ],
                        "margin": margin,
                        "entropy": ent,
                        "autofill_top1": autofill,
                    }
                )

    ckpt_dir = Path(args.ckpt).resolve().parent
    out_csv = Path(args.out_csv).resolve() if args.out_csv else ckpt_dir / f"{args.split}_predictions.csv"
    out_pred = Path(args.out_predjson).resolve() if args.out_predjson else ckpt_dir / "pred.json"

    write_predictions_csv(out_csv, rows)

    pred_obj = {
        "schema": "handshape_pred_v1",
        "model": {
            "exp_name": ckpt_dir.name,
            "ckpt": str(Path(args.ckpt).resolve()),
            "config": str(Path(args.config).resolve()),
            "split": args.split,
            "temperature": temperature,
        },
        "items": items,
    }
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    with out_pred.open("w", encoding="utf-8") as f:
        json.dump(pred_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps({"split": args.split, "out_csv": str(out_csv), "out_predjson": str(out_pred), "num_items": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
