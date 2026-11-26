#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    HandshapeEventDataset,
    build_class_balanced_sampler,
    collate_fn,
    compute_class_counts,
)
from losses.class_balanced_ce import ClassBalancedCrossEntropy, build_class_balanced_ce
from losses.infonce import symmetric_infonce
from metrics import aggregate_metrics, margin_and_entropy, tierwise_metrics
from models import HandshapeNet
from runtime_utils import (
    append_train_log,
    load_config,
    make_run_dir,
    probs_topk,
    save_json,
    save_yaml,
    set_seed,
    try_save_git_hash,
    try_save_requirements_lock,
)


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
    if bool(cfg.get("data", {}).get("on_the_fly_fallback", True)):
        raise ValueError(
            "model.enforce_roi_skel_only=true requires data.on_the_fly_fallback=false "
            "(use precomputed ROI/skeleton cache only)."
        )


def write_predictions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def build_dataloaders(cfg: Dict[str, Any], batch_size: int, mode: str):
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]

    common = dict(
        manifest_path=data_cfg["manifest_path"],
        split_dir=data_cfg["split_dir"],
        mode=mode,
        k=int(data_cfg["k"]),
        cache_roi_dir=data_cfg["cache_roi_dir"],
        cache_skeleton_dir=data_cfg["cache_skeleton_dir"],
        cache_meta_dir=data_cfg["cache_meta_dir"],
        tier_weights=data_cfg["tier_weights"],
        on_the_fly_fallback=bool(data_cfg.get("on_the_fly_fallback", True)),
    )

    train_ds = HandshapeEventDataset(split="train", **common)
    val_ds = HandshapeEventDataset(split="val", **common)

    imbalance = tr_cfg.get("class_imbalance", {}).get("strategy", "class_balanced_sampler")

    sampler = None
    shuffle = True
    if imbalance == "class_balanced_sampler":
        sampler = build_class_balanced_sampler(train_ds)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_ds, val_ds, train_loader, val_loader


def build_criterion(cfg: Dict[str, Any], train_ds: HandshapeEventDataset) -> ClassBalancedCrossEntropy:
    tr_cfg = cfg["training"]
    imb_cfg = tr_cfg.get("class_imbalance", {})
    strategy = imb_cfg.get("strategy", "class_balanced_sampler")
    beta = float(imb_cfg.get("beta", 0.9999))

    if strategy == "class_balanced_ce_weight":
        counts = compute_class_counts(train_ds)
        return build_class_balanced_ce(counts=counts, beta=beta)
    return ClassBalancedCrossEntropy(class_weights=None)


def make_model(cfg: Dict[str, Any], num_classes: int) -> HandshapeNet:
    mcfg = cfg["model"]
    model = HandshapeNet(
        mode=mcfg["mode"],
        num_classes=num_classes,
        roi_backbone=mcfg.get("roi_backbone", "convnext_tiny"),
        roi_dim=int(mcfg.get("roi_dim", 512)),
        skel_dim=int(mcfg.get("skel_dim", 256)),
        fusion_hidden=int(mcfg.get("fusion_hidden", 512)),
        dropout=float(mcfg.get("dropout", 0.1)),
        pretrained=bool(mcfg.get("pretrained", True)),
        freeze_backbone=bool(mcfg.get("freeze_backbone", True)),
    )
    return model


def make_optimizer(cfg: Dict[str, Any], model: HandshapeNet) -> torch.optim.Optimizer:
    tr_cfg = cfg["training"]
    lr_head = float(tr_cfg.get("lr_head", 3e-4))
    lr_backbone = float(tr_cfg.get("lr_backbone", 1e-5))
    wd = float(tr_cfg.get("weight_decay", 1e-4))

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "roi_encoder.backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(groups, weight_decay=wd)


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["label_idx"] = out["label_idx"].to(device, non_blocking=True)
    out["tier_weight"] = out["tier_weight"].to(device, non_blocking=True)
    if "roi" in out:
        out["roi"] = out["roi"].to(device, non_blocking=True)
    if "skeleton" in out:
        out["skeleton"] = out["skeleton"].to(device, non_blocking=True)
    return out


@torch.no_grad()
def evaluate(
    model: HandshapeNet,
    loader: DataLoader,
    criterion: ClassBalancedCrossEntropy,
    idx_to_label: Dict[int, str],
    cfg: Dict[str, Any],
    device: torch.device,
    epoch: int | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_tiers: List[str] = []
    all_sample_ids: List[str] = []
    all_gt_names: List[str] = []

    total_loss = 0.0
    total_n = 0

    gate_cfg = cfg.get("inference", {}).get("gating", {})
    gate_margin = float(gate_cfg.get("margin_min", 0.2))
    gate_p1 = float(gate_cfg.get("p1_min", 0.7))
    lcfg = cfg.get("loss", {})
    nce_start_epoch = int(lcfg.get("nce_start_epoch", 1))
    use_nce = (epoch is None) or (epoch >= nce_start_epoch)

    for batch in loader:
        b = batch_to_device(batch, device)
        out = model(b)
        logits = out["logits"]
        targets = b["label_idx"]

        loss = criterion(logits, targets, sample_weight=b["tier_weight"])
        if model.mode == "roi_skel_deform_nce" and use_nce and out.get("nce_roi") is not None:
            tau = float(lcfg.get("tau", 0.07))
            lam = float(lcfg.get("lambda_nce", 0.1))
            loss = loss + lam * symmetric_infonce(out["nce_roi"], out["nce_skel"], tau=tau)

        n = targets.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_tiers.extend([str(t) for t in b["tier"]])
        all_sample_ids.extend(batch["sample_id"])
        all_gt_names.extend(batch["gt"])

    if total_n == 0:
        return {"loss": 0.0}, []

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    base = aggregate_metrics(logits_all, targets_all)
    base["loss"] = total_loss / max(total_n, 1)
    base["tierwise"] = tierwise_metrics(logits_all, targets_all, all_tiers)

    topk_idx, topk_prob = probs_topk(logits_all, k=3)
    margins, entropy = margin_and_entropy(logits_all)

    rows: List[Dict[str, Any]] = []
    for i in range(logits_all.shape[0]):
        p1 = float(topk_prob[i, 0]) if topk_prob.shape[1] > 0 else 0.0
        p2 = float(topk_prob[i, 1]) if topk_prob.shape[1] > 1 else 0.0
        p3 = float(topk_prob[i, 2]) if topk_prob.shape[1] > 2 else 0.0

        l1 = idx_to_label[int(topk_idx[i, 0])] if topk_idx.shape[1] > 0 else ""
        l2 = idx_to_label[int(topk_idx[i, 1])] if topk_idx.shape[1] > 1 else ""
        l3 = idx_to_label[int(topk_idx[i, 2])] if topk_idx.shape[1] > 2 else ""

        tier = str(all_tiers[i]).upper()
        m = float(margins[i])
        e = float(entropy[i])
        autofill = bool((tier in {"A", "B"}) and (m >= gate_margin) and (p1 >= gate_p1))

        rows.append(
            {
                "sample_id": all_sample_ids[i],
                "gt": all_gt_names[i],
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

    return base, rows


def train_impl(cfg: Dict[str, Any], run_dir: Path, batch_size: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = cfg["model"]["mode"]
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg, batch_size=batch_size, mode=mode)

    model = make_model(cfg, num_classes=train_ds.num_classes).to(device)
    criterion = build_criterion(cfg, train_ds).to(device)
    optimizer = make_optimizer(cfg, model)

    tr_cfg = cfg["training"]
    epochs = int(tr_cfg.get("epochs", 50))
    stage1_epochs = int(tr_cfg.get("stage1_epochs", 8))
    lcfg = cfg.get("loss", {})
    nce_start_epoch = int(lcfg.get("nce_start_epoch", 1))
    early_metric_raw = str(tr_cfg.get("early_stop_metric", "hit@3"))
    em = early_metric_raw.lower().replace(" ", "")
    if em in {"hit3", "hit@3", "val_hit3"}:
        early_metric = "hit@3"
    elif em in {"macro-f1", "macro_f1", "val_macro_f1"}:
        early_metric = "macro_f1"
    else:
        early_metric = early_metric_raw
    patience = int(tr_cfg.get("patience", 8))
    min_delta = float(tr_cfg.get("min_delta", 1e-4))

    # stage control
    if mode in {"roi_only", "roi_skel_fusion", "roi_skel_deform_nce"} and stage1_epochs > 0:
        model.freeze_roi_backbone()
        append_train_log(run_dir, f"[stage] stage1 freeze backbone for {stage1_epochs} epochs")

    history: List[Dict[str, Any]] = []
    best_score = -1e18
    best_epoch = 0
    best_rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if (
            mode in {"roi_only", "roi_skel_fusion", "roi_skel_deform_nce"}
            and stage1_epochs > 0
            and epoch == stage1_epochs + 1
        ):
            model.unfreeze_roi_last_stage()
            append_train_log(run_dir, f"[stage] stage2 unfreeze roi last stage at epoch={epoch}")

        model.train()
        run_loss = 0.0
        run_n = 0

        for batch in train_loader:
            b = batch_to_device(batch, device)
            out = model(b)
            logits = out["logits"]
            targets = b["label_idx"]

            loss = criterion(logits, targets, sample_weight=b["tier_weight"])
            if mode == "roi_skel_deform_nce" and epoch >= nce_start_epoch and out.get("nce_roi") is not None:
                tau = float(lcfg.get("tau", 0.07))
                lam = float(lcfg.get("lambda_nce", 0.1))
                loss = loss + lam * symmetric_infonce(out["nce_roi"], out["nce_skel"], tau=tau)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n = targets.shape[0]
            run_loss += float(loss.item()) * n
            run_n += n

        train_loss = run_loss / max(run_n, 1)

        val_metrics, val_rows = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            idx_to_label=train_ds.idx_to_label,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        score = float(val_metrics.get(early_metric, 0.0))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "early_metric": early_metric,
            "early_score": score,
        }
        history.append(row)

        log_line = (
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_top1={val_metrics.get('top1_acc', 0.0):.4f} "
            f"val_hit3={val_metrics.get('hit@3', 0.0):.4f} "
            f"val_f1={val_metrics.get('macro_f1', 0.0):.4f} "
            f"val_margin={val_metrics.get('mean_margin', 0.0):.4f} "
            f"val_entropy={val_metrics.get('mean_entropy', 0.0):.4f}"
        )
        print(log_line)
        append_train_log(run_dir, log_line)

        improved = score > (best_score + min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            best_rows = val_rows
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "label_to_idx": train_ds.label_to_idx,
                "idx_to_label": train_ds.idx_to_label,
                "best_score": best_score,
                "best_metric": early_metric,
            }
            torch.save(ckpt, run_dir / "best.ckpt")

        if epoch - best_epoch >= patience:
            append_train_log(run_dir, f"[early_stop] epoch={epoch} best_epoch={best_epoch} score={best_score:.6f}")
            break

    # save final artifacts
    metrics_out = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric": early_metric,
        "history": history,
        "batch_size_used": batch_size,
    }
    save_json(run_dir / "metrics.json", metrics_out)
    write_predictions_csv(run_dir / "val_predictions.csv", best_rows)

    return metrics_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    enforce_roi_skel_only(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    exp_cfg = cfg.get("experiment", {})
    exp_name = str(exp_cfg.get("exp_name", "handshape_data28"))
    runs_root = str(exp_cfg.get("runs_root", "runs"))

    run_dir = make_run_dir(runs_root=runs_root, exp_name=exp_name)
    cfg = dict(cfg)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["run_dir"] = str(run_dir)

    save_yaml(run_dir / "config.yaml", cfg)
    try_save_requirements_lock(run_dir)
    try_save_git_hash(run_dir)

    tr_cfg = cfg.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 64))
    min_batch_size = int(tr_cfg.get("min_batch_size", 4))
    if min_batch_size < 1:
        min_batch_size = 1
    if batch_size < min_batch_size:
        min_batch_size = batch_size

    bs_try = batch_size
    while True:
        try:
            metrics_out = train_impl(cfg=cfg, run_dir=run_dir, batch_size=bs_try)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise
            if bs_try <= min_batch_size:
                raise
            next_bs = max(min_batch_size, bs_try // 2)
            if next_bs == bs_try:
                raise
            append_train_log(run_dir, f"[oom] batch_size={bs_try} fallback_to={next_bs}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs_try = next_bs

    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
