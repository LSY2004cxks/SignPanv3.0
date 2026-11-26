#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.indexer import build_manifest
from src.data.split import load_signer_split
from src.features.builder import FeatureBuilder
from src.infer.exporter import build_prefill_json
from src.infer.postprocess import decode_boundaries
from src.models.tcn import BoundaryTCN
from src.utils.io import (
    ensure_dir,
    load_json,
    load_yaml,
    make_run_dir,
    parse_signer_list,
    save_json,
    select_device,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inference + Prefill-B export")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", help="train|val|test|all|dryrun")
    ap.add_argument("--signers", default=None, help="comma-separated signer override")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--label-root", default=None)
    ap.add_argument("--pose-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def _resolve_signers(split_obj: Dict[str, Any], split_name: str, override: List[str] | None) -> List[str]:
    if override:
        return override
    if split_name == "all":
        out = []
        for k in ("train", "val", "test"):
            out.extend(split_obj.get(k, []))
        return sorted(set(out))
    if split_name == "dryrun":
        out = []
        for k in ("train", "val", "test", "unused"):
            out.extend(split_obj.get(k, []))
        return sorted(set(out))
    if split_name not in split_obj:
        raise KeyError(f"split {split_name} not found in split file")
    return [str(x) for x in split_obj[split_name]]


def _load_best_tau(checkpoint_path: str) -> float | None:
    run_dir = Path(checkpoint_path).parent
    p = run_dir / "best_tau.json"
    if p.exists():
        try:
            d = load_json(str(p))
            return float(d.get("best_tau"))
        except Exception:
            return None
    return None


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
    signers = _resolve_signers(split_obj, args.split, parse_signer_list(args.signers))

    manifest = build_manifest(
        video_root=cfg["data"]["video_root"],
        label_root=cfg["data"]["label_root"],
        pose_root=cfg["data"]["pose_root"],
        signers=signers,
        allow_pose_missing=bool(cfg["data"].get("allow_pose_missing", False)),
    )
    records = manifest.records
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records for inference.")

    device = select_device(cfg["runtime"].get("device", "auto"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    input_dim = int(ckpt["input_dim"])
    model_cfg = ckpt.get("model_cfg", cfg["model"])

    model = BoundaryTCN(**_model_kwargs(model_cfg, input_dim)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    tau = args.tau
    if tau is None:
        tau = _load_best_tau(args.checkpoint)
    if tau is None:
        tau = float(cfg["infer"]["tau"])

    prefill_root = Path(cfg["paths"]["prefill_dir"]) / args.split
    probs_root = Path(cfg["paths"]["probs_dir"]) / args.split
    ensure_dir(str(prefill_root))
    ensure_dir(str(probs_root))

    feature_builder = FeatureBuilder(**_feature_builder_kwargs(cfg))

    used_names = set()

    for rec in records:
        if rec.pose_path is None:
            continue

        feat_data = feature_builder.build_from_pose_path(rec.pose_path)

        hand_outputs: Dict[str, Dict[str, Any]] = {}
        for hand in ("L", "R"):
            feat = feat_data[hand]["features"]
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)[0].cpu().numpy()  # (T,3)
            p = 1.0 / (1.0 + np.exp(-logits))
            p_start, p_end, p_inside = p[:, 0], p[:, 1], p[:, 2]

            decoded = decode_boundaries(
                p_start=p_start,
                p_end=p_end,
                p_inside=p_inside,
                tau=float(tau),
                peak_radius=int(cfg["infer"]["peak_radius"]),
                nms_window=int(cfg["infer"]["w_nms"]),
                Lmin=int(cfg["infer"]["Lmin"]),
                Lmax=int(cfg["infer"]["Lmax"]),
                inside_recommend_thr=float(cfg["infer"]["inside_recommend_thr"]),
                pair_mode=str(cfg["infer"].get("pair_mode", "score_then_nearest")),
            )

            hand_outputs[hand] = decoded

            np.savez_compressed(
                probs_root / f"{rec.video_id}_{hand}.npz",
                p_start=p_start.astype(np.float32),
                p_end=p_end.astype(np.float32),
                p_inside=p_inside.astype(np.float32),
            )

        export_obj = build_prefill_json(
            video_id=rec.video_id,
            signer_id=rec.signer_id,
            clip_id=rec.clip_id,
            schema_type=rec.schema_type,
            left_decoded=hand_outputs["L"],
            right_decoded=hand_outputs["R"],
            tau=float(tau),
            peak_radius=int(cfg["infer"]["peak_radius"]),
            w_nms=int(cfg["infer"]["w_nms"]),
            Lmin=int(cfg["infer"]["Lmin"]),
            Lmax=int(cfg["infer"]["Lmax"]),
        )

        name = rec.video_id
        if name in used_names:
            name = f"{rec.video_id}_{rec.clip_id}"
        used_names.add(name)

        save_json(str(prefill_root / f"{name}.json"), export_obj)

    infer_log_dir = make_run_dir(cfg["paths"]["logs_dir"])
    save_yaml(str(Path(infer_log_dir) / "config.yaml"), cfg)
    save_json(
        str(Path(infer_log_dir) / "schema_report.json"),
        {
            "split": args.split,
            "signers": signers,
            "tau": float(tau),
            "schema": manifest.schema_report,
            "feature_mapping": feature_builder.get_report(),
            "device": str(device),
            "checkpoint": args.checkpoint,
            "n_records": len(records),
        },
    )

    print(f"Exported Prefill-B jsons -> {prefill_root}")
    print(f"Prob curves -> {probs_root}")
    print(f"Infer log -> {infer_log_dir}")


if __name__ == "__main__":
    main()
