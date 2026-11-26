#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def ensure_manifest(project_root: Path, data_root: Path, manifest_path: Path) -> None:
    if manifest_path.exists():
        return
    run_cmd(
        [
            "python",
            "src/data/build_manifest.py",
            "--data_root",
            str(data_root),
            "--out_manifest",
            str(manifest_path),
            "--out_orphans",
            str(project_root / "manifest" / "orphans.jsonl"),
            "--out_stats",
            str(project_root / "manifest" / "manifest_stats.json"),
        ],
        cwd=project_root,
    )


def ensure_splits(project_root: Path, data_root: Path, split_dir: Path, manifest_path: Path) -> None:
    required = [split_dir / "train_signers.txt", split_dir / "val_signers.txt", split_dir / "test_signers.txt"]
    if all(p.exists() for p in required):
        return
    run_cmd(
        [
            "python",
            "src/data/split_signers.py",
            "--data_root",
            str(data_root),
            "--out_dir",
            str(split_dir),
            "--seed",
            "42",
            "--train_n",
            "18",
            "--val_n",
            "5",
            "--test_n",
            "5",
            "--manifest",
            str(manifest_path),
        ],
        cwd=project_root,
    )


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def tier_from_side(side: float) -> str:
    if side >= 96:
        return "A"
    if side >= 64:
        return "B"
    if side >= 48:
        return "C"
    return "D"


def load_meta(meta_dir: Path, sample_id: str) -> Dict[str, Any]:
    p = meta_dir / f"{sample_id}.json"
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def summarize_split(name: str, signers: List[str], rows: List[Dict[str, Any]], meta_dir: Path) -> Dict[str, Any]:
    rows_s = [r for r in rows if str(r.get("signer_id")) in set(signers)]

    labels: Dict[str, int] = {}
    tiers: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    roi_too_small = 0
    missing_meta = 0

    for r in rows_s:
        lb = str(r.get("gt_label"))
        labels[lb] = labels.get(lb, 0) + 1

        m = load_meta(meta_dir, r["sample_id"])
        if not m:
            missing_meta += 1
        tier = str(m.get("tier", "")).upper()
        if tier not in tiers:
            side = float(m.get("bbox_side_px_median", 0.0) or 0.0)
            tier = tier_from_side(side)
        tiers[tier] += 1

        roi_small = m.get("ROI_TOO_SMALL")
        if isinstance(roi_small, bool):
            roi_too_small += int(roi_small)
        else:
            side = float(m.get("bbox_side_px_median", 0.0) or 0.0)
            roi_too_small += int(side < 48.0)

    n = len(rows_s)
    return {
        "split": name,
        "signers": signers,
        "num_samples": n,
        "num_labels": len(labels),
        "label_counts": dict(sorted(labels.items(), key=lambda kv: kv[0])),
        "tiers": tiers,
        "ROI_TOO_SMALL_ratio": (roi_too_small / n) if n > 0 else 0.0,
        "missing_meta_samples": missing_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data-28")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--manifest", type=str, default="manifest/manifest.jsonl")
    parser.add_argument("--split_dir", type=str, default="splits")
    parser.add_argument("--meta_dir", type=str, default="cache/meta")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    split_dir = Path(args.split_dir).resolve()
    meta_dir = Path(args.meta_dir).resolve()

    ensure_manifest(project_root, data_root, manifest_path)
    ensure_splits(project_root, data_root, split_dir, manifest_path)

    rows = read_manifest(manifest_path)
    train_signers = read_lines(split_dir / "train_signers.txt")
    val_signers = read_lines(split_dir / "val_signers.txt")
    test_signers = read_lines(split_dir / "test_signers.txt")

    train_sum = summarize_split("train", train_signers, rows, meta_dir)
    val_sum = summarize_split("val", val_signers, rows, meta_dir)
    test_sum = summarize_split("test", test_signers, rows, meta_dir)

    train_labels = set(train_sum["label_counts"].keys())
    val_unseen = sorted(set(val_sum["label_counts"].keys()) - train_labels)
    test_unseen = sorted(set(test_sum["label_counts"].keys()) - train_labels)

    out = {
        "manifest": str(manifest_path),
        "split_dir": str(split_dir),
        "meta_dir": str(meta_dir),
        "train": train_sum,
        "val": val_sum,
        "test": test_sum,
        "unseen_labels": {
            "val_not_in_train": val_unseen,
            "test_not_in_train": test_unseen,
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if val_unseen or test_unseen:
        print("[ERROR] unseen labels found in val/test not present in train", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
