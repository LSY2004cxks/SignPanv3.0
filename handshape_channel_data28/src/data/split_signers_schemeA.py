#!/usr/bin/env python3
"""Scheme A split builder:

- signer-disjoint fixed test set
- Group K-Fold on remaining development signers
- anchor signers are forced to stay in train in every fold
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def signer_sort_key(s: str):
    m = re.match(r"^S(\d+)$", s.upper())
    if m:
        return int(m.group(1))
    return 10**9


def normalize_signer_name(name: str) -> str:
    x = str(name).strip().upper()
    if not x.startswith("S"):
        x = f"S{x}"
    return x


def discover_signers(data_root: Path) -> List[str]:
    signers: List[str] = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.endswith("-label"):
            continue
        if not name.upper().startswith("S"):
            continue
        signers.append(normalize_signer_name(name))
    return sorted(set(signers), key=signer_sort_key)


def read_manifest(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(str(v) + "\n")


def split_sizes(n_items: int, n_folds: int) -> List[int]:
    base = n_items // n_folds
    rem = n_items % n_folds
    return [base + (1 if i < rem else 0) for i in range(n_folds)]


def parse_signer_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [normalize_signer_name(x) for x in raw.split(",") if x.strip()]


def summarize_split(
    rows: Sequence[Dict[str, object]],
    signers: Sequence[str],
    all_labels: Sequence[str],
) -> Dict[str, object]:
    signer_set = set(signers)
    c = Counter()
    for r in rows:
        if str(r.get("signer_id")) in signer_set:
            c[str(r.get("gt_label"))] += 1
    missing = sorted([lb for lb in all_labels if c.get(lb, 0) <= 0])
    return {
        "num_signers": len(signers),
        "num_samples": int(sum(c.values())),
        "num_labels": len([lb for lb in all_labels if c.get(lb, 0) > 0]),
        "missing_labels": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data28")
    parser.add_argument("--manifest", type=str, default="manifest/manifest.jsonl")
    parser.add_argument("--out_dir", type=str, default="splits_schemeA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_n", type=int, default=5)
    parser.add_argument("--anchor_train_signers", type=str, default="S109")
    parser.add_argument("--fixed_test_signers", type=str, default="")
    parser.add_argument("--must_test_signers", type=str, default="")
    parser.add_argument("--must_val_signers", type=str, default="")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_signers = discover_signers(data_root)
    rows = read_manifest(manifest_path)
    all_labels = sorted({str(r.get("gt_label")) for r in rows})

    manifest_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        sid = normalize_signer_name(str(r.get("signer_id", "")))
        manifest_counts[sid] += 1

    anchor_train = sorted(set(parse_signer_list(args.anchor_train_signers)), key=signer_sort_key)
    for s in anchor_train:
        if s not in all_signers:
            raise RuntimeError(f"Anchor signer not found in signer dirs: {s}")

    must_test = sorted(set(parse_signer_list(args.must_test_signers)), key=signer_sort_key)
    must_val = sorted(set(parse_signer_list(args.must_val_signers)), key=signer_sort_key)
    for s in must_test + must_val:
        if s not in all_signers:
            raise RuntimeError(f"Constrained signer not found in signer dirs: {s}")
    if any(s in anchor_train for s in must_test):
        raise RuntimeError("Anchor train signer cannot be in must_test_signers.")
    if any(s in must_test for s in must_val):
        raise RuntimeError("A signer cannot be both in must_test_signers and must_val_signers.")
    if any(s in anchor_train for s in must_val):
        raise RuntimeError("Anchor train signer cannot be in must_val_signers.")

    fixed_test = parse_signer_list(args.fixed_test_signers)
    if fixed_test:
        fixed_test = sorted(set(fixed_test), key=signer_sort_key)
        for s in fixed_test:
            if s not in all_signers:
                raise RuntimeError(f"Fixed test signer not found in signer dirs: {s}")
        if any(s in anchor_train for s in fixed_test):
            raise RuntimeError("Anchor train signer cannot be in fixed test set.")
        if len(fixed_test) != int(args.test_n):
            raise RuntimeError(f"fixed_test_signers size must equal test_n={args.test_n}, got {len(fixed_test)}")
        if any(s not in fixed_test for s in must_test):
            raise RuntimeError("fixed_test_signers must contain all must_test_signers.")
        if any(s in fixed_test for s in must_val):
            raise RuntimeError("fixed_test_signers cannot include must_val_signers.")
    else:
        if len(must_test) > int(args.test_n):
            raise RuntimeError(
                f"must_test_signers size cannot exceed test_n={args.test_n}, got {len(must_test)}"
            )
        rng = random.Random(args.seed)
        blocked = set(anchor_train) | set(must_val) | set(must_test)
        pool = [s for s in all_signers if s not in blocked]
        if len(pool) < int(args.test_n):
            raise RuntimeError("Not enough signers to create fixed test set.")
        rng.shuffle(pool)
        n_extra = int(args.test_n) - len(must_test)
        fixed_test = sorted(must_test + pool[:n_extra], key=signer_sort_key)

    dev_signers = sorted([s for s in all_signers if s not in set(fixed_test)], key=signer_sort_key)
    if any(s not in dev_signers for s in anchor_train):
        raise RuntimeError("Anchor signer must remain in dev set.")
    if any(s not in dev_signers for s in must_val):
        raise RuntimeError("must_val_signers must remain in dev set (not in test).")

    val_pool = [s for s in dev_signers if s not in set(anchor_train)]
    n_folds = int(args.num_folds)
    if len(val_pool) < n_folds:
        raise RuntimeError(
            f"Not enough val-pool signers for folds: pool={len(val_pool)}, num_folds={n_folds}"
        )

    # Greedy balancing by sample counts.
    # Signers with more samples are assigned first to the fold with least current samples.
    targets = split_sizes(len(val_pool), n_folds)
    fold_signers: List[List[str]] = [[] for _ in range(n_folds)]
    fold_samples = [0 for _ in range(n_folds)]

    # Enforce must_val_signers in fold_1 validation set.
    val_pool_set = set(val_pool)
    if any(s not in val_pool_set for s in must_val):
        raise RuntimeError("must_val_signers must be selected from val pool.")
    fold_signers[0] = sorted(must_val, key=signer_sort_key)
    fold_samples[0] = sum(int(manifest_counts.get(s, 0)) for s in fold_signers[0])

    capacities = targets[:]
    capacities[0] = max(capacities[0], len(fold_signers[0]))

    val_pool_sorted = sorted(
        [s for s in val_pool if s not in set(must_val)],
        key=lambda s: (-manifest_counts.get(s, 0), signer_sort_key(s)),
    )
    for s in val_pool_sorted:
        candidates = [i for i in range(n_folds) if len(fold_signers[i]) < capacities[i]]
        best_i = min(candidates, key=lambda i: (fold_samples[i], len(fold_signers[i]), i))
        fold_signers[best_i].append(s)
        fold_samples[best_i] += int(manifest_counts.get(s, 0))

    # Write common files
    write_lines(out_dir / "fixed_test_signers.txt", fixed_test)
    write_lines(out_dir / "dev_signers.txt", dev_signers)

    folds_meta = []
    fixed_test_set = set(fixed_test)
    dev_set = set(dev_signers)
    for i in range(n_folds):
        fold_id = i + 1
        val_signers = sorted(fold_signers[i], key=signer_sort_key)
        val_set = set(val_signers)
        train_signers = sorted([s for s in dev_set if s not in val_set], key=signer_sort_key)
        test_signers = sorted(list(fixed_test_set), key=signer_sort_key)

        if any(s not in train_signers for s in anchor_train):
            raise RuntimeError(f"Fold {fold_id}: anchor signer missing in train split.")

        fold_dir = out_dir / f"fold_{fold_id}"
        write_lines(fold_dir / "train_signers.txt", train_signers)
        write_lines(fold_dir / "val_signers.txt", val_signers)
        write_lines(fold_dir / "test_signers.txt", test_signers)

        train_sum = summarize_split(rows, train_signers, all_labels)
        val_sum = summarize_split(rows, val_signers, all_labels)
        test_sum = summarize_split(rows, test_signers, all_labels)

        fold_meta = {
            "fold_id": fold_id,
            "train_signers": train_signers,
            "val_signers": val_signers,
            "test_signers": test_signers,
            "anchor_train_signers": anchor_train,
            "anchor_all_in_train": all(s in train_signers for s in anchor_train),
            "summary": {
                "train": train_sum,
                "val": val_sum,
                "test": test_sum,
            },
        }
        with (fold_dir / "split_meta.json").open("w", encoding="utf-8") as f:
            json.dump(fold_meta, f, ensure_ascii=False, indent=2)
        folds_meta.append(fold_meta)

    global_meta = {
        "scheme": "A_fixed_test_group_kfold",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "seed": int(args.seed),
        "num_folds": n_folds,
        "test_n": int(args.test_n),
        "all_signers_sorted": all_signers,
        "fixed_test_signers": fixed_test,
        "anchor_train_signers": anchor_train,
        "must_test_signers": must_test,
        "must_val_signers": must_val,
        "dev_signers": dev_signers,
        "manifest_counts_by_signer": {s: int(manifest_counts.get(s, 0)) for s in all_signers},
        "folds": folds_meta,
    }
    with (out_dir / "schemeA_meta.json").open("w", encoding="utf-8") as f:
        json.dump(global_meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(global_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
