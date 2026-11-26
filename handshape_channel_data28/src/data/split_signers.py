#!/usr/bin/env python3
"""Create signer-disjoint splits from top-level signer folders.

Default policy:
- collect top-level directories under data_root that match S* and not *-label
- deterministic shuffle with seed 42
- split into 18/5/5 (train/val/test)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List


def signer_sort_key(s: str):
    m = re.match(r"^S(\d+)$", s.upper())
    if m:
        return int(m.group(1))
    return 10**9


def normalize_signer_name(name: str) -> str:
    x = name.strip().upper()
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
    signers = sorted(set(signers), key=signer_sort_key)
    return signers


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(str(v) + "\n")


def count_manifest_samples(manifest_path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not manifest_path.exists():
        return out

    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = normalize_signer_name(str(row.get("signer_id", "")))
            out[sid] = out.get(sid, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data-28")
    parser.add_argument("--out_dir", type=str, default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_n", type=int, default=18)
    parser.add_argument("--val_n", type=int, default=5)
    parser.add_argument("--test_n", type=int, default=5)
    parser.add_argument("--manifest", type=str, default="manifest/manifest.jsonl")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    signers = discover_signers(data_root)
    total_needed = args.train_n + args.val_n + args.test_n
    if len(signers) < total_needed:
        raise RuntimeError(
            f"Not enough signer folders: found={len(signers)}, required={total_needed}."
        )

    shuffled = signers.copy()
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)

    train_signers = shuffled[: args.train_n]
    val_signers = shuffled[args.train_n : args.train_n + args.val_n]
    test_signers = shuffled[args.train_n + args.val_n : args.train_n + args.val_n + args.test_n]

    write_lines(out_dir / "train_signers.txt", train_signers)
    write_lines(out_dir / "val_signers.txt", val_signers)
    write_lines(out_dir / "test_signers.txt", test_signers)

    # Optional counts from manifest (for in-progress annotation state)
    manifest_counts = count_manifest_samples(Path(args.manifest).resolve())

    meta = {
        "data_root": str(data_root),
        "seed": args.seed,
        "all_signers_sorted": signers,
        "all_signers_shuffled": shuffled,
        "train_signers": train_signers,
        "val_signers": val_signers,
        "test_signers": test_signers,
        "split_sizes": {
            "train": len(train_signers),
            "val": len(val_signers),
            "test": len(test_signers),
        },
        "manifest_sample_counts": {
            "train": sum(manifest_counts.get(s, 0) for s in train_signers),
            "val": sum(manifest_counts.get(s, 0) for s in val_signers),
            "test": sum(manifest_counts.get(s, 0) for s in test_signers),
        },
        "manifest_counts_by_signer": {s: manifest_counts.get(s, 0) for s in signers},
    }

    with (out_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
