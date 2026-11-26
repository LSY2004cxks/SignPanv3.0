#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create signer-disjoint 18/5/5 split.")
    ap.add_argument("--data-root", default="data28", help="Root containing signer folders like S8/S10/...")
    ap.add_argument("--signers", default=None, help="Comma-separated signer ids. If omitted, auto-scan from data-root.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--output", default="splits/signer_split.json")
    return ap.parse_args()


def scan_signers(data_root: str) -> List[str]:
    root = Path(data_root)
    if not root.exists():
        return []
    signers = []
    for p in sorted(root.iterdir()):
        # only keep canonical signer dirs like S8/S124; skip S8-label, etc.
        if p.is_dir() and re.fullmatch(r"S\d+", p.name):
            signers.append(p.name)
    return signers


def main() -> None:
    args = parse_args()

    if args.signers:
        signers = [s.strip() for s in args.signers.split(",") if s.strip()]
    else:
        signers = scan_signers(args.data_root)

    need = args.train + args.val + args.test
    if len(signers) < need:
        raise SystemExit(f"Need at least {need} signers, got {len(signers)}")

    random.seed(args.seed)
    signers = sorted(signers)
    random.shuffle(signers)

    train = signers[: args.train]
    val = signers[args.train : args.train + args.val]
    test = signers[args.train + args.val : args.train + args.val + args.test]
    unused = signers[args.train + args.val + args.test :]

    out = {
        "seed": args.seed,
        "counts": {
            "total_input": len(signers),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "unused": len(unused),
        },
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "unused": sorted(unused),
    }

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote split -> {op}")
    print(out["counts"])


if __name__ == "__main__":
    main()
