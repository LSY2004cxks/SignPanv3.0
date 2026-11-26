#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["sample_id"]] = row
    return out


def to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="manifest/manifest.jsonl")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="handshape_data28")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--config", type=str, default="configs/handshape_data28.yaml")
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest).resolve())

    items: List[Dict[str, Any]] = []
    with Path(args.pred_csv).resolve().open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id", "")
            m = manifest.get(sid, {})

            topk = [
                {"label": row.get("top1_label", ""), "prob": to_float(row.get("top1_prob"))},
                {"label": row.get("top2_label", ""), "prob": to_float(row.get("top2_prob"))},
                {"label": row.get("top3_label", ""), "prob": to_float(row.get("top3_prob"))},
            ]

            items.append(
                {
                    "sample_id": sid,
                    "video_id": m.get("video_id", ""),
                    "hand": m.get("hand", ""),
                    "event_id": m.get("event_id", ""),
                    "span": [int(m.get("span_start", 0)), int(m.get("span_end", 0))],
                    "gt": row.get("gt", ""),
                    "tier": row.get("tier", ""),
                    "topk": topk,
                    "margin": to_float(row.get("margin")),
                    "entropy": to_float(row.get("entropy")),
                    "autofill_top1": str(row.get("autofill_top1", "false")).strip().lower() == "true",
                }
            )

    obj = {
        "schema": "handshape_pred_v1",
        "model": {
            "exp_name": args.exp_name,
            "ckpt": args.ckpt,
            "config": str(Path(args.config).resolve()),
        },
        "items": items,
    }

    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out_json": str(out), "num_items": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
