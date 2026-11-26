#!/usr/bin/env python3
"""
Alias script for Prefill-B export.
Internally calls scripts/infer.py to generate strict JSON under outputs/prefill_B/<split>/.
"""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export Prefill-B JSON (wrapper of infer.py)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--signers", default=None)
    ap.add_argument("--tau", default=None)
    ap.add_argument("--limit", default=None)
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--label-root", default=None)
    ap.add_argument("--pose-root", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
    ]
    for k in [
        "signers",
        "tau",
        "limit",
        "video_root",
        "label_root",
        "pose_root",
        "split_file",
        "device",
    ]:
        v = getattr(args, k)
        if v is not None:
            cmd.extend(["--" + k.replace("_", "-"), str(v)])
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
