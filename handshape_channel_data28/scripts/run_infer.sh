#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ckpt_path> [split] [config]"
  exit 1
fi

CKPT_PATH="$1"
SPLIT="${2:-test}"
CONFIG_PATH="${3:-configs/handshape_data28.yaml}"

RUN_DIR="$(cd "$(dirname "$CKPT_PATH")" && pwd)"

python src/infer.py --ckpt "$CKPT_PATH" --split "$SPLIT" --config "$CONFIG_PATH" --out_csv "$RUN_DIR/${SPLIT}_predictions.csv" --out_predjson "$RUN_DIR/pred.json"
python tools/export_predjson.py --pred_csv "$RUN_DIR/${SPLIT}_predictions.csv" --manifest manifest/manifest.jsonl --out_json "$RUN_DIR/pred.json" --ckpt "$CKPT_PATH" --config "$CONFIG_PATH" --exp_name "$(basename "$RUN_DIR")"
