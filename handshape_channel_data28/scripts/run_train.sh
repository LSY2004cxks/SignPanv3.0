#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/handshape_data28.yaml}"

read -r DATA_ROOT MANIFEST_PATH SPLIT_DIR <<EOF
$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8'))
print(cfg['data']['data_root'], cfg['data']['manifest_path'], cfg['data']['split_dir'])
PY
)
EOF

python src/data/build_manifest.py \
  --data_root "$DATA_ROOT" \
  --out_manifest "$MANIFEST_PATH" \
  --out_orphans manifest/orphans.jsonl \
  --out_stats manifest/manifest_stats.json

python src/data/split_signers.py \
  --data_root "$DATA_ROOT" \
  --out_dir "$SPLIT_DIR" \
  --seed 42 \
  --train_n 18 \
  --val_n 5 \
  --test_n 5 \
  --manifest "$MANIFEST_PATH"
python src/data/preprocess_keypoints.py --config "$CONFIG_PATH"
python src/data/preprocess_roi.py --config "$CONFIG_PATH"
python tools/check_dataset.py --data_root "$DATA_ROOT" --project_root "$ROOT_DIR"
python src/train.py --config "$CONFIG_PATH"
