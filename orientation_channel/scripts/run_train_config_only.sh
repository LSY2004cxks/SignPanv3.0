#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_path> [conda_env=pan] [out_dir]"
  exit 1
fi

CFG="$1"
CONDA_ENV="${2:-pan}"
OUT_DIR="${3:-}"

echo "[train] env=${CONDA_ENV}"
echo "[train] cfg=${CFG}"

if [[ -n "${OUT_DIR}" ]]; then
  echo "[train] out_dir=${OUT_DIR}"
  conda run -n "${CONDA_ENV}" python -m scripts.train_orientation --config "${CFG}" --out_dir "${OUT_DIR}"
else
  conda run -n "${CONDA_ENV}" python -m scripts.train_orientation --config "${CFG}"
fi

