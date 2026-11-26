#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/handshape_data28_schemeA_fold1_k24.yaml}"
CONDA_ENV="${2:-pan}"

echo "[train] env=${CONDA_ENV} config=${CONFIG_PATH}"
conda run -n "${CONDA_ENV}" python src/train.py --config "${CONFIG_PATH}"

