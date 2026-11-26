#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"

CONFIGS=(
  "configs/handshape_data28_schemeA_fold1_k24_skel_only.yaml"
  "configs/handshape_data28_schemeA_fold1_k24_roi_only.yaml"
  "configs/handshape_data28_schemeA_fold1_k24_fusion.yaml"
  "configs/handshape_data28_schemeA_fold1_k24_deform_no_nce.yaml"
  "configs/handshape_data28_schemeA_fold1_k24.yaml"
)

echo "[ablation] env=${CONDA_ENV}"
for cfg in "${CONFIGS[@]}"; do
  echo "========================================"
  echo "[ablation] start ${cfg}"
  conda run -n "${CONDA_ENV}" python src/train.py --config "${cfg}"
  echo "[ablation] done  ${cfg}"
done

