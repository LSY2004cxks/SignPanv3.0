#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"
ENV_FILE="${2:-environment.gpu-cu128-py312.yml}"
RUN_ROOT="${3:-}"
EPOCHS="${4:-36}"
TRAIN_BS="${5:-16}"
NUM_WORKERS="${6:-8}"
EVAL_BS="${7:-8}"
GPU_LIST="${8:-0,1,2,3,4,5}"
EVAL_DEVICE="${9:-cuda}"

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="outputs_ablation/core_parallel_$(date +%Y%m%d_%H%M%S)"
fi

echo "[oneclick] root=${ROOT_DIR}"
echo "[oneclick] env=${CONDA_ENV} env_file=${ENV_FILE}"
echo "[oneclick] run_root=${RUN_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[oneclick] conda not found. Please install Miniconda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[oneclick] update existing conda env: ${CONDA_ENV}"
  conda env update -n "${CONDA_ENV}" -f "${ENV_FILE}" --prune
else
  echo "[oneclick] create conda env: ${CONDA_ENV}"
  conda env create -n "${CONDA_ENV}" -f "${ENV_FILE}"
fi

echo "[oneclick] verify torch/cuda"
conda run -n "${CONDA_ENV}" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
PY

echo "[oneclick] launch core 6 experiments in parallel"
bash scripts/run_ablation_orientation_core_parallel.sh \
  "${CONDA_ENV}" \
  "${RUN_ROOT}" \
  "${EPOCHS}" \
  "${TRAIN_BS}" \
  "${NUM_WORKERS}" \
  "${EVAL_BS}" \
  "${GPU_LIST}" \
  "${EVAL_DEVICE}"
