#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"
RUN_ROOT="${2:-}"
if [[ -z "${RUN_ROOT}" ]]; then
  TS="$(date '+%Y%m%d_%H%M%S')"
  RUN_ROOT="outputs_ablation/core_${TS}"
fi
EVAL_BS="${3:-2}"
EVAL_DEVICE="${4:-cpu}"
mkdir -p "${RUN_ROOT}"

CONFIGS=(
  "configs/exp_resnet18_rgb_only.yaml"
  "configs/exp_resnet18_rgb_normal.yaml"
  "configs/exp_resnet18_rgb_text.yaml"
  "configs/exp_resnet18_full_stgcn_no_text.yaml"
  "configs/exp_resnet18_full_stgcn.yaml"
  "configs/exp_resnet18_full_transformer.yaml"
)

echo "[ablation-core] env=${CONDA_ENV}"
echo "[ablation-core] run_root=${RUN_ROOT}"
echo "[ablation-core] eval_bs=${EVAL_BS}"
echo "[ablation-core] eval_device=${EVAL_DEVICE}"

for cfg in "${CONFIGS[@]}"; do
  base="$(basename "${cfg}" .yaml)"
  out_dir="${RUN_ROOT}/${base}"
  mkdir -p "${out_dir}"
  echo "========================================"
  echo "[ablation-core] start ${cfg}"
  echo "[ablation-core] out   ${out_dir}"
  if [[ ! -f "${out_dir}/checkpoints/best.pt" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.train_orientation --config "${cfg}" --out_dir "${out_dir}"
  else
    echo "[ablation-core] skip train (best.pt exists)"
  fi
  if [[ ! -f "${out_dir}/calibration/temperature.json" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation --config "${cfg}" --split val --calibrate --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}"
  else
    echo "[ablation-core] skip val calibrate (temperature exists)"
  fi
  if [[ ! -f "${out_dir}/results/metrics_test.json" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation --config "${cfg}" --split test --use_calibration --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}"
  else
    echo "[ablation-core] skip test eval (metrics_test exists)"
  fi
  if [[ ! -f "${out_dir}/results/orientation_top3_test.jsonl" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.infer_orientation --config "${cfg}" --split test --use_calibration --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}"
  else
    echo "[ablation-core] skip test infer (top3 exists)"
  fi

  echo "[ablation-core] done  ${cfg}"
done

echo "========================================"
echo "[ablation-core] all done"
echo "[ablation-core] summarize with:"
echo "python -m scripts.summarize_ablation --run_root ${RUN_ROOT}"
