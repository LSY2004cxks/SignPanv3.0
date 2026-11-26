#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"
RUN_ROOT="${2:-}"
EPOCHS="${3:-36}"
TRAIN_BS="${4:-8}"
NUM_WORKERS="${5:-8}"
EVAL_BS="${6:-8}"
EVAL_DEVICE="${7:-cuda}"

if [[ -z "${RUN_ROOT}" ]]; then
  TS="$(date '+%Y%m%d_%H%M%S')"
  RUN_ROOT="outputs_ablation/textkv3_${TS}"
fi

CONFIGS=(
  "configs/exp_modal_roi_text_kv.yaml"
  "configs/exp_modal_skel_text_kv.yaml"
  "configs/exp_modal_roi_skel_text_kv.yaml"
)

mkdir -p "${RUN_ROOT}/tmp_cfg"

echo "[textkv3] env=${CONDA_ENV}"
echo "[textkv3] run_root=${RUN_ROOT}"
echo "[textkv3] epochs=${EPOCHS} train_bs=${TRAIN_BS} workers=${NUM_WORKERS}"
echo "[textkv3] eval_bs=${EVAL_BS} eval_device=${EVAL_DEVICE}"

build_tmp_cfg() {
  local base_cfg="$1"
  local tmp_cfg="$2"
  local base_abs
  base_abs="$(cd "$(dirname "${base_cfg}")" && pwd)/$(basename "${base_cfg}")"
  cat > "${tmp_cfg}" <<EOF
inherit: ${base_abs}
train:
  epochs: ${EPOCHS}
  batch_size: ${TRAIN_BS}
  num_workers: ${NUM_WORKERS}
  device: cuda
EOF
}

for cfg in "${CONFIGS[@]}"; do
  base="$(basename "${cfg}" .yaml)"
  out_dir="${RUN_ROOT}/${base}"
  tmp_cfg="${RUN_ROOT}/tmp_cfg/${base}.yaml"
  mkdir -p "${out_dir}"
  build_tmp_cfg "${cfg}" "${tmp_cfg}"

  echo "========================================" | tee -a "${out_dir}/run.log"
  echo "[textkv3] start ${base}" | tee -a "${out_dir}/run.log"
  echo "[textkv3] cfg=${cfg}" | tee -a "${out_dir}/run.log"
  echo "[textkv3] tmp_cfg=${tmp_cfg}" | tee -a "${out_dir}/run.log"
  echo "[textkv3] out_dir=${out_dir}" | tee -a "${out_dir}/run.log"

  if [[ ! -f "${out_dir}/checkpoints/best.pt" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.train_orientation \
      --config "${tmp_cfg}" --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[textkv3] skip train (best.pt exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/calibration/temperature.json" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation \
      --config "${tmp_cfg}" --split val --calibrate \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[textkv3] skip val calibrate (temperature exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/results/metrics_test.json" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation \
      --config "${tmp_cfg}" --split test --use_calibration \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[textkv3] skip test eval (metrics_test exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/results/orientation_top3_test.jsonl" ]]; then
    conda run -n "${CONDA_ENV}" python -m scripts.infer_orientation \
      --config "${tmp_cfg}" --split test --use_calibration \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[textkv3] skip test infer (top3 exists)" | tee -a "${out_dir}/run.log"
  fi

  echo "[textkv3] done ${base}" | tee -a "${out_dir}/run.log"
done

echo "========================================"
echo "[textkv3] all done"
echo "[textkv3] summarize with:"
echo "python -m scripts.summarize_ablation --run_root ${RUN_ROOT}"
