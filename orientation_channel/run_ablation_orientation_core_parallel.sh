#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"
RUN_ROOT="${2:-}"
EPOCHS="${3:-36}"
TRAIN_BS="${4:-16}"
NUM_WORKERS="${5:-8}"
EVAL_BS="${6:-8}"
GPU_LIST="${7:-0,1,2,3,4,5}"
EVAL_DEVICE="${8:-cuda}"

if [[ -z "${RUN_ROOT}" ]]; then
  TS="$(date '+%Y%m%d_%H%M%S')"
  RUN_ROOT="outputs_ablation/core_parallel_${TS}"
fi

CONFIGS=(
  "configs/exp_modal_roi.yaml"
  "configs/exp_modal_roi_text.yaml"
  "configs/exp_modal_skel.yaml"
  "configs/exp_modal_skel_text.yaml"
  "configs/exp_modal_roi_skel.yaml"
  "configs/exp_modal_roi_skel_text.yaml"
)

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${#CONFIGS[@]}" ]]; then
  echo "Need at least ${#CONFIGS[@]} GPUs in GPU_LIST, got ${#GPUS[@]}"
  exit 1
fi

mkdir -p "${RUN_ROOT}/tmp_cfg"

echo "[ablation-core-parallel] env=${CONDA_ENV}"
echo "[ablation-core-parallel] run_root=${RUN_ROOT}"
echo "[ablation-core-parallel] epochs=${EPOCHS} train_bs=${TRAIN_BS} workers=${NUM_WORKERS}"
echo "[ablation-core-parallel] eval_bs=${EVAL_BS} eval_device=${EVAL_DEVICE}"
echo "[ablation-core-parallel] gpus=${GPU_LIST}"

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

run_one() {
  local gpu="$1"
  local cfg="$2"
  local base out_dir tmp_cfg
  base="$(basename "${cfg}" .yaml)"
  out_dir="${RUN_ROOT}/${base}"
  tmp_cfg="${RUN_ROOT}/tmp_cfg/${base}.yaml"

  mkdir -p "${out_dir}"
  build_tmp_cfg "${cfg}" "${tmp_cfg}"

  echo "========================================" | tee -a "${out_dir}/run.log"
  echo "[${base}] gpu=${gpu} cfg=${cfg}" | tee -a "${out_dir}/run.log"
  echo "[${base}] tmp_cfg=${tmp_cfg}" | tee -a "${out_dir}/run.log"
  echo "[${base}] out_dir=${out_dir}" | tee -a "${out_dir}/run.log"

  if [[ ! -f "${out_dir}/checkpoints/best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run -n "${CONDA_ENV}" python -m scripts.train_orientation \
      --config "${tmp_cfg}" --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[${base}] skip train (best.pt exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/calibration/temperature.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation \
      --config "${tmp_cfg}" --split val --calibrate \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" \
      --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[${base}] skip val calibrate (temperature exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/results/metrics_test.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation \
      --config "${tmp_cfg}" --split test --use_calibration \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" \
      --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[${base}] skip test eval (metrics_test exists)" | tee -a "${out_dir}/run.log"
  fi

  if [[ ! -f "${out_dir}/results/orientation_top3_test.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
      conda run -n "${CONDA_ENV}" python -m scripts.infer_orientation \
      --config "${tmp_cfg}" --split test --use_calibration \
      --batch_size "${EVAL_BS}" --device "${EVAL_DEVICE}" \
      --out_dir "${out_dir}" 2>&1 | tee -a "${out_dir}/run.log"
  else
    echo "[${base}] skip test infer (top3 exists)" | tee -a "${out_dir}/run.log"
  fi

  echo "[${base}] done" | tee -a "${out_dir}/run.log"
}

pids=()
names=()
for i in "${!CONFIGS[@]}"; do
  run_one "${GPUS[$i]}" "${CONFIGS[$i]}" &
  pids+=($!)
  names+=("$(basename "${CONFIGS[$i]}" .yaml)")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[ablation-core-parallel] failed: ${names[$i]}"
    fail=1
  fi
done

echo "========================================"
if [[ "${fail}" -ne 0 ]]; then
  echo "[ablation-core-parallel] finished with failures"
  exit 1
fi
echo "[ablation-core-parallel] all done"
echo "[ablation-core-parallel] summarize with:"
echo "python -m scripts.summarize_ablation --run_root ${RUN_ROOT}"
