#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuan/桌面/SignPanv2.0/boundary_prefill_B_final"
PY="/home/yuan/anaconda3/envs/mmpose_cuda126/bin/python"
cd "$ROOT"

run_train_eval() {
  local name="$1"
  local config="$2"
  local logdir="$3"

  echo "==== [$name] TRAIN ===="
  $PY -u scripts/train.py --config "$config" --device cuda:0 --log-interval 50

  local run_dir
  run_dir="$(ls -td "$logdir"/run_* | head -n1)"
  local best_tau
  best_tau="$($PY - <<PY
import json
print(json.load(open("$run_dir/best_tau.json","r",encoding="utf-8"))["best_tau"])
PY
)"

  echo "==== [$name] EVAL test @ tau=${best_tau} ===="
  $PY -u scripts/evaluate_boundary.py \
    --config "$config" \
    --checkpoint "$run_dir/best.pt" \
    --split test \
    --tau "$best_tau" \
    --device cuda:0

  local eval_dir
  eval_dir="$(ls -td "$logdir"/run_* | head -n1)"
  echo "$run_dir" > "$logdir/last_train_run.txt"
  echo "$eval_dir" > "$logdir/last_eval_run.txt"
}

run_train_eval \
  "pose_only_hard" \
  "configs/train_data28_pose_ablation_poseonly_hard.yaml" \
  "logs/ablation_pose_components/pose_only_hard"

run_train_eval \
  "inside_head_hard" \
  "configs/train_data28_pose_ablation_inside_hard.yaml" \
  "logs/ablation_pose_components/inside_head_hard"

run_train_eval \
  "inside_soft" \
  "configs/train_data28_pose_ablation_soft.yaml" \
  "logs/ablation_pose_components/inside_soft"

POSE_EVAL="$(cat logs/ablation_pose_components/pose_only_hard/last_eval_run.txt)/evaluation_report.json"
INSIDE_EVAL="$(cat logs/ablation_pose_components/inside_head_hard/last_eval_run.txt)/evaluation_report.json"
SOFT_EVAL="$(cat logs/ablation_pose_components/inside_soft/last_eval_run.txt)/evaluation_report.json"
SOFT_TRAIN_RUN="$(cat logs/ablation_pose_components/inside_soft/last_train_run.txt)"
SOFT_TAU="$($PY - <<PY
import json
print(json.load(open("$SOFT_TRAIN_RUN/best_tau.json","r",encoding="utf-8"))["best_tau"])
PY
)"

echo "==== BUILD TABLE 3-3 REQUIRED FORMAT ===="
$PY -u scripts/build_table_3_3_required.py \
  --pose-report "$POSE_EVAL" \
  --inside-report "$INSIDE_EVAL" \
  --soft-report "$SOFT_EVAL" \
  --soft-config "configs/train_data28_pose_ablation_soft.yaml" \
  --soft-checkpoint "$SOFT_TRAIN_RUN/best.pt" \
  --soft-tau "$SOFT_TAU" \
  --device cuda:0 \
  --out-prefix "outputs/dataset_stats/table_3_3_span_f1_required_format"

echo "==== DONE ===="
echo "table: outputs/dataset_stats/table_3_3_span_f1_required_format.md"

