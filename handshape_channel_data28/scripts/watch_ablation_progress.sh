#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

interval="${1:-2}"

while true; do
  if [[ -n "${TERM:-}" ]]; then
    clear || true
  fi
  now="$(date '+%F %T')"
  echo "[watch] ${now}"

  # Find active training process and config path.
  line="$(ps -eo pid,cmd | grep 'src/train.py --config configs/handshape_data28_schemeA_fold1_k24' | grep -v grep | head -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "No active SchemeA fold-1 ablation train process found."
    echo
    echo "Recent runs:"
    ls -dt runs_schemeA/handshape_data28_schemeA_fold1_k24* 2>/dev/null | head -n 5 || true
    sleep "${interval}"
    continue
  fi

  pid="$(awk '{print $1}' <<<"${line}")"
  cfg="$(sed -E 's/.*--config ([^ ]+).*/\1/' <<<"${line}")"
  cfg_base="$(basename "${cfg}" .yaml)"
  run_dir="$(ls -dt runs_schemeA/${cfg_base}_* 2>/dev/null | head -n 1 || true)"

  echo "PID: ${pid}"
  echo "Config: ${cfg}"
  echo "Run dir: ${run_dir:-N/A}"

  if [[ -z "${run_dir}" || ! -f "${run_dir}/train.log" ]]; then
    echo
    echo "train.log not ready yet."
    sleep "${interval}"
    continue
  fi

  total_epochs="$(python - <<'PY' "${cfg}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(int(cfg.get('training', {}).get('epochs', 50)))
PY
)"

  last_epoch_line="$(grep '^\[epoch ' "${run_dir}/train.log" | tail -n 1 || true)"
  if [[ -z "${last_epoch_line}" ]]; then
    echo
    echo "No epoch line yet. Waiting for first validation log..."
    tail -n 5 "${run_dir}/train.log" || true
    sleep "${interval}"
    continue
  fi

  cur_epoch="$(sed -E 's/^\[epoch[[:space:]]+([0-9]+)\].*/\1/' <<<"${last_epoch_line}")"
  cur_epoch_num=$((10#${cur_epoch}))
  total_epochs_num=$((10#${total_epochs}))
  percent=$(( cur_epoch_num * 100 / total_epochs_num ))

  # Draw simple ASCII progress bar.
  width=40
  fill=$(( percent * width / 100 ))
  bar="$(printf '%*s' "${fill}" '' | tr ' ' '=')"
  rem=$(( width - fill ))
  bar="${bar}$(printf '%*s' "${rem}" '' | tr ' ' '.')"

  echo
  echo "Progress: [${bar}] ${percent}% (${cur_epoch_num}/${total_epochs_num})"
  echo "Latest:   ${last_epoch_line}"
  echo
  echo "Recent log tail:"
  tail -n 8 "${run_dir}/train.log" || true

  sleep "${interval}"
done
