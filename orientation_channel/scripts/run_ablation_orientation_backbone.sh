#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${1:-pan}"
TS="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="outputs_ablation/backbone_${TS}"
mkdir -p "${RUN_ROOT}"

CONFIGS=(
  "configs/exp_backbone_resnet18_full_stgcn_img.yaml"
  "configs/exp_backbone_convnext_tiny_full_stgcn_img.yaml"
)

echo "[ablation-backbone] env=${CONDA_ENV}"
echo "[ablation-backbone] run_root=${RUN_ROOT}"

# img-mode backbone comparison requires ROI images either from precomputed events
# (key: roi_img) or external ROI cache (key: roi).
python - <<'PY'
from pathlib import Path
import numpy as np

pre = Path("/home/yuan/桌面/SignPanv2.0/data28/precomputed/events")
ext = Path("/home/yuan/桌面/SignPanv2.0/data28_roi_fixlen_mmpose")

def check_npz(files, key):
    for p in files[:50]:
        with np.load(p, allow_pickle=True) as d:
            if key in d:
                arr = d[key]
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return True
    return False

ok = False
if pre.exists():
    files = sorted(pre.glob("*.npz"))
    ok = check_npz(files, "roi_img")
if (not ok) and ext.exists():
    files = sorted(ext.rglob("*.npz"))
    ok = check_npz(files, "roi")

if not ok:
    raise SystemExit(
        "Backbone img-mode needs ROI image cache. Neither precomputed/events(roi_img) "
        "nor external data28_roi_fixlen_mmpose(roi) is ready."
    )
print("[check] ROI image cache found.")
PY

for cfg in "${CONFIGS[@]}"; do
  base="$(basename "${cfg}" .yaml)"
  out_dir="${RUN_ROOT}/${base}"
  mkdir -p "${out_dir}"
  echo "========================================"
  echo "[ablation-backbone] start ${cfg}"
  echo "[ablation-backbone] out   ${out_dir}"

  conda run -n "${CONDA_ENV}" python -m scripts.train_orientation --config "${cfg}" --out_dir "${out_dir}"
  conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation --config "${cfg}" --split val --calibrate --out_dir "${out_dir}"
  conda run -n "${CONDA_ENV}" python -m scripts.eval_orientation --config "${cfg}" --split test --use_calibration --out_dir "${out_dir}"
  conda run -n "${CONDA_ENV}" python -m scripts.infer_orientation --config "${cfg}" --split test --use_calibration --out_dir "${out_dir}"

  echo "[ablation-backbone] done  ${cfg}"
done

echo "========================================"
echo "[ablation-backbone] all done"
echo "[ablation-backbone] summarize with:"
echo "python -m scripts.summarize_ablation --run_root ${RUN_ROOT}"
