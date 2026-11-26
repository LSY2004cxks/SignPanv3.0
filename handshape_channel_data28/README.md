# handshape_channel_data28

Event-level handshape classification pipeline for `mhsl-1.0-lite` `data-28`.

- Target label: `handshape.label` (single-label classification)
- Keep events only when:
  - `event_type == "effective"`
  - `handshape.label != "NOH"`
- Signer split: signer-disjoint `18/5/5` with fixed `seed=42`
- Inference output: Top-3 candidates + gating flag for Top-1 autofill
- Original annotation JSON files are never modified

## 1. Environment (Python 3.10)

```bash
cd /home/yuan/桌面/Sign-PAN-Project/experiment2/handshape_channel_data28

# Example with conda
conda create -n hs28 python=3.10 -y
conda activate hs28

pip install -r requirements.txt

# Recommended for OpenMMLab compatibility
pip install -U openmim
mim install "mmengine>=0.10" "mmcv>=2.1" "mmdet>=3.3" "mmpose>=1.3"
```

## 2. Data assumptions

Default config uses:
- `data_root: ../data28`

Manifest scanner behavior:
- Recursively scans all `*.json`
- Uses `video.path` + normalized `signer_id` to locate video
- If not found, writes orphan record to `manifest/orphans.jsonl`

Split behavior:
- Uses top-level signer video folders (`S*`, excluding `*-label`)
- Always creates 28-signer `18/5/5` split (unannotated signer may have 0 samples)

## 3. Full pipeline (from scratch)

### Step A: Build manifest

```bash
python src/data/build_manifest.py \
  --data_root ../data-28 \
  --out_manifest manifest/manifest.jsonl \
  --out_orphans manifest/orphans.jsonl \
  --out_stats manifest/manifest_stats.json
```

### Step B: Split signers

```bash
python src/data/split_signers.py \
  --data_root ../data28 \
  --out_dir splits \
  --seed 42 --train_n 18 --val_n 5 --test_n 5 \
  --manifest manifest/manifest.jsonl
```

### Step B2: Scheme A (fixed test + 5-fold dev CV, force `S109` in train)

```bash
python src/data/split_signers_schemeA.py \
  --data_root ../data28 \
  --manifest manifest/manifest.jsonl \
  --out_dir splits_schemeA \
  --seed 42 \
  --num_folds 5 \
  --test_n 5 \
  --anchor_train_signers S109 \
  --must_test_signers S48,S21,S57 \
  --must_val_signers S20,S92,S65
```

Outputs:
- `splits_schemeA/fixed_test_signers.txt`
- `splits_schemeA/fold_{1..5}/{train,val,test}_signers.txt`
- `splits_schemeA/schemeA_meta.json`

### Step C: Keypoint preprocessing (mmpose/mediapipe/hybrid)

```bash
python src/data/preprocess_keypoints.py --config configs/handshape_data28.yaml
```

Notes:
- Default engine is `mmpose`
- If `data28_video_health.csv` exists and provides `best_engine`, it overrides per video
- `hybrid` runs mmpose + mediapipe and chooses by success + larger bbox side + higher confidence
- Dual failure uses previous bbox fallback up to 3 frames and marks low quality

### Step D: ROI + skeleton cache and tiering

```bash
python src/data/preprocess_roi.py --config configs/handshape_data28.yaml
```

Outputs:
- `cache/roi/{sample_id}.npy` `(K,224,224,3)`
- `cache/skeleton/{sample_id}.npy` `(K,V,3)`
- `cache/meta/{sample_id}.json` includes `bbox_side_px_median`, `tier`, `ROI_TOO_SMALL`

### Step E: Dataset health check

```bash
python tools/check_dataset.py --data_root ../data28 --project_root .
```

Checks:
- split signer lists
- sample/label coverage per split
- unseen labels in val/test compared with train (error if non-zero)
- tier distribution and ROI_TOO_SMALL ratio

### Step F: Train

```bash
python src/train.py --config configs/handshape_data28.yaml
```

Training outputs in `runs/<exp_name_timestamp>/`:
- `config.yaml`
- `metrics.json`
- `best.ckpt`
- `val_predictions.csv`
- `requirements_lock.txt`
- `git_hash.txt` (if available)

### Step G: Inference + pred.json

```bash
python src/infer.py \
  --config configs/handshape_data28.yaml \
  --ckpt runs/<exp_name>/best.ckpt \
  --split test
```

Outputs in checkpoint directory:
- `test_predictions.csv`
- `pred.json` (`schema: handshape_pred_v1`)

## 4. One-click scripts

```bash
bash scripts/run_train.sh configs/handshape_data28.yaml
bash scripts/run_infer.sh runs/<exp_name>/best.ckpt test configs/handshape_data28.yaml
```

## 4.1 Paper-aligned ROI+Skeleton (K=24, SchemeA fold-1)

```bash
# 1) build keypoints/roi/skeleton cache first
python src/data/preprocess_keypoints.py --config configs/handshape_data28_schemeA_fold1_k24.yaml
python src/data/preprocess_roi.py --config configs/handshape_data28_schemeA_fold1_k24.yaml

# 2) train
python src/train.py --config configs/handshape_data28_schemeA_fold1_k24.yaml
```

Notes:
- This config enforces `roi_skel_deform_nce` only (`model.enforce_roi_skel_only=true`).
- It disables on-the-fly fallback and requires complete cache.
- It enables ROI expansion for small boxes (`roi_expand_small=true`) to improve hand visibility.

### Recommended ablations (same split/K/cache)

Configs:
- `configs/handshape_data28_schemeA_fold1_k24_skel_only.yaml`
- `configs/handshape_data28_schemeA_fold1_k24_roi_only.yaml`
- `configs/handshape_data28_schemeA_fold1_k24_fusion.yaml`
- `configs/handshape_data28_schemeA_fold1_k24_deform_no_nce.yaml`
- `configs/handshape_data28_schemeA_fold1_k24.yaml` (full: deform+NCE)

Run one:
```bash
bash scripts/run_train_config_only.sh configs/handshape_data28_schemeA_fold1_k24_fusion.yaml pan
```

Run all:
```bash
bash scripts/run_ablation_schemeA_fold1.sh pan
```

## 5. CSV / JSON output schema

`*_predictions.csv` columns:
- `sample_id,gt,top1_label,top1_prob,top2_label,top2_prob,top3_label,top3_prob,margin,entropy,tier,autofill_top1`

`pred.json` schema:
- `schema = handshape_pred_v1`
- `model` block with experiment info
- `items[]` with
  - `sample_id, video_id, hand, event_id, span, gt, tier`
  - `topk` list of `{label, prob}`
  - `margin, entropy, autofill_top1`

## 6. Gating rule for Top-1 autofill

Autofill is true only when all are satisfied:
- `tier in {A, B}`
- `margin (p1 - p2) >= 0.2`
- `p1 >= 0.7`

Otherwise UI should show Top-3 only.
