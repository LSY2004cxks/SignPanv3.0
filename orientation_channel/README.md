# Orientation Channel (Event-level Palm Orientation)

A reproducible PyTorch project for signer-disjoint event-level palm orientation classification.

## Key Features
- Signer-disjoint split (18/5/5) with reproducible seed
- Event-level Top-3 outputs + HITL gating
- Multi-modal inputs: ROI RGB + 3D landmarks + palm_normal + geometry encoder
- Geometry encoder switchable: ST-GCN (default) or Skeleton Transformer
- Cross-attention fusion + attention pooling
- Temperature scaling calibration + ECE + reliability diagram

## Data Layout (default, configurable)
```
/home/yuan/桌面/SignPanv2.0/data28
  metadata/signers.csv               # video_id, signer_id
  annotations/mhsl_events.jsonl      # event records
  videos/<video_id>.mp4              # or frames/<video_id>/{frame:06d}.jpg
  precomputed/                       # optional cache
```

## 0) Create Environment (GPU, CUDA 12.4)
```bash
cd /home/yuan/桌面/SignPanv2.0/orientation_channel
conda env create -f environment.gpu-cu124.yml
conda activate pan
```
If you use Blackwell GPUs (e.g., RTX 50 series), use CUDA 12.8 env:
```bash
conda env create -f environment.gpu-cu128.yml
conda activate pan
```
If your image is Python 3.12 (recommended on rented RTX 50 servers):
```bash
conda env create -f environment.gpu-cu128-py312.yml
conda activate pan
```
If env `pan` already exists:
```bash
conda env update -n pan -f environment.gpu-cu124.yml --prune
conda activate pan
```

## 1) Prepare Data28 Manifest
```bash
cd /home/yuan/桌面/SignPanv2.0/orientation_channel
python -m scripts.prepare_data28_manifest \
  --data_root /home/yuan/桌面/SignPanv2.0/data28 \
  --label_root /home/yuan/桌面/SignPanv2.0/data28
```

## 2) Make signer-disjoint splits
```bash
python -m scripts.make_split_signer_disjoint \
  --data_root /home/yuan/桌面/SignPanv2.0/data28 \
  --seed 42 \
  --train_signers 18 --val_signers 5 --test_signers 5 \
  --out_dir /home/yuan/桌面/SignPanv2.0/data28/split
```

If you already use constrained signers (for example fixed val/test identities), keep using the generated files in:
`/home/yuan/桌面/SignPanv2.0/data28/split/{train,val,test}.jsonl`.

This project now supports two cache formats:
- native `data28/precomputed/events/*.npz` (keys like `roi_img`, `landmarks`)
- external dual-cache used by handshape experiments:
  - `/home/yuan/桌面/SignPanv2.0/data28_roi_fixlen_mmpose` (key `roi`)
  - `/home/yuan/桌面/SignPanv2.0/data28_mp_hand3d_effective` (key `xyz`)

## 3) Preprocess (MediaPipe -> ROI + landmarks + normal)
```bash
python -m scripts.preprocess_mediapipe_hands \
  --data_root /home/yuan/桌面/SignPanv2.0/data28 \
  --split /home/yuan/桌面/SignPanv2.0/data28/split/train.jsonl \
  --out_dir /home/yuan/桌面/SignPanv2.0/data28/precomputed
```
If you want to cache ROI features directly, add `--extract_feat`. Otherwise, keep `roi_input: img` or `roi_input: auto` in config.

## 4) Build text prototypes (CLIP)
```bash
python -m scripts.build_text_prototypes \
  --labels_file configs/orientation_labels.txt \
  --prompt_file configs/orientation_prompts.txt \
  --out_path /home/yuan/桌面/SignPanv2.0/data28/precomputed/prototypes.pt
```

## 5) Train
```bash
python -m scripts.train_orientation \
  --config configs/full_stgcn.yaml
```

## 6) Evaluate
```bash
python -m scripts.eval_orientation \
  --config configs/full_stgcn.yaml \
  --split val
```
Calibration (temperature scaling + ECE + reliability diagram) on val:
```bash
python -m scripts.eval_orientation --config configs/full_stgcn.yaml --split val --calibrate
```

## 7) Infer & Export Top-3
```bash
python -m scripts.infer_orientation \
  --config configs/full_stgcn.yaml \
  --split test
```

## Geometry Encoder Ablations
- ST-GCN: `configs/full_stgcn.yaml`
- Skeleton Transformer: `configs/full_transformer.yaml`

## Recommended Experiment Matrix (Paper-ready)
Core ablations (ResNet-18 baseline):
- `configs/exp_modal_roi.yaml`
- `configs/exp_modal_roi_text.yaml`
- `configs/exp_modal_skel.yaml`
- `configs/exp_modal_skel_text.yaml`
- `configs/exp_modal_roi_skel.yaml`
- `configs/exp_modal_roi_skel_text.yaml`

Backbone comparison (same setting, image input):
- `configs/exp_backbone_resnet18_full_stgcn_img.yaml`
- `configs/exp_backbone_convnext_tiny_full_stgcn_img.yaml`

Note: backbone comparison must use `roi_input: img`, otherwise precomputed `roi_feat` may hide backbone differences.

## Batch Run Scripts
Single config:
```bash
bash scripts/run_train_config_only.sh configs/exp_resnet18_full_stgcn.yaml pan
```

Run core ablations:
```bash
bash scripts/run_ablation_orientation_core.sh pan
```

Run core ablations on 6 GPUs in parallel (recommended for rented multi-GPU server):
```bash
bash scripts/run_ablation_orientation_core_parallel.sh \
  pan \
  outputs_ablation/core_parallel_YYYYMMDD_HHMMSS \
  36 16 8 8 \
  0,1,2,3,4,5 \
  cuda
```
Arguments:
- `$1` conda env
- `$2` run_root
- `$3` epochs override
- `$4` train batch size
- `$5` num_workers
- `$6` eval/infer batch size
- `$7` gpu list (comma separated)
- `$8` eval device (`cuda` or `cpu`)

One-click setup + start (Blackwell/RTX50, py312 image):
```bash
bash scripts/remote_oneclick_train_core6.sh \
  pan \
  environment.gpu-cu128-py312.yml \
  outputs_ablation/core_parallel_$(date +%Y%m%d_%H%M%S) \
  36 16 8 8 \
  0,1,2,3,4,5 \
  cuda
```

Run backbone comparison:
```bash
bash scripts/run_ablation_orientation_backbone.sh pan
```

Summarize one batch run:
```bash
python -m scripts.summarize_ablation --run_root outputs_ablation/core_YYYYMMDD_HHMMSS
```

## Outputs
- checkpoints: `outputs/checkpoints/{best.pt,last.pt}`
- metrics: `outputs/logs/metrics.csv`
- calibration: `outputs/calibration/temperature.json`, `reliability.png`
- results: `outputs/results/orientation_top3_<split>.jsonl`
