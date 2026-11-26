# Location Channel (Event-Level) — SignPan v2.0 Data28

本项目从零构建“位置通道 (location channel) 事件级自动标注”的可训练/可推理工程，面向 28 signer 数据集。**严格 signer‑disjoint** (18/5/5) 划分；不使用 gloss。

## ✅ 特性
- **双模型**：ProtoDist（距离基线） + FrameMLP+MIL（主模型）
- **关键点链路**：MMpose pose 作为身体坐标系，MediaPipe Hand 作为手点来源
- **坐标对齐**：支持记录 resize/crop 变换并反变换回原图像素坐标
- **事件级输出**：Top‑1 / Top‑3 / Macro‑F1 / Macro‑Recall
- **可审计**：预测输出 JSONL，包含 `event_id / signer_id / top3 / margin`
- **日志**：TensorBoard + jsonlines

---

## 0. 环境安装

```bash
cd /home/yuan/桌面/SignPanv2.0/location_channel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **MMpose 依赖提示**：需要 `mmpose + mmcv`。请根据你的CUDA版本安装。配置文件中填写MMpose config/weights。

---

## 1. 数据准备（labels）

### 1.1 配置映射
编辑 `configs/dataset.yaml`：
- `paths.ann_dir`：标注 JSON 文件夹
- `paths.video_root`：视频根目录
- `mapping.*`：字段映射 (signer_id, events, span, location)

### 1.2 生成 labels.csv

```bash
python tools/make_labels.py \
  --config configs/dataset.yaml \
  --ann_dir /home/yuan/桌面/SignPanv2.0/data28/annotations \
  --out data/labels.csv
```

输出字段包含：
```
sample_id, event_id, signer_id, video_id, hand, start_frame, end_frame, label_loc, clip_path, video_path, fps
```

---

## 2. 提取关键点（MMpose + MediaPipe）

```bash
python tools/extract_keypoints.py \
  --config configs/default.yaml \
  --labels data/labels.csv \
  --out data/features \
  --num_frames 24
```

输出每个事件一个 `.npz`：
```
pose_kpts[T,J,2], pose_score[T,J]
hand_point[T,2], hand_score[T]
valid_mask[T]
```
并保存 `*.meta.json`（含frame索引与变换信息）。

说明：事件抽帧默认固定 `T=24`，短事件会按均匀插值重复采样到固定长度（与其他信道保持一致）。

---

## 3. 生成 signer‑disjoint split (18/5/5)

```bash
python tools/make_splits.py \
  --labels data/labels.csv \
  --out data/splits_18_5_5 \
  --seed 42
```

输出：
- train.txt / val.txt / test.txt
- split_manifest.json（含signer列表与统计）

---

## 4. 训练

### 4.1 FrameMLP + MIL（主模型）

```bash
python -m src.train.train \
  --config configs/model_mlp_mil.yaml \
  --labels data/labels.csv \
  --features data/features \
  --splits data/splits_18_5_5 \
  --run_name loc_mlp_mil
```

### 4.2 ProtoDist（距离基线）

```bash
python -m src.train.train \
  --config configs/model_protodist.yaml \
  --labels data/labels.csv \
  --features data/features \
  --splits data/splits_18_5_5 \
  --run_name loc_protodist
```

---

## 5. 评估（必须输出预测jsonl）

```bash
python -m src.train.eval --run_dir runs/loc_mlp_mil
```

输出：
- `predictions_val.jsonl`
- `predictions_test.jsonl`
- `summary_val.json`, `summary_test.json`
- `confusion_val.csv`, `confusion_test.csv`

---

## 6. 推理（单事件/批量）

```bash
python -m src.train.infer \
  --run_dir runs/loc_mlp_mil \
  --labels data/labels.csv \
  --features data/features \
  --splits data/splits_18_5_5 \
  --split test \
  --out predictions.jsonl
```

每条jsonl含：
```
event_id, signer_id, y_true, y_pred_top1, top3, prob_top3, margin
```

---

## 7. 自检 / 单元测试

```bash
python tests/test_geometry.py
```

---

## 注意事项
- **不要使用 gloss 作为输入或监督**（本工程未使用）。
- 如使用 resize/crop，请在 `extract_keypoints.py` 中设置变换并启用反变换回原图坐标。
- `configs/location_labels.json` 固定为 31 类。
- MMpose 配置与权重务必指定（`configs/default.yaml -> pose`）。
- 默认拓扑参照包含肩/肘/髋 + 眼/鼻/嘴中心（`mouth_center`；若缺少口部关键点则回退到 `nose`）。
