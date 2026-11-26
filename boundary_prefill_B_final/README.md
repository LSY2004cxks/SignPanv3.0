# boundary_prefill_B_final

单手事件边界切分（Boundary Proposal）+ Prefill-B 导出工程。

## 功能
- 单手 L/R 严格分开训练与推理
- 标注 schema 自动识别（A/B/C）并写入 `schema_report.json`
- Pose 特征预处理：score gating + 插值 + 中值滤波 + root/scale 归一化
- 可选 Flow 特征：Pose 引导 ROI + Dual TV-L1 统计（`mu/p95/p99/delta_mu/u_mean/v_mean`）
- 双流融合：`early`（帧级拼接）与 `gated`（学习门控系数）
- 后处理配对策略：`infer.pair_mode = score_then_nearest`（可切 `nearest` / `score`）
- Stage-A 模型：TCN + 三头输出（start/end/inside）
- 高斯软标签（`sigma=3`, `radius=9`）
- Focal BCE(start/end) + BCE(inside, weight=0.5)
- 训练默认：`AdamW`、`lr=1e-3`、`weight_decay=1e-2`、`batch_size=16`、`epochs=50`
- 早停：监控 `Mean F1@7`（可配）
- 后处理：peak picking + NMS + start/end pairing（`Lmin=6`, `Lmax=80`）
- 导出 `outputs/prefill_B/<split>/<video_id>.json`

## 结构
- `configs/default.yaml`
- `src/`：data/features/models/infer/metrics
- `scripts/`：split/train/infer/eval/vis/export
- `splits/signer_split.json`
- `logs/run_*/`
- `outputs/prefill_B/*`, `outputs/vis/*`

## 快速命令

### 1) 生成 18/5/5 signer split
```bash
python scripts/make_split_18_5_5.py --data-root ../data28 --output splits/signer_split.json
```

### 2) 训练（默认按 split 的 train/val signer）
```bash
python scripts/train.py --config configs/default.yaml
```

训练日志目录会保存：
- `config.yaml`
- `code_version.json`（python/torch/git hash）
- `fps_report.json`
- `metrics.csv`
- `tau_scan.csv` / `tau_scan.png`
- `best.pt`

### 3) 推理并导出 Prefill-B
```bash
python scripts/infer.py --config configs/default.yaml --checkpoint logs/run_xxx/best.pt --split test
```

### 4) 评估与 tau 扫描
```bash
python scripts/evaluate_boundary.py --config configs/default.yaml --checkpoint logs/run_xxx/best.pt --split val
```
评估输出包含：
- 帧级：`start/end/mean F1@3/5/7`
- span级：`IoU@0.3/0.5/0.7` 与 `Avg proposals/video`
- 表格文件：`table_3_2_frame_f1.csv`、`table_3_3_span_f1.csv`

### 5) 时间线可视化
```bash
python scripts/vis_timeline.py --config configs/default.yaml --checkpoint logs/run_xxx/best.pt --split val --K 20
```

### 6) 数据集统计与事件长度直方图（Figure 3-1）
```bash
python tools/report_dataset_stats.py \
  --video-root /home/yuan/桌面/SignPanv2.0/data28 \
  --label-root /home/yuan/桌面/SignPanv2.0/data28 \
  --pose-root /home/yuan/桌面/SignPanv2.0/data28_pose \
  --split-file splits/signer_split_data28.json \
  --out-dir outputs/dataset_stats
```

## 预处理工程（已与训练工程分离）
预处理脚本已迁移到独立目录：

- `/home/yuan/桌面/SignPanv2.0/boundary_prefill_B_preprocess/scripts`

其中光流提取脚本：
```bash
python /home/yuan/桌面/SignPanv2.0/boundary_prefill_B_preprocess/scripts/extract_flow_tvl1_pose_roi.py \
  --data-root /home/yuan/桌面/SignPanv2.0/data28 \
  --pose-root /home/yuan/桌面/SignPanv2.0/data28_pose \
  --out-root /home/yuan/桌面/SignPanv2.0/data28_flow_tvl1
```

若启用 flow 拼接训练，配置项：
- `features.flow.enabled: true`
- `features.flow.root: /path/to/data28_flow_tvl1`
- `features.expected_dim: 29`（23 pose + 6 flow）
- `model.fusion_mode: early`（早期融合，默认）或 `gated`（门控融合）

例如：
```bash
python /home/yuan/桌面/SignPanv2.0/boundary_prefill_B_preprocess/scripts/extract_pose_mmpose.py \
  --video-root data28/videos \
  --output-root data28/pose \
  --device mps
```

> 注意：不同数据目录布局可通过 `--video-root/--label-root/--pose-root` 覆盖。
