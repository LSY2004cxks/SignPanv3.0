# 盲审冻结与后续扩展计划（2026-03-02）

## 1. 本次盲审采用的冻结结果（直接可用）

### 1.1 帧级边界主表（表 3-2）
- 结果表：
  - `outputs/dataset_stats/table_3_2_frame_f1_test.md`
  - `outputs/dataset_stats/table_3_2_frame_f1_test.csv`
  - `outputs/dataset_stats/table_3_2_frame_f1_test.json`
- 口径：`每模型用 val 最优 tau 在 test 上评估`
- 对应 test 报告：
  - Pose-only: `logs/run_20260301_215611/evaluation_report.json`
  - Flow-only: `logs/flow_only_farneback/run_20260301_220142/evaluation_report.json`
  - Fusion (concat): `logs/pose_flow_farneback/run_20260301_220744/evaluation_report.json`
  - Fusion (gated): `logs/pose_flow_gated_farneback/run_20260301_221330/evaluation_report.json`

### 1.2 Span 级跨模态对比（表 3-3）
- 结果表：
  - `outputs/dataset_stats/table_3_3_span_f1_test.md`
  - `outputs/dataset_stats/table_3_3_span_f1_test.csv`
  - `outputs/dataset_stats/table_3_3_span_f1_test.json`
- 口径同上（val-best tau -> test）。

### 1.3 组件消融专用表（你要求的格式）
- 结果表：
  - `outputs/dataset_stats/table_3_3_span_f1_required_format.md`
  - `outputs/dataset_stats/table_3_3_span_f1_required_format.csv`
  - `outputs/dataset_stats/table_3_3_span_f1_required_format.json`
- 行定义：
  - `Pose-only` / `+ inside head` / `+ soft boundary` / `+ validity filter`
- 其中 `+validity filter` 的 after 指标基于 `spans_recommended` 统计。

### 1.4 盲审正文建议使用的 checkpoint（已存在）
- Pose-only: `logs/run_20260228_215646/best.pt`
- Flow-only: `logs/flow_only_farneback/run_20260301_152114/best.pt`
- Fusion (concat): `logs/pose_flow_farneback/run_20260301_160341/best.pt`
- Fusion (gated): `logs/pose_flow_gated_farneback/run_20260301_210004/best.pt`

### 1.5 数据统计与图（论文工作量支撑）
- 事件统计：`outputs/dataset_stats/dataset_stats.json`
- 事件表：`outputs/dataset_stats/event_table_effective.csv`
- 图 3-1（左右手长度分布）：
  - `outputs/dataset_stats/figure_3_1_event_length_hist_LR.png`
  - `outputs/dataset_stats/figure_3_1_event_length_hist_LR.pdf`

## 2. 已做快照备份（防止后续覆盖）
- 目录：`outputs/experiment_snapshots/20260301_234506`
- 包含：
  - 当时 `dataset_stats` 全部结果
  - 关键 run 目录（用于回滚核对）

## 3. 盲审阶段结论（可直接写入）
- 帧级定位：Fusion(gated) 最优（表 3-2）。
- 事件级 span：Pose-only 在当前配对策略下可更稳（表 3-3 跨模态对比）。
- 组件消融：`soft boundary` 是主要增益来源；`validity filter` 主要用于减少候选量并轻微提 F1。

## 4. 延期实现方案（本轮不执行）
以下内容保留到盲审提交后：

1. 多 seed 复现实验（建议 3 seeds）
- 目标：给出均值±标准差，增强统计可信度。

2. 误差分析补强
- 边界偏移分布（|tp-tg|）
- proposal 数量与 F1 权衡曲线
- 典型成功/失败案例可视化

3. 口径统一最终稿
- 将“模态对比轴”和“组件消融轴”在正文中拆分描述，避免读者误读为同一组横向实验。

---

该文档用途：  
1) 盲审阶段按“冻结结果”直接写作与提交；  
2) 提交后按“延期实现方案”继续扩展，不影响当前版本一致性。
