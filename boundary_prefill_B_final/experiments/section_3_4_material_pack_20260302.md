# 3.4 小节素材包（可直接用于写作）

## 3.4.1 主结果与阈值权衡

- 图 3-2a（Val Mean F1@7 vs tau）  
  `outputs/dataset_stats/figure_3_2a_val_mean_f1_w7_vs_tau.png`  
  `outputs/dataset_stats/figure_3_2a_val_mean_f1_w7_vs_tau.pdf`

- 图 3-2b（Val PR 曲线）  
  `outputs/dataset_stats/figure_3_2b_val_pr_curve.png`  
  `outputs/dataset_stats/figure_3_2b_val_pr_curve.pdf`

- 阈值扫描原始数据  
  `outputs/dataset_stats/figure_3_2_val_tau_scan_metrics.csv`

- 表 3-5（tau 敏感性）  
  `outputs/dataset_stats/table_3_5_main_result_tau_sensitivity.md`

## 3.4.2 消融实验与模态贡献

- 表 3-6（模态消融：Pose/Flow/Fusion）  
  `outputs/dataset_stats/table_3_6_modal_ablation.md`

- 组件设置对照表（hard/soft、inside on/off、w_nms）  
  `outputs/dataset_stats/table_3_6_component_ablation.md`

说明：`w_nms=2` 与 `w_nms=7` 在当前配置下结果接近，原因是 `peak_radius=7` 已经抑制了大部分近邻重复峰。

## 3.4.3 误差分析与典型案例

- 图 3-3a（成功案例）  
  `outputs/dataset_stats/figure_3_3a_success_timeline.png`

- 图 3-3b（连发型错误）  
  `outputs/dataset_stats/figure_3_3b_burst_timeline.png`

- 图 3-3c（漏检型错误）  
  `outputs/dataset_stats/figure_3_3c_miss_timeline.png`

- 图 3-3d（错配型错误）  
  `outputs/dataset_stats/figure_3_3d_mismatch_timeline.png`

- 案例选择与元信息  
  `outputs/dataset_stats/figure_3_3_case_selection.json`  
  `outputs/dataset_stats/figure_3_3_manifest.json`

## 引用建议（写作时）

1. 在 3.4.1 强调阈值权衡：`tau*` 与 `tau*±0.1` 对 F1 与候选数的 trade-off。  
2. 在 3.4.2 明确两条主结论：  
   - 模态层面：Fusion(gated) 帧级最优；  
   - 结构层面：soft boundary 提升显著，validity filter 主要降低候选负担。  
3. 在 3.4.3 用四类案例解释错误来源与可编辑性影响（连发/漏检/错配/成功）。
