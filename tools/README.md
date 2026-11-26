# Developer tools

本目录存放诊断、调试和离线可视化工具，不属于生产推理入口。

所有命令请在项目根目录执行：

```bash
python tools/diagnose_path.py
python tools/diagnose_features.py
python tools/diagnose_builder.py
python tools/debug_probs.py
python tools/visualize_result.py --sample_id S142_V6326_R_Event_1_RIGHT
```

正式入口保留在项目根目录：`run_all.py`、`video2json.py`、
`segment_video.py`、`run_from_features.py` 和 `gui_server.py`。
