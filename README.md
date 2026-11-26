# SignPanv3.0

这是独立可运行的项目版本。网页标注、命令行预测、左右手切分、四个音系通道、人工修订和 JSON / CSV / EAF 导出都在。训练集视频没有放进来，`data28`、`data28v2.0` 以及各通道里由这些视频算出的特征缓存都不在本目录。标注边界索引 `data28/annotations/mhsl_events.jsonl` 还在，所以“仅标注边界”仍能按原语料文件名查找时间段；视频本身需要使用时另外提供。

推理只用当前这一套权重：BoundaryTCN `boundary_prefill_B_final/logs/run_20260228_210024/best.pt`，手形 `runs_schemeA/handshape_data28_schemeA_fold1_k24_deform_nce_20260303_231251/best.ckpt`，位置 `location_channel/runs/loc_mlp_mil_pxv3/best.pt`，朝向 `orientation_channel/outputs_ablation/modal6_20260305_201215/exp_modal_roi/checkpoints/best.pt`。运动仍按手腕轨迹计算，没有单独的视频模型。

手语音系学参数识别与自动标注系统。系统接收手语视频或预处理结果，完成事件切分，
统一调度 Location、Handshape、Orientation、Movement 四个音系通道，并输出
`sign_phonology_v1` JSON。Web 界面支持结果浏览、人工修订、历史标注分页与删除，
以及规范 v3.0 JSON、CSV 和 ELAN EAF 导出。中文界面为 `/`，英文界面为 `/en`。

> 当前版本是可运行的研究原型。BoundaryTCN 切分已接入；部分通道在缺少当前
> 样本模型结果时会使用明确标记的回退策略。
> 不要把回退结果描述为当前视频的神经网络推理结果。

## 功能

- BoundaryTCN 左右手事件切分，支持模型、标注和自动回退三种模式
- Location、Handshape、Orientation、Movement 四通道统一调用
- 单事件、批量、视频和预处理 JSON 输入
- 统一 Top-1、Top-3、置信度、状态码及元数据
- Flask 可视化标注界面
- 人工标签保存及相同事件再次上传时自动恢复
- 规范 v3.0 JSON、CSV、ELAN EAF 导出
- 历史标注按每页 8 条分页浏览，可打开或删除单条记录

## 处理流程

```text
视频
  └─ BoundaryTCN 左/右手事件切分
       └─ MediaPipe / OpenCV 特征提取
            ├─ Location
            ├─ Handshape
            ├─ Orientation
            └─ Movement
                 └─ run_all.py 统一为 sign_phonology_v1
                      └─ GUI 浏览、修订、保存、导出
```

## 工程结构

```text
SignPanv3.0/
├─ run_all.py                    # 四通道统一入口
├─ video2json.py                 # 视频预处理与端到端调度
├─ segment_video.py              # BoundaryTCN 单视频切分
├─ run_from_features.py          # 预处理目录批量入口
├─ gui_server.py                 # Flask 后端
├─ gui.html / gui_en.html        # 中文与英文标注界面
├─ AUTHORS.md                    # 作者名单（发布前只改这里）
├─ COPYRIGHT.md                  # 版权声明
├─ location_channel/             # 位置通道
├─ handshape_channel_data28/     # 手形通道
├─ orientation_channel/          # 手掌朝向通道
├─ movement_channel/             # 运动通道
├─ boundary_prefill_B_final/     # 事件边界模型
├─ docs/
│  └─ phonology_schema.md        # 统一 JSON 契约
├─ tests/                        # 契约、GUI、切分测试
├─ tools/                        # 诊断、调试、离线可视化
├─ archive/                      # 历史备份与遗留日志
├─ data28/                       # 本地数据集（需确认发布授权）
└─ output/                       # 运行结果、上传文件和人工标注
```

四个通道内部的训练与评估方法分别见：

- [Location](location_channel/README.md)
- [Handshape](handshape_channel_data28/README.md)
- [Orientation](orientation_channel/README.md)
- [Movement](movement_channel/README.md)
- [BoundaryTCN](boundary_prefill_B_final/README.md)

## 环境

推荐 Python 3.10。所有命令均从项目根目录执行。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install flask numpy scipy opencv-python mediapipe torch
```

上述依赖可运行统一编排、视频预处理、切分和 GUI。训练三个通道还需要各通道
README 中列出的 OpenMMLab、TorchVision 或 GPU 环境依赖。

## 快速开始

### 启动标注界面

```powershell
.\.venv\Scripts\python.exe gui_server.py
```

浏览器访问 <http://127.0.0.1:5000>，英文界面为 <http://127.0.0.1:5000/en>。上传视频后可以：

1. 选择 BoundaryTCN 或标注边界；
2. 查看每个事件的手形、位置、朝向和运动；
3. 修改标签或点击 Top-3 候选；
4. 保存人工修订；
5. 在历史标注中分页打开已保存结果，或删除某一条；
6. 导出规范 v3.0 JSON、CSV 或 EAF。

关于页为 <http://127.0.0.1:5000/about>。

人工修订保存在 `output/annotations/`。持久化索引使用
`signer_id + video_id + event_id + hand + channel`，不会修改模型权重。

### 视频转统一 JSON

```powershell
python run_all.py `
  --video "path\to\S43_slr-4-example-43-6332.mp4" `
  --output output\result.json
```

默认切分模式为 `auto`：优先调用 BoundaryTCN，未切出事件或推理失败时使用
已有标注边界。也可以直接使用预处理入口：

```powershell
python video2json.py `
  --video "path\to\video.mp4" `
  --segment_mode auto `
  --run_all
```

切分模式：

- `auto`：BoundaryTCN 优先，失败时回退已有标注
- `model`：只使用 BoundaryTCN
- `annotation`：只使用 `mhsl_events.jsonl` 中的事件边界

### 查询单个已知事件

```powershell
python run_all.py `
  --sample_id S142_V6326_R_Event_1_RIGHT `
  --output output\single_event.json
```

只使用短 `R_Event_1` 可能匹配多个视频。建议提供完整 `sample_id`，或同时提供
`signer_id`、`video_id`、`event_id` 和 `hand`。

### 输入预处理结果

```powershell
python run_all.py `
  --preprocessed path\to\channel_results.json `
  --output output\unified.json
```

输入文件应包含事件上下文和四个通道的原始结果：

```json
{
  "context": {
    "signer_id": "S142",
    "video_id": "V6326",
    "event_id": "R_Event_1",
    "hand": "right"
  },
  "channel_results": {
    "location": {"items": []},
    "handshape": {"items": []},
    "orientation": {"items": []},
    "movement": {"items": []}
  }
}
```

完整输出字段见 [统一 JSON 规范](docs/phonology_schema.md)。

## 输出

默认运行目录为 `output/`：

- `predictions.json`：视频的统一预测结果
- `annotations/<task_id>.json`：人工标注会话
- `annotations/_corrections.json`：跨任务人工修订索引
- `uploads/<task_id>/`：GUI 上传的视频
- `features/`：视频预处理产生的中间特征

通道状态包括 `OK`、`SAMPLE_NOT_FOUND`、`MODEL_UNAVAILABLE` 和
`AMBIGUOUS_SAMPLE`。调用方不应只检查 `prediction.label`，还应检查
`status.code` 和 `metadata.source`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

当前测试覆盖：

- 跨通道事件精确对齐
- 歧义事件拒绝
- 模型不可用状态
- GUI 保存、历史分页、删除和导出
- 人工修订持久化
- BoundaryTCN 事件结构转换

## 开发工具

非生产脚本已统一放入 `tools/`：

```powershell
python tools\diagnose_path.py
python tools\diagnose_features.py
python tools\diagnose_builder.py
python tools\debug_probs.py
python tools\visualize_result.py --sample_id S142_V6326_R_Event_1_RIGHT
```

详见 [tools/README.md](tools/README.md)。

## 作者与版权

作者名单只写在 [AUTHORS.md](AUTHORS.md)。`Supervisor:` / `Authors:` 和 `导师：` / `作者：` 两种写法都会被关于页读取。当前内容为：

> Supervisor: Dengfeng Yao
> Authors: Yuan Zhao, Bowen Ye, Peng Jin, Siyuan Jing, Ziwen Meng

界面“关于”页读取的是这个文件，不需要再改 `gui.html`。版权声明见 [COPYRIGHT.md](COPYRIGHT.md)。未写明开源许可证前，程序代码保留全部权利；语料、视频和模型权重需要单独授权。

## 当前限制

1. **模型资产尚不完整。** BoundaryTCN checkpoint 可用；Location 的
   `loc_smoke/best.pt` 仅训练一个 epoch，并且尚未作为正式在线模型接入。
   Handshape、Orientation 的正式 checkpoint 需要从训练环境补充。
2. **回退结果不是当前视频的完整模型推理。** Location、Handshape 在当前
   signer 无预测时可能使用 `cross_signer_fallback`；Orientation 在缺少模型
   结果时可能使用 `geometric_fallback`；Movement 在能对齐语料时使用
   `gold_lookup`，否则使用跨 signer 投票或手腕轨迹 `geometric_fallback`。
   GUI 会明确提示人工复核。
3. **标注切分依赖数据。** `annotation` 模式默认读取
   `data28/annotations/mhsl_events.jsonl`。
4. **新视频文件名不限。** 原语料的 `...-signer-video` 命名仍会保留其
   signer/video 身份；其他文件名会自动生成稳定的合成身份。仅使用
   `annotation` 切分时，视频仍需能够匹配标注库；普通新视频应使用默认的
   `auto` 或 `model`（BoundaryTCN）切分。
5. **数据发布需授权。** `data28/`、视频、标注和模型权重在公开前必须确认许可，
   不应直接推送到公共仓库。

## 后续计划

- 将三个通道 checkpoint 接入当前事件在线推理
- 为 Movement 训练事件级分类器，替换金标查询与几何回退
- 增加 Non-manual 通道
- 补充统一安装脚本和最小示例数据
- 如需开源，在 COPYRIGHT.md 末尾写明许可证
- 完成数据脱敏、模型卡与发布检查

