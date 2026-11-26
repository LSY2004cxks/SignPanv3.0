# Movement Channel — SignPanv3.0

手控四参数中的运动通道。主标签取金标里的 **全局运动** `movement.global.label.type`。

当前没有训练好的运动分类器。通道按下面顺序给出结果，并在 `metadata.source` 里标明来源：

1. `gold_lookup`：当前 signer / video / hand 能对齐到 `data28` 金标
2. `cross_signer_fallback`：同一 `video_id` 其他 signer 的同手事件投票
3. `geometric_fallback`：用手腕轨迹估计 HOLD / ST / CU / CI
4. 否则 `MODEL_UNAVAILABLE`

## 标签

全局（通道主结果）：

| 标签 | 含义 |
|---|---|
| `NOM` | 无运动 / 未标注 |
| `HOLD` | 保持 |
| `ST` | 直线运动 |
| `CU` | 曲线运动 |
| `CI` | 圆周运动 |

局部运动（`TME` / `TMW` / `TMN` 等）写入 `metadata.local_label`，不作为当前 Top-1。

## 调用

```powershell
python movement_channel\scripts\run_movement.py `
  --signer_id S92 --video_id V6446 --event_id R_Event_1 --hand right
```

统一入口：

```powershell
python run_all.py --sample_id S92_V6446_R_Event_1_RIGHT
```
