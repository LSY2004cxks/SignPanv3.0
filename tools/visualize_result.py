from __future__ import annotations

import sys
import subprocess
import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

# =======================
# ✅ 唯一可靠的方式：subprocess 调用 run_all.py
# =======================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_result(
    sample_id: Optional[str] = None,
    split: str = "test",
    exp_dir: str = "smoke_external_no_text",
) -> Dict[str, Any]:
    """
    通过子进程调用 run_all.py
    彻底规避 import / sys.path / IDE 的一切坑
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_all.py"),
        "--split", split,
    ]
    if sample_id:
        cmd.extend(["--sample_id", sample_id])
    if exp_dir:
        cmd.extend(["--exp_dir", exp_dir])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
    )

    if proc.returncode != 0:
        raise RuntimeError(f"run_all.py 执行失败:\n{proc.stderr}")

    return json.loads(proc.stdout)


# =======================
# ✅ 置信度条
# =======================

def confidence_bar(value: float) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    if pct >= 60:
        color = "#4CAF50"
        level = "高"
    elif pct >= 30:
        color = "#FF9800"
        level = "中"
    else:
        color = "#F44336"
        level = "低"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'<span class="conf-label">{level}({pct:.1f}%)</span>'
        f'</div>'
    )


# =======================
# ✅ 工具函数
# =======================

def _topk_chips(topk):
    chips = ""
    for i, t in enumerate(topk):
        cls = "chip-first" if i == 0 else "chip"
        chips += f'<span class="{cls}">#{i+1} {t.get("label")} ({t.get("prob"):.3f})</span>'
    return chips


def _format_span(span):
    if isinstance(span, dict):
        return f"帧 {span.get('start_frame')} ~ {span.get('end_frame')}"
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return f"帧 {span[0]} ~ {span[1]}"
    return str(span) if span else "—"


def _safe(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _status_badge(data) -> str:
    if data and _safe(data, "status", "code") == "OK":
        return '<span class="badge badge-ok">● OK</span>'
    return '<span class="badge badge-na">● 不可用</span>'


# =======================
# ✅ 通道卡片（三列横向）
# =======================

def render_location(data):
    badge = _status_badge(data)
    if not data or _safe(data, "status", "code") != "OK":
        return f'''
<div class="card card-loc">
  <div class="card-head">
    <span class="icon">📍</span><span class="title">Location</span>
    {badge}
  </div>
  <div class="card-body muted">当前样本无 Location 预测结果</div>
</div>'''
    p = data["prediction"]
    m = data["metadata"]
    return f'''
<div class="card card-loc">
  <div class="card-head">
    <span class="icon">📍</span><span class="title">Location</span>
    {badge}
  </div>
  <div class="card-body">
    <div class="pred">{p.get("label")}</div>
    <div class="kv">GT：{m.get("gt","—")}</div>
    <div class="kv">Event：{data.get("event_id")}</div>
    <div class="kv">置信度</div>
    {confidence_bar(p.get("conf",0))}
  </div>
</div>'''


def render_handshape(data):
    badge = _status_badge(data)
    if not data or _safe(data, "status", "code") != "OK":
        return f'''
<div class="card card-hs">
  <div class="card-head">
    <span class="icon">✋</span><span class="title">Handshape</span>
    {badge}
  </div>
  <div class="card-body muted">当前样本无 Handshape 预测结果</div>
</div>'''
    p = data["prediction"]
    m = data["metadata"]
    return f'''
<div class="card card-hs">
  <div class="card-head">
    <span class="icon">✋</span><span class="title">Handshape</span>
    {badge}
  </div>
  <div class="card-body">
    <div class="pred">{p.get("label")}</div>
    <div class="kv">GT：{m.get("gt","—")}</div>
    <div class="kv">Entropy：{m.get("entropy","—")}</div>
    <div class="kv">Span：{_format_span(m.get("span"))}</div>
    <div class="kv">置信度</div>
    {confidence_bar(p.get("conf",0))}
  </div>
</div>'''


def render_orientation(data):
    badge = _status_badge(data)
    if not data or _safe(data, "status", "code") != "OK":
        return f'''
<div class="card card-ori">
  <div class="card-head">
    <span class="icon">🤚</span><span class="title">Orientation</span>
    {badge}
  </div>
  <div class="card-body muted">当前样本无 Orientation 预测结果</div>
</div>'''
    p = data["prediction"]
    m = data["metadata"]
    return f'''
<div class="card card-ori">
  <div class="card-head">
    <span class="icon">🤚</span><span class="title">Orientation</span>
    {badge}
  </div>
  <div class="card-body">
    <div class="pred">{p.get("label")}</div>
    <div class="kv">Span：{_format_span(m.get("span"))}</div>
    <div class="kv">HITL：{m.get("hitl_gate","—")}</div>
    <div class="kv">置信度</div>
    {confidence_bar(p.get("conf",0))}
  </div>
</div>'''


def render_movement(data):
    badge = _status_badge(data)
    if not data or _safe(data, "status", "code") != "OK":
        return f'''
<div class="card card-mv">
  <div class="card-head">
    <span class="icon">↔️</span><span class="title">Movement</span>
    {badge}
  </div>
  <div class="card-body muted">当前样本无 Movement 预测结果</div>
</div>'''
    p = data["prediction"]
    m = data["metadata"]
    return f'''
<div class="card card-mv">
  <div class="card-head">
    <span class="icon">↔️</span><span class="title">Movement</span>
    {badge}
  </div>
  <div class="card-body">
    <div class="pred">{p.get("label")}</div>
    <div class="kv">来源：{m.get("source","—")}</div>
    <div class="kv">局部：{m.get("local_label","—")}</div>
    <div class="kv">置信度</div>
    {confidence_bar(p.get("conf",0))}
  </div>
</div>'''


# =======================
# HTML 模板（2x2 Grid）
# =======================

HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>手语音系标注系统 – {sid}</title>
<style>
:root {{
  --loc: #2196F3;
  --hs: #FF9800;
  --ori: #9C27B0;
  --mv: #7C3AED;
  --bg: #eef2f7;
  --card: #ffffff;
  --muted: #94a3b8;
}}
body {{
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
  background: linear-gradient(135deg,#eef2f7,#e3e8f0);
  padding: 28px;
  margin: 0;
}}
.container {{
  max-width: 1180px;
  margin: auto;
}}
.header {{
  margin-bottom: 22px;
}}
.header h1 {{
  margin: 0 0 6px;
  font-size: 22px;
}}
.header span {{
  color: #64748b;
  font-size: 13px;
}}

/* 三列 */
.grid {{
  display: grid;
  grid-template-columns: repeat(2,1fr);
  gap: 22px;
}}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}

/* 卡片 */
.card {{
  background: var(--card);
  border-radius: 16px;
  box-shadow: 0 6px 22px rgba(0,0,0,.07);
  overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;
}}
.card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 12px 28px rgba(0,0,0,.1);
}}
.card-head {{
  display: flex;
  align-items: center;
  padding: 14px 18px;
  border-top: 5px solid transparent;
}}
.card-loc .card-head {{ border-top-color: var(--loc); }}
.card-hs .card-head {{ border-top-color: var(--hs); }}
.card-ori .card-head {{ border-top-color: var(--ori); }}
.card-mv .card-head {{ border-top-color: var(--mv); }}
.icon {{ font-size: 20px; margin-right: 8px; }}
.title {{ flex: 1; font-weight: 600; }}
.badge {{
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
}}
.badge-ok {{ background: #dcfce7; color: #15803d; }}
.badge-na {{ background: #fee2e2; color: #b91c1c; }}

.card-body {{
  padding: 16px 18px 18px;
  font-size: 13px;
  color: #334155;
}}
.pred {{
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 10px;
}}
.kv {{
  margin: 6px 0;
  color: #475569;
}}
.muted {{
  color: var(--muted);
  font-style: italic;
}}

/* 置信度条 */
.conf-bar-wrap {{
  height: 14px;
  background: #e2e8f0;
  border-radius: 7px;
  position: relative;
  margin-top: 4px;
}}
.conf-bar-fill {{
  height: 100%;
  border-radius: 7px;
}}
.conf-label {{
  position: absolute;
  right: 6px;
  top: 0;
  font-size: 11px;
  line-height: 14px;
  color: #475569;
}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🧩 手语音系标注系统</h1>
    <span>样本：<b>{sid}</b>｜Schema：sign_phonology_v1</span>
  </div>
  <div class="grid">
    {hs}
    {mv}
    {loc}
    {ori}
  </div>
</div>
</body>
</html>'''

# =======================
# ✅ CLI
# =======================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample_id")
    p.add_argument("--split", default="test")
    p.add_argument("--exp_dir", default="smoke_external_no_text")
    args = p.parse_args()

    result = get_result(args.sample_id, args.split, args.exp_dir)

    out = Path("output")
    out.mkdir(exist_ok=True)
    sid = args.sample_id or "overview"
    html_path = out / f"viz_{sid}.html"

    html = HTML.format(
        sid=sid,
        loc=render_location(result["channels"]["location"]),
        hs=render_handshape(result["channels"]["handshape"]),
        mv=render_movement(result["channels"].get("movement") or {}),
        ori=render_orientation(result["channels"]["orientation"]),
    )

    html_path.write_text(html, encoding="utf-8")
    print("✅", html_path.resolve())

    try:
        webbrowser.open(html_path.resolve().as_uri())
    except Exception:
        pass