"""
run_from_features.py
====================
输入：预处理结果（特征文件 / 已有预测 JSONL）
输出：统一 JSON（sign_phonology_v1 schema）

用法:
    python run_from_features.py --input location_channel/data/features --limit 5
    python run_from_features.py --input handshape_channel_data28 --limit 3
    python run_from_features.py --all   # 跑全部
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# 路径设置（确保能 import run_all）
# ============================================================

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_all import run_all


# ============================================================
# event_id 提取（方案 A：正则从 npz 文件名提取短名）
# ============================================================

def collect_event_ids_from_npz(npz_dir: Path) -> List[str]:
    """
    从 .npz 文件名提取 event_id（短名）

    文件名格式: S100_V6326_R_Event_1_RIGHT.npz
    提取结果:   R_Event_1

    规则: 找 [RL]_Event_\d+ 模式
    """
    event_ids = set()
    for f in npz_dir.rglob("*.npz"):
        m = re.search(r"[RL]_Event_\d+", f.stem)
        if m:
            event_ids.add(m.group(0))
    return sorted(event_ids)


def collect_event_ids_from_jsonl(jsonl_dir: Path) -> List[str]:
    """
    从 .jsonl 预测文件里收集 event_id（最可靠，直接用已有的短名）
    """
    seen = set()
    event_ids = []
    for jf in jsonl_dir.rglob("*.jsonl"):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    eid = item.get("event_id") or item.get("sample_id")
                    if eid and eid not in seen:
                        seen.add(eid)
                        event_ids.append(eid)
                except json.JSONDecodeError:
                    continue
    return event_ids


def collect_event_ids_from_test_pred(json_dir: Path) -> List[str]:
    """
    从 test_pred.json 里收集 event_id
    """
    json_files = list(json_dir.rglob("test_pred.json"))
    if not json_files:
        return []
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", []) if isinstance(data, dict) else data
    return [item.get("event_id") for item in items if item.get("event_id")]


def collect_event_ids(input_path: Path) -> List[str]:
    """
    统一入口：按优先级尝试三种方式收集 event_id
    1. .jsonl 预测文件（最可靠）
    2. test_pred.json
    3. .npz 特征文件（用正则提取短名）
    """
    # 优先级 1: jsonl
    eids = collect_event_ids_from_jsonl(input_path)
    if eids:
        return eids

    # 优先级 2: test_pred.json
    eids = collect_event_ids_from_test_pred(input_path)
    if eids:
        return eids

    # 优先级 3: npz（正则提取）
    eids = collect_event_ids_from_npz(input_path)
    if eids:
        return eids

    return []


def collect_requests(input_path: Path) -> List[Dict[str, Any]]:
    """尽量从预处理文件名恢复完整事件上下文，避免 event_id 跨视频冲突。"""
    requests: Dict[tuple, Dict[str, Any]] = {}
    pattern = re.compile(
        r"(?P<signer>S\d+)_(?P<video>V\d+).*?"
        r"(?P<event>[LR]_Event_\d+)",
        re.IGNORECASE,
    )
    for feature_file in input_path.rglob("*.npz"):
        match = pattern.search(feature_file.stem)
        if not match:
            continue
        event_id = match.group("event")
        hand = "left" if event_id.upper().startswith("L_") else "right"
        request = {
            "event_id": event_id,
            "signer_id": match.group("signer"),
            "video_id": match.group("video"),
            "hand": hand,
        }
        key = tuple(request.values())
        requests[key] = request

    if requests:
        return sorted(
            requests.values(),
            key=lambda item: (
                item["signer_id"],
                item["video_id"],
                item["event_id"],
                item["hand"],
            ),
        )
    return [{"event_id": event_id} for event_id in collect_event_ids(input_path)]


# ============================================================
# 核心调度
# ============================================================

def run_on_input(
    input_path: str,
    limit: int = 5,
    split: str = "test",
    exp_dir: str = "smoke_external_no_text",
) -> List[Dict[str, Any]]:
    in_path = Path(input_path)
    if not in_path.exists():
        print(f"❌ 输入路径不存在: {in_path}")
        sys.exit(1)

    # 1️⃣ 收集 event_id
    requests = collect_requests(in_path)
    if not requests:
        print(f"❌ 在 {in_path} 下没找到可用的 event_id")
        print("   请确保目录包含 .jsonl / test_pred.json / .npz 文件")
        sys.exit(1)

    print(f"📂 输入目录: {in_path}")
    print(f"📊 找到 {len(requests)} 个事件请求")
    if limit and limit < len(requests):
        print(f"⚡ 限制: 只跑前 {limit} 个")
        requests = requests[:limit]

    # 2️⃣ 逐个调 run_all
    results = []
    for i, request in enumerate(requests):
        eid = request["event_id"]
        print(f"  [{i+1}/{len(requests)}] 处理 {eid} ...", end=" ")
        r = run_all(split=split, exp_dir=exp_dir, **request)
        results.append(r)
        ch_status = {k: v["status"]["code"] for k, v in r["channels"].items()}
        print(f"✅ {ch_status}")

    return results


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="🧩 输入预处理结果 → 输出统一 JSON"
    )
    parser.add_argument(
        "--input", "-i",
        help="预处理文件/目录路径（含 .npz / .jsonl / .json）",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=5,
        help="限制处理样本数（默认 5，设 0 表示全部）",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--exp_dir", default="smoke_external_no_text")
    parser.add_argument(
        "--all", action="store_true",
        help="跑全部样本（等价于 --limit 0）",
    )
    parser.add_argument(
        "--output", "-o", default="output/batch_result.json",
        help="输出 JSON 路径",
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        print("\n⚠️ 必须指定 --input 路径")
        sys.exit(1)

    limit = 0 if args.all else args.limit

    # 跑
    results = run_on_input(
        input_path=args.input,
        limit=limit,
        split=args.split,
        exp_dir=args.exp_dir,
    )

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总
    total = len(results)
    hs_ok = sum(1 for r in results if r["channels"]["handshape"]["status"]["code"] == "OK")
    loc_ok = sum(1 for r in results if r["channels"]["location"]["status"]["code"] == "OK")
    ori_ok = sum(1 for r in results if r["channels"]["orientation"]["status"]["code"] == "OK")
    mv_ok = sum(1 for r in results if r["channels"].get("movement", {}).get("status", {}).get("code") == "OK")

    print(f"\n{'='*50}")
    print(f"✅ 完成！输出: {out_path}")
    print(f"   总样本: {total}")
    print(f"   Handshape OK:  {hs_ok}/{total}")
    print(f"   Location OK:   {loc_ok}/{total}")
    print(f"   Orientation OK: {ori_ok}/{total}")
    print(f"   Movement OK:    {mv_ok}/{total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()