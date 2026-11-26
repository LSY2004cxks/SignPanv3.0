from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return [row for row in r]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge b into a (copy)."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_with_base(path: Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    base = cfg.get('__base__')
    if base:
        base_cfg = load_yaml_with_base(Path(base))
        cfg = merge_dict(base_cfg, {k: v for k, v in cfg.items() if k != '__base__'})
    return cfg
