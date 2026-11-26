import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import yaml


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def deep_update(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_with_inherit(path: str, _seen: Optional[Set[Path]] = None) -> dict:
    cfg_path = Path(path).resolve()
    if _seen is None:
        _seen = set()
    if cfg_path in _seen:
        raise ValueError(f"Cyclic inherit detected: {cfg_path}")
    _seen.add(cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    inherit = cfg.pop('inherit', None)
    if not inherit:
        return cfg

    inherit_path = Path(inherit)
    if not inherit_path.is_absolute():
        # Resolve only relative to current file directory.
        inherit_path = (cfg_path.parent / inherit_path).resolve()
    if not inherit_path.exists():
        raise FileNotFoundError(f"Inherited config not found: {inherit_path}")

    parent = load_yaml_with_inherit(str(inherit_path), _seen)
    return deep_update(parent, cfg)
