import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    ensure_dir(str(Path(path).parent))
    exists = Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(logs_dir: str) -> str:
    run_dir = Path(logs_dir) / f"run_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_str: Optional[str] = None) -> torch.device:
    if device_str and device_str.lower() != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_signer_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    xs = [x.strip() for x in s.split(",") if x.strip()]
    return xs if xs else None


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out
