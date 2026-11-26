#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path).resolve()
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {p}")
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(runs_root: str, exp_name: str) -> Path:
    root = ensure_dir(Path(runs_root).resolve())
    run = root / f"{exp_name}_{now_tag()}"
    ensure_dir(run)
    return run


def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def try_save_requirements_lock(run_dir: Path) -> None:
    out = run_dir / "requirements_lock.txt"
    try:
        cmd = ["python", "-m", "pip", "freeze"]
        txt = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        out.write_text(txt, encoding="utf-8")
    except Exception as e:
        out.write_text(f"failed to export pip freeze: {e}\n", encoding="utf-8")


def try_save_git_hash(run_dir: Path) -> None:
    out = run_dir / "git_hash.txt"
    try:
        txt = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
        out.write_text(txt + "\n", encoding="utf-8")
    except Exception as e:
        out.write_text(f"git hash unavailable: {e}\n", encoding="utf-8")


def probs_topk(
    logits: torch.Tensor,
    k: int = 3,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = torch.softmax(logits / float(temperature), dim=-1)
    top = torch.topk(p, k=min(k, p.shape[-1]), dim=-1)
    return top.indices.detach().cpu().numpy(), top.values.detach().cpu().numpy()


def append_train_log(run_dir: Path, msg: str) -> None:
    p = run_dir / "train.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default
