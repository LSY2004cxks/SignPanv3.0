from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter

from .io import ensure_dir


class JsonlLogger:
    def __init__(self, path: Path):
        ensure_dir(path.parent)
        self.path = path

    def log(self, data: Dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


class TBLogger:
    def __init__(self, log_dir: Path):
        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()
