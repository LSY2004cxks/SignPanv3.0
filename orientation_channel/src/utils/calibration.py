import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def temperature_scale(logits: torch.Tensor, labels: torch.Tensor, init_t: float = 1.0, max_iter: int = 50):
    device = logits.device
    temperature = nn.Parameter(torch.ones(1, device=device) * init_t)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


def apply_temperature(logits: torch.Tensor, temperature: float):
    return logits / temperature


def save_temperature(path: str, temperature: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'temperature': temperature}, f, indent=2)


def load_temperature(path: str) -> float:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['temperature']
