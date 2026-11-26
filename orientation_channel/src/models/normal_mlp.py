import torch
import torch.nn as nn


class NormalMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # x: (B,T,3)
        return self.net(x)
