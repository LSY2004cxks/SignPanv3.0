from typing import List

import torch
from torch import nn


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoundaryTCN(nn.Module):
    """
    Stage-A model:
    - TCN backbone
    - 1x1 conv heads for start/end/inside logits
    """

    def __init__(
        self,
        input_dim: int,
        channels: int = 128,
        layers: int = 4,
        dilations: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        fusion_mode: str = "early",
        pose_dim: int = 23,
        flow_dim: int = 6,
        fusion_dim: int = 64,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).lower()
        self.pose_dim = int(pose_dim)
        self.flow_dim = int(flow_dim)
        self.fusion_dim = int(fusion_dim)
        self.gate_hidden = int(gate_hidden)

        if self.fusion_mode not in {"early", "gated"}:
            raise ValueError(f"unknown fusion_mode={fusion_mode}, expected early|gated")

        if self.fusion_mode == "gated":
            if self.pose_dim <= 0 or self.flow_dim <= 0:
                raise ValueError("gated fusion requires pose_dim>0 and flow_dim>0")
            if int(input_dim) < self.pose_dim + self.flow_dim:
                raise ValueError(
                    f"input_dim={input_dim} is smaller than pose_dim+flow_dim={self.pose_dim + self.flow_dim}"
                )
            self.pose_proj = nn.Linear(self.pose_dim, self.fusion_dim)
            self.flow_proj = nn.Linear(self.flow_dim, self.fusion_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.pose_dim + self.flow_dim, self.gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gate_hidden, self.fusion_dim),
            )
            tcn_in_dim = self.fusion_dim
        else:
            tcn_in_dim = int(input_dim)

        if dilations is None:
            dilations = [1, 2, 4, 8]
        if len(dilations) < layers:
            dilations = (dilations + [dilations[-1]] * layers)[:layers]
        else:
            dilations = dilations[:layers]

        self.in_proj = nn.Conv1d(tcn_in_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualDilatedBlock(channels, kernel_size, d, dropout) for d in dilations]
        )
        self.head = nn.Conv1d(channels, 3, kernel_size=1)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode != "gated":
            return x
        pose = x[:, :, : self.pose_dim]
        flow = x[:, :, self.pose_dim : self.pose_dim + self.flow_dim]
        pose_h = self.pose_proj(pose)
        flow_h = self.flow_proj(flow)
        gate_in = torch.cat([pose, flow], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))
        return gate * pose_h + (1.0 - gate) * flow_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self._fuse(x)
        h = x.transpose(1, 2)  # (B, F, T)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h).transpose(1, 2).contiguous()  # (B, T, 3)
        return out
