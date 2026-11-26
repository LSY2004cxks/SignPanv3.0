from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from .pooling import pool
from src.kpts.geometry import build_anchors


class ProtoDist(nn.Module):
    def __init__(self, codes: List[str], joint_names: List[str], anchor_defs: Dict[str, Any],
                 hand_rep: str = 'center', tau: float = 0.5, pool_mode: str = 'logsumexp'):
        super().__init__()
        self.codes = codes
        self.joint_names = joint_names
        self.anchor_defs = anchor_defs
        self.hand_rep = hand_rep
        self.pool_mode = pool_mode
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        self.bias = nn.Parameter(torch.zeros(len(codes)))

    def forward(self, hand_xy: torch.Tensor, pose_xy: torch.Tensor, mask: torch.Tensor) -> dict:
        # hand_xy: (B,T,2)
        # pose_xy: (B,T,J,2)
        B, T, J, _ = pose_xy.shape
        # anchors per batch
        # build anchors in numpy for each batch item
        anchors_list = []
        for b in range(B):
            anchors = build_anchors(pose_xy[b].detach().cpu().numpy(), self.joint_names, self.anchor_defs)
            # order by codes
            anchors_np = np.stack([anchors[c] for c in self.codes], axis=1)
            anchors_list.append(torch.from_numpy(anchors_np))
        anchors_t = torch.stack(anchors_list, dim=0).to(pose_xy.device)  # (B,T,C,2)

        h = hand_xy
        diff = anchors_t - h.unsqueeze(2)
        dist2 = (diff * diff).sum(dim=-1)  # (B,T,C)
        tau = torch.exp(self.log_tau).clamp_min(1e-4)
        logits_frame = -dist2 / tau + self.bias.view(1, 1, -1)
        logits_clip = pool(logits_frame, mask, mode=self.pool_mode)
        return {'logits_frame': logits_frame, 'logits_clip': logits_clip}
