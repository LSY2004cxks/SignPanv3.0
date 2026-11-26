#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_hop_distance(num_node: int, edge, max_hop: int = 1):
    a = np.zeros((num_node, num_node))
    for i, j in edge:
        a[i, j] = 1
        a[j, i] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(a, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(a):
    dl = np.sum(a, 0)
    num_node = a.shape[0]
    dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)
    return np.dot(a, dn)


class Graph:
    def __init__(self, max_hop: int = 1):
        self.num_node = 21
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
        ]
        self.edge = self.self_link + self.neighbor
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.A = self._adjacency(max_hop=max_hop)

    def _adjacency(self, max_hop: int = 1):
        valid_hop = range(0, max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        norm = normalize_digraph(adjacency)
        a = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            a[i][self.hop_dis == hop] = norm[self.hop_dis == hop]
        return a.astype(np.float32)


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, a: torch.Tensor, temporal_kernel: int = 5, stride: int = 1) -> None:
        super().__init__()
        self.a = nn.Parameter(a.clone(), requires_grad=True)
        ks = a.shape[0]
        pad = (temporal_kernel - 1) // 2

        self.conv = nn.Conv2d(in_c, out_c * ks, kernel_size=(temporal_kernel, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        if in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c),
            )

        self.ks = ks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,V)
        res = self.residual(x)
        y = self.conv(x)
        b, kc, t, v = y.shape
        y = y.view(b, self.ks, kc // self.ks, t, v)
        y = torch.einsum("bkctv,kvw->bctw", y, self.a)
        y = self.bn(y)
        y = y + res
        return self.relu(y)


class STGCNEncoder(nn.Module):
    """ST-GCN encoder for hand skeleton sequences."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        graph = Graph(max_hop=1)
        a = torch.tensor(graph.A, dtype=torch.float32)

        self.input_proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.layers = nn.Sequential(
            STGCNBlock(64, 64, a),
            STGCNBlock(64, 128, a),
            STGCNBlock(128, hidden_dim, a),
        )

    def forward(self, skel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            skel: (B,T,V,3) or (B,T,D)
        Returns:
            frame_feat: (B,T,C)
            event_feat: (B,C)
        """
        if skel.ndim == 3:
            b, t, d = skel.shape
            if d % 3 != 0:
                raise ValueError(f"Unexpected skeleton dim: {d}")
            v = d // 3
            skel = skel.view(b, t, v, 3)

        b, t, v, c = skel.shape
        if c < 3:
            raise ValueError(f"Skeleton last dim must be >=3, got {c}")
        skel = skel[..., :3]

        # pad/truncate V to 21
        if v < 21:
            pad = torch.zeros((b, t, 21 - v, 3), dtype=skel.dtype, device=skel.device)
            skel = torch.cat([skel, pad], dim=2)
        elif v > 21:
            skel = skel[:, :, :21, :]

        x = skel.permute(0, 3, 1, 2).contiguous()  # (B,3,T,V)
        x = self.input_proj(x)
        x = self.layers(x)  # (B,C,T,V)

        frame_feat = x.mean(dim=-1).permute(0, 2, 1).contiguous()  # (B,T,C)
        event_feat = frame_feat.mean(dim=1)
        return frame_feat, event_feat
