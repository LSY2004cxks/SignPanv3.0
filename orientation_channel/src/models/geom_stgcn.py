import torch
import torch.nn as nn


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (5, 17),
]


def build_adjacency(num_nodes=21, edges=HAND_EDGES):
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    A = A + torch.eye(num_nodes)
    # normalize
    D = A.sum(1)
    D_inv = torch.diag(1.0 / D)
    return D_inv @ A


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, kernel_size=3):
        super().__init__()
        # Keep adjacency on the same device as module parameters.
        self.register_buffer('A', A.clone())
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                             padding=(kernel_size // 2, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C, T, V)
        res = self.residual(x)
        # graph conv: X * A
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, in_channels=3, out_dim=256, num_nodes=21):
        super().__init__()
        A = build_adjacency(num_nodes)
        self.register_buffer('A', A)
        self.layer1 = STGCNBlock(in_channels, 64, self.A)
        self.layer2 = STGCNBlock(64, 128, self.A)
        self.layer3 = STGCNBlock(128, out_dim, self.A)

    def forward(self, x, mask=None):
        # x: (B,T, V, C)
        x = x.permute(0, 3, 1, 2)  # (B,C,T,V)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 3, 1)  # (B,T,V,C)
        # pool over joints
        x = x.mean(dim=2)  # (B,T,C)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask
        return x
