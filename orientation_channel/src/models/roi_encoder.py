import torch
import torch.nn as nn


class ROIEncoder(nn.Module):
    def __init__(self, backbone: str = 'resnet18', out_dim: int = 512):
        super().__init__()
        try:
            import torchvision
        except Exception as e:
            raise ImportError("torchvision is required for ROIEncoder. Install it via requirements.txt") from e
        if backbone == 'resnet18':
            try:
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            except Exception:
                m = torchvision.models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
            self.feat_dim = 512
        elif backbone == 'convnext_tiny':
            try:
                m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
            except Exception:
                m = torchvision.models.convnext_tiny(weights=None)
            self.backbone = nn.Sequential(*list(m.children())[:-1])
            self.feat_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.proj = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x):
        # x: (B,T,3,H,W) or (B,3,H,W)
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            feat = self.backbone(x).flatten(1)
            feat = self.proj(feat)
            feat = feat.reshape(B, T, -1)
            return feat
        else:
            feat = self.backbone(x).flatten(1)
            feat = self.proj(feat)
            return feat
