#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


class ROIConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny frame encoder with event-level mean pooling."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm is required for ROI encoder but is not installed.") from e

        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.backbone_dim = int(self.backbone.num_features)
        self.proj = nn.Linear(self.backbone_dim, out_dim)
        self.freeze_backbone_flag = bool(freeze_backbone)

        if self.freeze_backbone_flag:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_stage(self) -> None:
        # ConvNeXt in timm usually has stages[-1]
        if hasattr(self.backbone, "stages"):
            for p in self.backbone.stages[-1].parameters():
                p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
        else:
            # safe fallback: unfreeze full backbone
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(self, roi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            roi: (B,3,T,H,W)
        Returns:
            frame_feats: (B,T,D)
            event_feat:  (B,D)
        """
        b, c, t, h, w = roi.shape
        x = roi.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        if self.freeze_backbone_flag and (not any(p.requires_grad for p in self.backbone.parameters())):
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)

        feat = self.proj(feat).reshape(b, t, -1)
        event_feat = feat.mean(dim=1)
        return feat, event_feat
