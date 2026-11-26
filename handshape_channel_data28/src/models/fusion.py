#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn

from .deformable_attn import DeformableTemporalAttention
from .roi_convnext import ROIConvNeXtEncoder
from .stgcn import STGCNEncoder


class HandshapeNet(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        roi_backbone: str = "convnext_tiny",
        roi_dim: int = 512,
        skel_dim: int = 256,
        fusion_hidden: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        assert mode in {"roi_only", "skel_only", "roi_skel_fusion", "roi_skel_deform_nce"}
        self.mode = mode

        self.roi_encoder = None
        self.skel_encoder = None

        if mode in {"roi_only", "roi_skel_fusion", "roi_skel_deform_nce"}:
            self.roi_encoder = ROIConvNeXtEncoder(
                backbone=roi_backbone,
                out_dim=roi_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
            )

        if mode in {"skel_only", "roi_skel_fusion", "roi_skel_deform_nce"}:
            self.skel_encoder = STGCNEncoder(in_channels=3, hidden_dim=skel_dim)

        if mode == "roi_only":
            self.cls = nn.Linear(roi_dim, num_classes)

        elif mode == "skel_only":
            self.cls = nn.Linear(skel_dim, num_classes)

        elif mode == "roi_skel_fusion":
            self.fuse = nn.Sequential(
                nn.Linear(roi_dim + skel_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.GELU(),
            )
            self.cls = nn.Linear(fusion_hidden, num_classes)

        else:  # roi_skel_deform_nce
            self.skel_to_roi = nn.Linear(skel_dim, roi_dim)
            self.deform_attn = DeformableTemporalAttention(dim=roi_dim, num_heads=4, dropout=dropout)
            self.gate = nn.Sequential(
                nn.Linear(roi_dim * 2, roi_dim),
                nn.GELU(),
                nn.Linear(roi_dim, 1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(roi_dim + roi_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.GELU(),
            )
            self.cls = nn.Linear(fusion_hidden, num_classes)
            self.nce_roi_proj = nn.Sequential(nn.Linear(roi_dim, roi_dim), nn.GELU(), nn.Linear(roi_dim, roi_dim))
            self.nce_skel_proj = nn.Sequential(nn.Linear(roi_dim, roi_dim), nn.GELU(), nn.Linear(roi_dim, roi_dim))

    def freeze_roi_backbone(self) -> None:
        if self.roi_encoder is not None:
            self.roi_encoder.freeze_backbone_flag = True
            self.roi_encoder.freeze_backbone()

    def unfreeze_roi_last_stage(self) -> None:
        if self.roi_encoder is not None:
            self.roi_encoder.freeze_backbone_flag = False
            self.roi_encoder.unfreeze_last_stage()

    def forward(self, batch: dict) -> dict:
        out = {
            "logits": None,
            "event_feat": None,
            "nce_roi": None,
            "nce_skel": None,
        }

        if self.mode == "roi_only":
            _, roi_event = self.roi_encoder(batch["roi"])
            out["event_feat"] = roi_event
            out["logits"] = self.cls(roi_event)
            return out

        if self.mode == "skel_only":
            _, skel_event = self.skel_encoder(batch["skeleton"])
            out["event_feat"] = skel_event
            out["logits"] = self.cls(skel_event)
            return out

        if self.mode == "roi_skel_fusion":
            _, roi_event = self.roi_encoder(batch["roi"])
            _, skel_event = self.skel_encoder(batch["skeleton"])
            feat = self.fuse(torch.cat([roi_event, skel_event], dim=-1))
            out["event_feat"] = feat
            out["logits"] = self.cls(feat)
            return out

        # roi_skel_deform_nce
        roi_frame, roi_event = self.roi_encoder(batch["roi"])  # (B,T,Cr), (B,Cr)
        skel_frame, skel_event = self.skel_encoder(batch["skeleton"])  # (B,T,Cs), (B,Cs)
        skel_frame_proj = self.skel_to_roi(skel_frame)
        skel_event_proj = self.skel_to_roi(skel_event)

        aligned = self.deform_attn(roi_seq=roi_frame, skel_seq=skel_frame_proj)
        g = self.gate(torch.cat([aligned, roi_frame], dim=-1))
        fused_frame = g * aligned + (1.0 - g) * roi_frame
        fused_event = fused_frame.mean(dim=1)

        feat = self.fuse(torch.cat([fused_event, skel_event_proj], dim=-1))
        logits = self.cls(feat)

        out["event_feat"] = feat
        out["logits"] = logits
        out["nce_roi"] = self.nce_roi_proj(roi_event)
        out["nce_skel"] = self.nce_skel_proj(skel_event_proj)
        return out
