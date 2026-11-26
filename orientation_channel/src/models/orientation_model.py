import torch
import torch.nn as nn
import torch.nn.functional as F

from .roi_encoder import ROIEncoder
from .geom_stgcn import STGCN
from .geom_transformer import SkeletonTransformer
from .normal_mlp import NormalMLP


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x, mask=None):
        # x: (B,T,D), mask: (B,T)
        B, T, D = x.shape
        q = self.query.expand(B, 1, D)
        # attention score
        scores = torch.matmul(q, x.transpose(1,2)).squeeze(1) / (D ** 0.5)  # (B,T)
        if mask is not None:
            valid = mask > 0
            # Avoid all-masked rows -> NaN softmax.
            no_valid = ~valid.any(dim=1)
            if no_valid.any():
                valid = valid.clone()
                valid[no_valid, 0] = True
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~valid, neg)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled


class OrientationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        roi_backbone: str = 'resnet18',
        roi_feat_dim: int = 512,
        geom_encoder_type: str = 'stgcn',
        geom_dim: int = 256,
        normal_dim: int = 128,
        fused_dim: int = 256,
        cross_attn_heads: int = 4,
        use_roi: bool = True,
        use_geom: bool = True,
        use_normal: bool = True,
        use_text: bool = True,
        text_prototypes: torch.Tensor = None,
        use_text_as_kv: bool = False,
    ):
        super().__init__()
        self.use_roi = use_roi
        self.use_geom = use_geom
        self.use_normal = use_normal
        self.use_text = use_text and text_prototypes is not None
        self.use_text_as_kv = use_text_as_kv

        # encoders
        if self.use_roi:
            if roi_backbone is None or roi_backbone == 'precomputed':
                self.roi_encoder = None
            else:
                self.roi_encoder = ROIEncoder(backbone=roi_backbone, out_dim=roi_feat_dim)
            self.roi_proj = nn.Linear(roi_feat_dim, fused_dim)
            self.register_buffer('roi_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
            self.register_buffer('roi_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        if self.use_geom:
            if geom_encoder_type == 'stgcn':
                self.geom_encoder = STGCN(in_channels=3, out_dim=geom_dim)
            elif geom_encoder_type == 'transformer':
                self.geom_encoder = SkeletonTransformer(in_channels=3, d_model=geom_dim)
            else:
                raise ValueError(f"Unknown geom_encoder_type: {geom_encoder_type}")
            self.geom_proj = nn.Linear(geom_dim, fused_dim)

        if self.use_normal:
            self.normal_mlp = NormalMLP(in_dim=3, out_dim=normal_dim)
            self.normal_proj = nn.Linear(normal_dim, fused_dim)

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(fused_dim, cross_attn_heads, batch_first=True)
        self.attn_pool = AttentionPooling(fused_dim)

        # prototype head
        if self.use_text:
            self.register_buffer('text_prototypes', F.normalize(text_prototypes, dim=-1))
            self.proj_head = nn.Linear(fused_dim, text_prototypes.shape[-1])
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
            if self.use_text_as_kv:
                self.text_kv_proj = nn.Linear(text_prototypes.shape[-1], fused_dim)
        else:
            self.cls = nn.Linear(fused_dim, num_classes)

    def forward(self, batch):
        # batch contains roi_feat (B,T,D), landmarks (B,T,21,3), palm_normal (B,T,3), mask (B,T)
        mask = batch.get('mask', None)

        # branch encodings
        v = None
        if self.use_roi:
            if 'roi_img' in batch:
                if self.roi_encoder is None:
                    raise ValueError('roi_img provided but roi_encoder is disabled (roi_backbone=None/precomputed)')
                x = batch['roi_img']  # (B,T,3,H,W)
                if x.max() > 1.5:
                    x = x / 255.0
                x = (x - self.roi_mean) / self.roi_std
                v = self.roi_encoder(x)
            elif 'roi_feat' in batch:
                v = batch['roi_feat']  # (B,T,Dv)
            else:
                raise ValueError('ROI input is required (roi_img or roi_feat)')
            v = self.roi_proj(v)

        g = None
        u = None
        if self.use_geom:
            g = self.geom_encoder(batch['landmarks'], mask=mask)  # (B,T,geom_dim)
            g = self.geom_proj(g)
        if self.use_normal:
            u = self.normal_mlp(batch['palm_normal'])
            u = self.normal_proj(u)

        # choose query sequence: ROI > GEOM > NORMAL
        if v is not None:
            q = v
            q_mask = mask
            kv_candidates = [x for x in (g, u) if x is not None]
        elif g is not None:
            q = g
            q_mask = mask
            kv_candidates = [x for x in (u,) if x is not None]
        elif u is not None:
            q = u
            q_mask = mask
            kv_candidates = []
        else:
            raise ValueError('At least one modality must be enabled (roi/geom/normal).')

        kv_list = []
        kv_mask = None
        if len(kv_candidates) == 0:
            kv = q
        else:
            kv_list.extend(kv_candidates)
            kv = torch.cat(kv_list, dim=1)  # concat on time axis
            if mask is not None:
                kv_mask = mask.repeat(1, len(kv_list))

        if self.use_text and self.use_text_as_kv:
            B = q.shape[0]
            text_kv = self.text_kv_proj(self.text_prototypes).unsqueeze(0).expand(B, -1, -1)
            kv = torch.cat([kv, text_kv], dim=1)
            if kv_mask is not None:
                text_mask = torch.ones(B, text_kv.shape[1], device=kv_mask.device)
                kv_mask = torch.cat([kv_mask, text_mask], dim=1)

        # cross-attention: Q=v, K/V=kv
        key_padding_mask = None
        if kv_mask is not None:
            valid = kv_mask > 0
            no_valid = ~valid.any(dim=1)
            if no_valid.any():
                valid = valid.clone()
                valid[no_valid, 0] = True
            key_padding_mask = ~valid
        fused, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)

        # event pooling
        z = self.attn_pool(fused, mask=q_mask)

        # classification via prototypes or linear head
        if self.use_text:
            z = F.normalize(self.proj_head(z), dim=-1)
            logits = self.logit_scale * (z @ self.text_prototypes.t())
        else:
            logits = self.cls(z)

        return logits
