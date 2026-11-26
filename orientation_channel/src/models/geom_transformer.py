import torch
import torch.nn as nn


class SkeletonTransformer(nn.Module):
    def __init__(self, in_channels=3, d_model=256, nhead=4, num_layers=3, num_joints=21, max_len=64):
        super().__init__()
        self.num_joints = num_joints
        self.proj = nn.Linear(in_channels, d_model)
        self.joint_emb = nn.Embedding(num_joints, d_model)
        self.time_emb = nn.Embedding(max_len, d_model)

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B,T,V,C)
        B,T,V,C = x.shape
        x = self.proj(x)
        # add joint and time embeddings
        joint_ids = torch.arange(V, device=x.device)
        time_ids = torch.arange(T, device=x.device)
        x = x + self.joint_emb(joint_ids)[None, None, :, :] + self.time_emb(time_ids)[None, :, None, :]

        # flatten tokens
        x = x.reshape(B, T*V, -1)

        key_padding_mask = None
        if mask is not None:
            # mask: (B,T), mask==0 -> pad
            # expand to (B,T*V)
            key_padding_mask = (mask == 0).unsqueeze(-1).expand(B, T, V).reshape(B, T*V)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)
        x = x.reshape(B, T, V, -1).mean(dim=2)  # (B,T,d_model)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
