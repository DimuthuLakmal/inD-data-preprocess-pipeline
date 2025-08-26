import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MapEncoderAttention(nn.Module):
    """
    Cross-attn: queries from GAT; keys/values from U-Net features.
    Returns per-cell fused features and attention maps (optional).
    """

    def __init__(self, img_dim, node_dim, d=128, residual=True):
        super().__init__()
        self.q = nn.Linear(node_dim, d, bias=False)  # node -> d
        self.k = nn.Linear(img_dim, d, bias=False)  # img -> d
        self.scale = 1.0 / math.sqrt(d)
        self.residual = residual
        # candidate transform mixes node + image, then we gate towards it
        self.candidate = nn.Sequential(
            nn.Linear(node_dim + img_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, img, nodes, mask=None):
        B, N, F_node = nodes.shape
        img_proj = self.k(img)  # (B, d)
        node_proj = self.q(nodes)  # (B, N, d)
        score = torch.einsum('bnd,bd->bn', node_proj, img_proj) * self.scale  # (B, N)
        gate = torch.sigmoid(score)  # (B, N)

        img_exp = img.unsqueeze(1).expand(-1, N, -1)  # (B, N, F_img)
        cand = self.candidate(torch.cat([nodes, img_exp], dim=-1))  # (B, N, F_node)

        fused = nodes + gate.unsqueeze(-1) * (cand - nodes) if self.residual \
            else (1 - gate).unsqueeze(-1) * nodes + gate.unsqueeze(-1) * cand

        if mask is not None:
            m = mask.bool().unsqueeze(-1)
            fused = torch.where(m, fused, torch.zeros_like(fused))
            gate = torch.where(mask.bool(), gate, torch.zeros_like(gate))

        return gate, fused

