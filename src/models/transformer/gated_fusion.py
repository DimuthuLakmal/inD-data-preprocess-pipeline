import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(3*d),
            nn.Linear(3*d, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )  # outputs a scalar gate per cell
    def forward(self, h_cell, h_from_veh, h_from_img):
        # h_cell: original cell token (query features) [B,N2,D]
        g = torch.sigmoid(self.gate(torch.cat([h_cell, h_from_veh, h_from_img], dim=-1)))  # [B,N2,1]
        return g * h_from_img + (1 - g) * h_from_veh, g  # fused features, gate for inspection
