import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, d_veh, d_img, q_dim=0, d_gate=128, use_cell_in_gate=True):
        super().__init__()
        self.use_cell = use_cell_in_gate and q_dim > 0

        # Gate computed in a small joint space
        gate_in = d_veh + d_img + (q_dim if self.use_cell else 0)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, 1)
        )
        # Project img to veh space for mixing
        self.img_to_veh = nn.Linear(d_img, d_veh)
        self.out_ln = nn.LayerNorm(d_veh)

    def forward(self, h_from_veh, h_from_img):
        # g_in = [h_from_veh, h_from_img]
        # g = torch.sigmoid(self.gate(torch.cat(g_in, dim=-1)))  # [B,N2,1]

        g = torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.tensor(0.2, dtype=torch.float32), dim=-1), h_from_veh.shape[0]), dim=-1).to(h_from_img.device)

        i_in_veh = self.img_to_veh(h_from_img)                 # [B,N2,d_veh]
        fused = g * i_in_veh + (1.0 - g) * h_from_veh          # [B,N2,d_veh]
        return self.out_ln(fused), g