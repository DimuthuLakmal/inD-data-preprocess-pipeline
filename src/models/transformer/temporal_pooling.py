import torch
import torch.nn as nn

class TemporalAttnPool(nn.Module):
    """
    H: [B,N,T,F], time_valid: [B,N,T] (True=valid)
    returns P: [B,N,F]
    """
    def __init__(self, Fdim, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(Fdim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)  # score per time step
        )

    def forward(self, H, seq_pad):
        B,N,T,Fdim = H.shape
        scores = self.phi(H)                         # [B,N,T,1]
        scores = scores.squeeze(-1)                  # [B,N,T]
        # mask: -inf where invalid
        scores = scores.masked_fill(seq_pad, float('-inf'))
        alpha  = torch.softmax(scores, dim=-1)       # [B,N,T]
        alpha  = torch.nan_to_num(alpha, nan=0.0)    # in case all masked
        P = torch.einsum('bnt,bntf->bnf', alpha, H)  # [B,N,F]
        return P