import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CellGuidedCrossAttention(nn.Module):
    """
    Cross-attn: queries from GAT; keys/values from U-Net features.
    Returns per-cell fused features and attention maps (optional).
    """
    def __init__(self, unet_channels, q_dim, d_model=256, n_heads=8, out_dim=256,
                 add_xy_to_q=True, xy_dim=64, use_pos_enc=True, dropout=0.0):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.add_xy_to_q = add_xy_to_q
        self.d_model = d_model

        # Projections
        self.k_proj = nn.Linear(unet_channels, d_model)
        self.v_proj = nn.Linear(unet_channels, d_model)
        self.q_proj = nn.Linear(q_dim + (xy_dim if add_xy_to_q else 0), d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                         batch_first=True, dropout=dropout)

        # FFN head: fuse attended features with original query
        self.out = nn.Sequential(
            nn.LayerNorm(d_model + q_dim),
            nn.Linear(d_model + q_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

        # Optional small MLP to compress xy positional embedding
        if add_xy_to_q:
            self.xy_proj = nn.Sequential(
                nn.Linear(xy_dim, xy_dim),
                nn.GELU(),
                nn.Linear(xy_dim, xy_dim)
            )

    def forward(self, unet_feats, gat_q, cell_xy=None, key_mask=None, query_mask=None):
        """
        unet_feats: (B,C,H,W)
        gat_q: (B,M,Dq)
        cell_xy: (B,M,2) in [0,1] (optional but recommended)
        key_mask: (B, L) bool, True=pad/mask-out tokens
        query_mask: (B, M) bool, True=invalid queries (we'll zero them)
        """
        B, C, H, W = unet_feats.shape
        _, M, Dq = gat_q.shape
        device = unet_feats.device

        # Flatten U-Net features to tokens
        feats = unet_feats.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B,L,C)

        # Add 2D positional encodings to keys/values
        if self.use_pos_enc:
            pe = self.sinusoidal_2d_pos_enc(H, W, self.d_model, device=device)  # (L,d_model)
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B,L,d_model)
        else:
            pe = torch.zeros(B, H*W, self.d_model, device=device)

        K = self.k_proj(feats) + pe           # (B,L,d_model)
        V = self.v_proj(feats)                # (B,L,d_model)

        Q = self.q_proj(gat_q)  # (B,M,d_model)

        # Cross-attention (batch_first=True): Q attends to K,V
        # key_padding_mask: True to ignore; attn_mask can also be used if needed.
        attn_out, attn_weights = self.mha(Q, K, V, key_padding_mask=key_mask)  # out: (B,M,d_model), weights: (B,M,L)

        # Residual fusion with original GAT query
        fused = torch.cat([attn_out, gat_q], dim=-1)  # (B,M,d_model+Dq)
        fused = self.out(fused)  # (B,M,out_dim)

        # Optionally zero-out invalid query positions
        if query_mask is not None:
            fused = fused.masked_fill(query_mask.unsqueeze(-1), 0.0)
            attn_weights = attn_weights.masked_fill(query_mask.unsqueeze(-1), 0.0)

        return fused, attn_weights.view(B, M, H, W)

    def sinusoidal_2d_pos_enc(self, H, W, dim, device):
        """Return (H*W, dim) sin-cos 2D positional encodings."""
        assert dim % 4 == 0
        dim_half = dim // 2
        dim_quarter = dim // 4
        y = torch.linspace(0, 1, steps=H, device=device)
        x = torch.linspace(0, 1, steps=W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H,W)
        yy = yy.reshape(-1, 1)  # (L,1)
        xx = xx.reshape(-1, 1)  # (L,1)
        div = torch.exp(torch.arange(0, dim_quarter, device=device) * (-math.log(10000.0) / dim_quarter))
        pe_y = torch.cat([torch.sin(yy * div), torch.cos(yy * div)], dim=1)  # (L, dim/2)
        pe_x = torch.cat([torch.sin(xx * div), torch.cos(xx * div)], dim=1)  # (L, dim/2)
        pe = torch.cat([pe_y, pe_x], dim=1)  # (L, dim)
        return pe  # (L, dim)
