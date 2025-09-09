import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.gated_fusion import GatedFusion
from src.models.vision.base_models import ImageBackbone


# --- Small helpers ---
class FFN(nn.Module):
    def __init__(self, d_model, d_ff=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff * d_model),
            nn.GELU(),
            nn.Linear(d_ff * d_model, d_model),
        )

    def forward(self, x): return self.net(x)


class CrossAttnBlock(nn.Module):
    """One decoder block WITHOUT self-attention: only cross-attn + FFN."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q_tokens, kv_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None, attn_bias=None):
        """
        q_tokens: [B, N2, D]  (cells)
        kv_tokens:[B, N1, D]  (vehicles)
        src_key_padding_mask: [B, N1]  True = pad in encoder
        tgt_key_padding_mask: [B, N2]  True = pad in decoder (optional)
        attn_bias: optional additive bias for attention logits, shape [B, N2, N1]
        """
        q = self.ln1(q_tokens)

        # MultiheadAttention supports attn_mask of shape [N2, N1] or [B*nH, N2, N1].
        # For per-batch bias, we can fold it per-head via custom modules; here we skip attn_bias for simplicity.
        x, _ = self.cross(q, kv_tokens, kv_tokens,
                          key_padding_mask=src_key_padding_mask)  # no tgt_key_padding_mask arg in cross-attn
        x = x + q_tokens  # residual

        y = self.ffn(self.ln2(x))
        y = y + x  # residual

        # Optionally zero out padded target positions (keep gradients off them downstream)
        if tgt_key_padding_mask is not None:
            y = y.masked_fill(~tgt_key_padding_mask.unsqueeze(-1), 0.0)
        return y


# --- Sinusoidal 1D positional encoding (time) ---
class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        # x: [B*, T, D]  -> add PE for first T positions
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # [1,T,D] + [B*,T,D]


class VehicleTemporalEncoder(nn.Module):
    """
    Inputs:
      x            : [B, N1, T, d_in]
      time_valid   : [B, N1, T]   (bool) True = real timestep, False = pad
      veh_valid    : [B, N1]      (bool) optional; if provided, invalid vehicles are zeroed in output

    Output:
      h_veh        : [B, N1, d_model]  (one embedding per vehicle)
    """

    def __init__(self, d_in, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.posenc = SinusoidalPosEnc(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # One learnable CLS token shared across vehicles (expanded per sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, time_valid, veh_valid=None):
        """
        x: [B, N1, T, d_in]
        time_valid: [B, N1, T]  (True=pad, False=keep). Required.
        veh_valid:  [B, N1]     (True=keep). Optional; if given, will zero invalid outputs.
        """
        B, N1, T, d_in = x.shape
        assert time_valid.shape == (B, N1, T), "time_valid must be [B,N1,T] bool"

        # Flatten vehicles into batch axis for per-vehicle temporal encoding
        x = self.proj_in(x).view(B * N1, T, -1)  # [B*N1, T, D]
        pad_t = time_valid.view(B * N1, T)  # True = pad (Transformer convention)

        # Prepend CLS (not padded)
        cls = self.cls_token.expand(B * N1, 1, -1)  # [B*N1, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B*N1, T+1, D]

        # Build padding mask for T+1 (CLS is always valid => False)
        cls_pad = torch.zeros(B * N1, 1, dtype=torch.bool, device=pad_t.device)
        src_key_padding_mask = torch.cat([cls_pad, pad_t], dim=1)  # [B*N1, T+1], True=pad

        # Add positional encodings (T+1 because of CLS at position 0)
        x = self.posenc(x)

        # Encode (self-attn over time per vehicle)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B*N1, T+1, D]

        # Take CLS as per-vehicle summary
        h_cls = h[:, 0, :]  # [B*N1, D]
        h_veh = self.ln_out(h_cls).view(B, N1, -1)  # [B, N1, D]

        # Optionally zero out invalid vehicle slots
        if veh_valid is not None:
            h_veh = h_veh * veh_valid.unsqueeze(-1).to(h_veh.dtype)

        return h_veh


class CellDecoder(nn.Module):
    def __init__(self, d_in, d_model, n_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([CrossAttnBlock(d_model, nhead, dropout) for _ in range(n_layers)])
        # self.head = nn.Linear(d_model, 1)

    def forward(self, cell_feats, enc_feats, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # cell_feats: [B, N2, d_in] (could be learned queries or embedded (x,y))
        h = self.proj(cell_feats)
        for layer in self.layers:
            h = layer(h, enc_feats, src_key_padding_mask=src_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask)
        # logits = self.head(h)  # [B, N2, 1]
        return h


class CellsFromVehicles(nn.Module):
    def __init__(self, d_vehicle_in, d_cell_in, d_model=128, nhead=4, Lenc=2, Ldec=2):
        super().__init__()
        self.enc = VehicleTemporalEncoder(d_vehicle_in, d_model, nhead=4, num_layers=2, dropout=0.1)
        self.dec = CellDecoder(d_cell_in, d_model, n_layers=Ldec, nhead=nhead)

    def forward(self, veh_feats, cell_feats, src_pad_mask=None, tgt_pad_mask=None):
        enc = self.enc(veh_feats, src_pad_mask)  # [B,N1,D]
        logits = self.dec(cell_feats, enc, tgt_key_padding_mask=tgt_pad_mask)  # [B,N2,1]
        return logits


class CellQueryEncoder(nn.Module):
    """
    Build Q_cell tokens from cell coords and optional attributes.

    Inputs:
      cell_xy_norm : [B, N2, 2]  (x,y in [-1,1] at the IMAGE space you're using)
      cell_attr    : [B, N2, F_attr]  (optional extra features per cell)
      cell_valid   : [B, N2]  bool mask (optional; True=valid)

    Output:
      Q_cell       : [B, N2, D]
    """

    def __init__(
            self,
            d_model: int = 128,
            d_pos: int = 32,  # positional embedding size (even)
            dropout: float = 0.1,
            use_ln: bool = True,
    ):
        super().__init__()
        assert d_pos % 2 == 0, "d_pos must be even"

        # Final fusion + norm
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.pos_to_model = nn.Linear(d_pos, d_model)

    def forward(
            self,
            cell_xy_norm: torch.Tensor,  # [B,N2,2] in [-1,1]
            cell_valid: torch.Tensor = None  # [B,N2] bool or None
    ) -> torch.Tensor:
        q = self.pos_to_model(cell_xy_norm)  # [B,N2,d_model]
        q = self.ln(q)
        q = self.dropout(q)

        if cell_valid is not None:
            q = q * cell_valid.unsqueeze(-1).to(q.dtype)  # zero-out padded cells

        return q  # [B,N2,d_model]


class CellFromVehicleAndMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells_from_vehicles = CellsFromVehicles(9, 2, 64)
        self.image_encoder = ImageBackbone(out_dim=64)
        self.fusion = GatedFusion(64)
        self.query_encoder = CellQueryEncoder(d_model=64, d_pos=2)
        self.head = nn.Linear(64, 1)

    def forward(self, veh_feats, cell_feats, seq_mask, mask, map):
        """
        veh_feats: [B, N1, T, d_in]  (historical adjacent vehicles)
        cell_feats: [B, N2, d_in]    (hidden ogm cells)
        seq_mask: [B, N1, T]         (True=pad in vehicle history)
        mask: [B, N2]                (True=valid cell, False=padded cell)
        """
        h_veh = self.cells_from_vehicles(veh_feats, cell_feats,
                                         src_pad_mask=seq_mask,
                                         tgt_pad_mask=mask)  # [B,N2,1]

        h_img = self.image_encoder(map.permute(0, 3, 1, 2))  # [B, D, Hf, Wf]
        h_img = self.sample_cells_from_feat(h_img, cell_feats)  # [B, N2, D]
        Q_cell = self.query_encoder(cell_feats, cell_valid=mask)  # [B,N2,D]

        H_fused, gates = self.fusion(Q_cell, h_veh, h_img)
        logits = self.head(H_fused)  # [B,N2,1]

        return logits

    def sample_cells_from_feat(self, feat_map, cell_xy_norm):
        """
        feat_map:       [B, D, Hf, Wf] from ImageBackbone
        cell_xy_norm:   [B, N2, 2] with x,y in [-1, 1] in image coords (after any resizing/cropping)
        returns:        [B, N2, D] sampled via bilinear
        """
        B, D, Hf, Wf = feat_map.shape
        # grid_sample expects [B, H_out, W_out, 2], we want one sample per cell -> H_out=1, W_out=N2
        grid = cell_xy_norm.unsqueeze(1)  # [B, 1, N2, 2]
        # grid is in (x,y) order in [-1,1]; align_corners=False matches torchvision defaults
        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=False)  # [B, D, 1, N2]
        return sampled.squeeze(2).transpose(1, 2)  # [B, N2, D]
