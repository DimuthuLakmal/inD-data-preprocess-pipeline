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


class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, nhead: int = 4, head_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.nhead   = nhead
        self.head_dim= head_dim
        self.d_attn  = nhead * head_dim

        self.Wq = nn.Linear(q_dim,  self.d_attn, bias=False)
        self.Wk = nn.Linear(kv_dim, self.d_attn, bias=False)
        self.Wv = nn.Linear(kv_dim, self.d_attn, bias=False)

        self.out = nn.Linear(self.d_attn, q_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_o = nn.LayerNorm(q_dim)

    def _reshape_heads(self, x):
        # x: [B, L, d_attn] -> [B, nH, L, head_dim]
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, Q, KV, key_padding_mask=None):
        """
        Q : [B, N2, q_dim]
        KV: [B, T,  kv_dim]
        key_padding_mask: [B, T] bool (True=pad)
        """
        B, N2, _ = Q.shape
        _,  T, _ = KV.shape

        q = self.ln_q(Q)
        qh = self._reshape_heads(self.Wq(q))        # [B, nH, N2, Hd]
        kh = self._reshape_heads(self.Wk(KV))       # [B, nH,  T, Hd]
        vh = self._reshape_heads(self.Wv(KV))       # [B, nH,  T, Hd]

        # scaled dot-product attention
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,nH,N2,T]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, vh)                # [B,nH,N2,Hd]
        ctx = ctx.transpose(1, 2).contiguous().view(B, N2, self.d_attn)  # [B,N2,d_attn]
        out = self.out(ctx)                          # [B,N2,q_dim]
        out = self.ln_o(Q + out)                     # residual + norm
        return out


class CellDecoderBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, nhead=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.cross = CrossAttention(q_dim, kv_dim, nhead=nhead, head_dim=head_dim, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, 4*q_dim), nn.GELU(), nn.Linear(4*q_dim, q_dim)
        )

    def forward(self, Q_cell, H_enc, key_padding_mask=None):
        h = self.cross(Q_cell, H_enc, key_padding_mask)
        h = h + self.ffn(h)
        return h  # [B,N2,q_dim]


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

    def forward(self, x, time_maks, veh_maks=None):
        """
        x: [B, N1, T, d_in]
        time_valid: [B, N1, T]  (True=pad, False=keep). Required.
        veh_valid:  [B, N1]     (True=keep). Optional; if given, will zero invalid outputs.
        """
        B, N1, T, d_in = x.shape
        assert time_maks.shape == (B, N1, T), "time_valid must be [B,N1,T] bool"

        # Flatten vehicles into batch axis for per-vehicle temporal encoding
        x = self.proj_in(x).view(B * N1, T, -1)  # [B*N1, T, D]
        pad_t = time_maks.view(B * N1, T)  # True = pad (Transformer convention)

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

        if veh_maks is not None:
            assert veh_maks.shape == (B, N1)
            h_veh = h_veh * ~veh_maks.unsqueeze(-1)

        return h_veh


class CellQueryEmb(nn.Module):
    """
    Inputs
      cell_xy_norm : [B, N2, 2]  (x,y ∈ [-1,1] in the SAME image/map space used by your frames)
      cell_pad     : [B, N2] bool (True=pad)  optional

    Output
      Q_cell       : [B, N2, D]
    """
    def __init__(self, d_model=256, mode="fourier", n_freq=8, mlp_hidden=128, dropout=0.1):
        super().__init__()
        assert mode in {"mlp", "fourier"}
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        in_dim = 2

        self.mode_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # Small init on the last layer helps keep logits stable early on
        nn.init.trunc_normal_(self.mode_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mode_mlp[-1].bias)

    def forward(self, cell_xy_norm, cell_pad=None):
        feats = cell_xy_norm                        # [B,N2,2]

        q = self.mode_mlp(feats)                        # [B,N2,D]
        q = self.ln(self.dropout(q))

        if cell_pad is not None:
            q = q.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return q


class CellDecoder(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, n_layers=2, nhead=4, head_dim=32, dropout=0.1):
        super().__init__()

        self.q_emb = CellQueryEmb(d_model=q_dim, mode="mlp")

        self.blocks = nn.ModuleList([
            CellDecoderBlock(q_dim, kv_dim, nhead, head_dim, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(q_dim, 1)

    def forward(self, Q_cell, H_enc, key_padding_mask=None, cell_pad=None):
        h = Q_cell
        h = self.q_emb(h, cell_pad)
        for blk in self.blocks:
            h = blk(h, H_enc, key_padding_mask)
        if cell_pad is not None:
            h = h.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return self.head(h)  # [B,N2,1]


class CellsFromVehicles(nn.Module):
    def __init__(self, d_vehicle_in, q_dim, d_model=128, nhead=4, Lenc=2, Ldec=2):
        super().__init__()
        self.enc = VehicleTemporalEncoder(d_vehicle_in, d_model, nhead=nhead, num_layers=Lenc, dropout=0.1)
        self.dec = CellDecoder(q_dim, d_model, n_layers=Ldec, nhead=nhead, head_dim=16, dropout=0.1)

    def forward(self, veh_feats, cell_feats, seq_pad_mask=None, vehicle_pad_mask=None, tgt_pad_mask=None):
        enc = self.enc(veh_feats, seq_pad_mask, vehicle_pad_mask)  # [B,N1,D]
        logits = self.dec(cell_feats, enc, cell_pad=tgt_pad_mask)  # [B,N2,1]
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
        self.cells_from_vehicles = CellsFromVehicles(d_vehicle_in=3, q_dim=16, d_model=64, nhead=4, Lenc=4, Ldec=4)
        self.image_encoder = ImageBackbone(out_dim=128)
        self.fusion = GatedFusion(d_veh=16, d_img=128, q_dim=16)
        self.query_encoder = CellQueryEncoder(d_model=16, d_pos=2)
        self.head = nn.Linear(16, 1)

    def forward(self, veh_feats, cell_feats, seq_mask, cell_mask, vehicle_maks, map):
        """
        veh_feats: [B, N1, T, d_in]  (historical adjacent vehicles)
        cell_feats: [B, N2, d_in]    (hidden ogm cells)
        seq_mask: [B, N1, T]         (True=pad in vehicle history)
        cell_mask: [B, N2]                (True=valid cell, False=padded cell)
        """
        h_veh = self.cells_from_vehicles(veh_feats, cell_feats,
                                         seq_pad_mask=seq_mask,
                                         vehicle_pad_mask=vehicle_maks,
                                         tgt_pad_mask=~cell_mask)  # [B,N2,1]

        # h_img = self.image_encoder(map.permute(0, 3, 1, 2))  # [B, D, Hf, Wf]
        # h_img = self.sample_cells_from_feat(h_img, cell_feats)  # [B, N2, D]
        # Q_cell = self.query_encoder(cell_feats, cell_valid=cell_mask)  # [B,N2,D]
        #
        # H_fused, gates = self.fusion(Q_cell, h_veh, h_img)
        # logits = self.head(H_fused)  # [B,N2,1]

        return h_veh

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
