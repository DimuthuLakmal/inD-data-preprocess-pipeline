import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.gated_fusion import GatedFusion
from src.models.vision.base_models import FrameEncoder


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


class TwoStageVehicleCrossAttn(nn.Module):
    """
    Two-stage cross-attention:
      Stage A (temporal): per cell, attend over T for each vehicle separately.
      Stage B (vehicles): per cell, attend over vehicles using the Stage-A summaries.

    Inputs:
      Q_cell   : [B, N2, q_dim]             (cell query tokens; small dim ok)
      H_seq    : [B, N1, T, enc_dim]        (per-vehicle temporal encoder outputs)
      time_pad : [B, N1, T] (bool)          True = PAD timestep (ignored)
      veh_pad  : [B, N1]     (bool) or None True = PAD vehicle (ignored)

    Output:
      H_cell   : [B, N2, q_dim]             (updated cell features in q_dim space)
      (optional) alpha_t, beta_v for inspection (see forward(..., return_attn=True))
    """
    def __init__(self, q_dim: int, enc_dim: int, nhead: int = 4, head_dim: int = 32,
                 dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        assert (q_dim > 0) and (enc_dim > 0)
        self.nhead = nhead
        self.head_dim = head_dim
        self.d_attn = nhead * head_dim

        # Stage A projections (time attention within each vehicle)
        self.WqA = nn.Linear(q_dim,  self.d_attn, bias=False)
        self.WkA = nn.Linear(enc_dim, self.d_attn, bias=False)
        self.WvA = nn.Linear(enc_dim, self.d_attn, bias=False)

        # Stage B projections (vehicle attention using Stage-A summaries)
        self.WkB = nn.Linear(self.d_attn, self.d_attn, bias=False)
        self.WvB = nn.Linear(self.d_attn, self.d_attn, bias=False)

        # Output projection back to q_dim + post-attn FFN
        self.Wo  = nn.Linear(self.d_attn, q_dim)
        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_o = nn.LayerNorm(q_dim)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, ffn_mult*q_dim), nn.GELU(),
            nn.Linear(ffn_mult*q_dim, q_dim),
        )
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x, B, L):
        # x: [B, L, d_attn] -> [B, nH, L, Hd]
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, Q_cell, H_seq, time_pad, vehicle_pad_mask=None):
        """
        return_attn: if True, returns (H_cell, alpha_t, beta_v)
            alpha_t: [B, nH, N2, N1, T]  (temporal weights per vehicle)
            beta_v : [B, nH, N2, N1]     (vehicle weights)
        """
        B, N2, qd = Q_cell.shape
        _, N1, T, ed = H_seq.shape

        assert time_pad.shape == (B, N1, T)
        if vehicle_pad_mask is None:
            vehicle_pad_mask = torch.zeros(B, N1, dtype=torch.bool, device=Q_cell.device)

        # ---- Pre-norm Q ----
        Qn = self.ln_q(Q_cell)                               # [B,N2, q_dim]

        # Projections
        qA = self._split_heads(self.WqA(Qn), B, N2)          # [B,nH,N2,Hd]
        kA = self._split_heads(self.WkA(H_seq.view(B*N1*T, ed)).view(B, N1*T, -1), B, N1*T)
        vA = self._split_heads(self.WvA(H_seq.view(B*N1*T, ed)).view(B, N1*T, -1), B, N1*T)
        # Reshape kA/vA to [B,nH,N1,T,Hd]
        kA = kA.view(B, self.nhead, N1, T, self.head_dim)
        vA = vA.view(B, self.nhead, N1, T, self.head_dim)

        # ---- Stage A: temporal attention within each vehicle ----
        # scoresA: [B,nH,N2,N1,T] = qA · kA^T (over Hd)
        # qA: [B,nH,N2,Hd], kA: [B,nH,N1,T,Hd]
        scoresA = torch.einsum('bhid,bhntd->bhint', qA, kA) / (self.head_dim ** 0.5)

        # Mask padded timesteps
        # time_pad: [B,N1,T] True=pad -> expand to [B,1,1,N1,T]
        scoresA = scoresA.masked_fill(time_pad[:, None, None, :, :], float('-inf'))
        alpha_t = torch.softmax(scoresA, dim=-1)             # over T
        alpha_t = torch.nan_to_num(alpha_t, nan=0.0)         # handle all-pad edge cases

        # context per vehicle: [B,nH,N2,N1,Hd]
        ctxA = torch.einsum('bhint,bhntd->bhind', alpha_t, vA)

        # Merge heads across Hd per vehicle: [B,N2,N1,d_attn]
        ctxA = ctxA.transpose(1, 2).contiguous().view(B, N2, N1, self.d_attn)

        # ---- Stage B: attention across vehicles ----
        kB = self._split_heads(self.WkB(ctxA.view(B*N2*N1, self.d_attn)).view(B, N2*N1, -1), B, N2*N1)
        vB = self._split_heads(self.WvB(ctxA.view(B*N2*N1, self.d_attn)).view(B, N2*N1, -1), B, N2*N1)
        # reshape per cell: [B,nH,N2,N1,Hd]
        kB = kB.view(B, self.nhead, N2, N1, self.head_dim)
        vB = vB.view(B, self.nhead, N2, N1, self.head_dim)

        # reuse qA (same queries) for vehicle attention
        # scoresB: [B,nH,N2,N1] = qA · kB (over Hd)
        scoresB = torch.einsum('bhid,bhind->bhin', qA, kB)

        # Mask padded vehicles: veh_pad [B,N1] True=pad -> [B,1,1,N1]
        scoresB = scoresB.masked_fill(vehicle_pad_mask[:, None, None, :], float('-inf'))
        beta_v = torch.softmax(scoresB / (self.head_dim ** 0.5), dim=-1)
        beta_v = torch.nan_to_num(beta_v, nan=0.0)

        # Final context per head: [B,nH,N2,Hd]
        ctxB = torch.einsum('bh in, b h i n d -> b h i d', beta_v, vB)
        # Merge heads -> [B,N2,d_attn]
        ctxB = ctxB.transpose(1, 2).contiguous().view(B, N2, self.d_attn)

        # Output proj back to q_dim + residual + FFN
        out = self.Wo(self.drop(ctxB))                       # [B,N2,q_dim]
        h  = self.ln_o(Q_cell + out)
        return h


class CellDecoderBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, nhead=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.cross = TwoStageVehicleCrossAttn(q_dim, kv_dim, nhead=nhead, head_dim=head_dim, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, 4*q_dim), nn.GELU(), nn.Linear(4*q_dim, q_dim)
        )

    def forward(self, Q_cell, H_enc, key_padding_mask=None, vehicle_pad_mask=None):
        h = self.cross(Q_cell, H_enc, key_padding_mask, vehicle_pad_mask)
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
      x         : [B, N1, T, d_in]
      time_pad  : [B, N1, T]  (bool) True = PAD (ignored by attention), False = valid
      veh_pad   : [B, N1]     (bool) optional; True = PAD vehicle (zero out output)

    Outputs:
      h_seq     : [B, N1, T, d_model]  (per-vehicle sequence embeddings)
      h_cls     : [B, N1, d_model]     (optional; only if return_cls=True)
    """
    def __init__(self, d_in, d_model=128, nhead=4, num_layers=2, dropout=0.1, use_cls=False):
        super().__init__()
        self.use_cls = use_cls

        self.proj_in = nn.Linear(d_in, d_model)
        self.posenc  = SinusoidalPosEnc(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.ln_tok = nn.LayerNorm(d_model)
        self.ln_cls = nn.LayerNorm(d_model) if self.use_cls else None

    def forward(self, x, time_pad, veh_pad=None):
        """
        time_pad: True=pad, False=valid (PyTorch src_key_padding_mask convention)
        return_cls: if True and use_cls=True, also returns per-vehicle CLS summary
        """
        B, N1, T, _ = x.shape
        assert time_pad.shape == (B, N1, T)

        # per-vehicle temporal encoding (no mixing between vehicles)
        x = self.proj_in(x).view(B * N1, T, -1)          # [B*N1, T, D]
        pad_t = time_pad.view(B * N1, T)                 # [B*N1, T], True=pad

        x = self.posenc(x)                             # [B*N1, T, D]
        h = self.encoder(x, src_key_padding_mask=pad_t)# [B*N1, T, D]
        h_seq = self.ln_tok(h).view(B, N1, T, -1)      # [B, N1, T, D]

        # Optionally zero out invalid vehicles
        if veh_pad is not None:
            assert veh_pad.shape == (B, N1)
            h_seq = h_seq * (~veh_pad).unsqueeze(-1).unsqueeze(-1)

        h_seq = torch.nan_to_num(h_seq, nan=0.0)
        return h_seq


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

    def forward(self, Q_cell, H_enc, key_padding_mask=None, cell_pad=None, vehicle_pad_mask=None):
        h = Q_cell
        h = self.q_emb(h, cell_pad)
        for blk in self.blocks:
            h = blk(h, H_enc, key_padding_mask, vehicle_pad_mask)
        if cell_pad is not None:
            h = h.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return h # [B,N2,1]


class CellsFromVehicles(nn.Module):
    def __init__(self, d_vehicle_in, q_dim, d_model=128, nhead=4, Lenc=2, Ldec=2):
        super().__init__()
        self.enc = VehicleTemporalEncoder(d_vehicle_in, d_model, nhead=nhead, num_layers=Lenc, dropout=0.1)
        self.dec = CellDecoder(q_dim, d_model, n_layers=Ldec, nhead=nhead, head_dim=16, dropout=0.1)

    def forward(self, veh_feats, cell_feats, seq_pad_mask=None, vehicle_pad_mask=None, tgt_pad_mask=None):
        enc = self.enc(veh_feats, seq_pad_mask, vehicle_pad_mask)  # [B,N1,D]
        logits = self.dec(cell_feats, enc, cell_pad=tgt_pad_mask, key_padding_mask=seq_pad_mask, vehicle_pad_mask=vehicle_pad_mask)  # [B,N2,1]
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
        self.image_encoder = FrameEncoder(d_model=256, pretrained=True, global_pool='avg')
        self.fusion = GatedFusion(d_veh=16, d_img=256, q_dim=16, use_cell_in_gate=False)
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
                                         tgt_pad_mask=~cell_mask).squeeze()  # [B,N2,1]

        h_img = self.image_encoder(map.permute(0, 3, 1, 2))  # [B, D, Hf, Wf]
        # h_img = self.sample_cells_from_feat(h_img, cell_feats)  # [B, N2, D]
        # Q_cell = self.query_encoder(cell_feats, cell_valid=cell_mask)  # [B,N2,D]

        H_fused, gates = self.fusion(None, h_veh, h_img)
        logits = self.head(H_fused).unsqueeze(-1)  # [B,N2,1]

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

