import torch
import math
import torch.nn as nn

from models.transformer.temporal_pooling import TemporalAttnPool


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
    def __init__(self, d_in, d_model=128, d_out=128, nhead=4, num_layers=2, dropout=0.1, use_cls=False):
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
        self.temp_pool = TemporalAttnPool(d_model)
        self.ln_out = nn.Linear(d_model, d_out)

    def forward(self, x, time_pad, veh_pad=None):
        """
        time_pad: True=pad, False=valid (PyTorch src_key_padding_mask convention)
        return_cls: if True and use_cls=True, also returns per-vehicle CLS summary
        """
        B, N1, T, _ = x.shape
        # assert time_pad.shape == (B, N1, T)

        # per-vehicle temporal encoding (no mixing between vehicles)
        x = self.proj_in(x).view(B * N1, T, -1)          # [B*N1, T, D]
        
        pad_t = None
        if time_pad is not None:
            pad_t = time_pad.view(B * N1, T)                 # [B*N1, T], True=pad

        x = self.posenc(x)                             # [B*N1, T, D]
        h = self.encoder(x, src_key_padding_mask=pad_t)# [B*N1, T, D]
        h_seq = self.ln_tok(h).view(B, N1, T, -1)      # [B, N1, T, D]

        # Optionally zero out invalid vehicles
        if veh_pad is not None:
            assert veh_pad.shape == (B, N1)
            h_seq = h_seq * (~veh_pad).unsqueeze(-1).unsqueeze(-1)

        h_seq = torch.nan_to_num(h_seq, nan=0.0)
        h_seq = self.temp_pool(h_seq, time_pad)  # [B, N1, D]
        h_seq = self.ln_out(h_seq)
        return h_seq