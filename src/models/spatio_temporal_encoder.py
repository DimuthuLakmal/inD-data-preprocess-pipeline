import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import math

class FrameEncoder(nn.Module):
    """
    Map image -> per-frame embedding.
    Input : imgs [B, T, 3, H, W]
    Output: emb  [B, T, D]
    """
    def __init__(self, d_model=256, pretrained=True, global_pool='avg'):
        super().__init__()
        m = resnet50(weights="DEFAULT" if pretrained else None)

        # Take ResNet trunk up to C5
        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        c5 = 2048

        # Project to d_model after global pooling
        self.global_pool = global_pool
        self.proj = nn.Linear(c5, d_model)

    def forward(self, imgs):  # [B, T, 3, H, W]
        B, T = imgs.shape[:2]
        x = imgs.view(B*T, *imgs.shape[2:])         # [B*T, 3, H, W]
        f = self.backbone(x)                        # [B*T, 2048, Hf, Wf]

        if self.global_pool == 'avg':
            f = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)   # [B*T, 2048]
        elif self.global_pool == 'max':
            f = F.adaptive_max_pool2d(f, 1).squeeze(-1).squeeze(-1)
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        emb = self.proj(f)                          # [B*T, D]
        emb = emb.view(B, T, -1)                    # [B, T, D]
        return emb


class SinPE1D(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)

class TemporalEncoder(nn.Module):
    """
    Inputs:
      seq_emb    : [B, T, D]     (from FrameEncoder)
      time_valid : [B, T] bool   (True=valid timestep, False=pad)
    Output:
      H_enc      : [B, T, D]     (sequence output, not pooled)
    """
    def __init__(self, d_model=256, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pe = SinPE1D(d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, seq_emb, src_key_padding_mask):
        # PyTorch wants True=PAD
        x = self.pe(seq_emb)                # [B, T, D]
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, D]
        return self.ln(h)


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

        if mode == "mlp":
            in_dim = 2
        else:
            # Fourier features for x and y separately: for each dim -> [sin,cos] * n_freq
            # total = 2 (raw x,y) + 4*n_freq
            in_dim = 2 + 4 * n_freq
            # register frequencies
            freqs = math.pi * (2.0 ** torch.arange(n_freq).float())  # [n_freq]
            self.register_buffer("freqs", freqs, persistent=False)

        self.mode_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # Small init on the last layer helps keep logits stable early on
        nn.init.trunc_normal_(self.mode_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mode_mlp[-1].bias)

    def fourier_encode(self, xy):  # xy: [B,N2,2]
        x, y = xy[..., 0:1], xy[..., 1:2]               # [B,N2,1]
        # [B,N2,n_freq]
        xw = x * self.freqs
        yw = y * self.freqs
        enc = torch.cat([
            torch.sin(xw), torch.cos(xw),               # [B,N2,n_freq*2]
            torch.sin(yw), torch.cos(yw)                # [B,N2,n_freq*2]
        ], dim=-1)
        return torch.cat([xy, enc], dim=-1)             # [B,N2, 2 + 4*n_freq]

    def forward(self, cell_xy_norm, cell_pad=None):
        if self.mode == "mlp":
            feats = cell_xy_norm                        # [B,N2,2]
        else:
            feats = self.fourier_encode(cell_xy_norm)   # [B,N2, 2+4*n_freq]

        q = self.mode_mlp(feats)                        # [B,N2,D]
        q = self.ln(self.dropout(q))

        if cell_pad is not None:
            q = q.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return q


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


class CellDecoder(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, n_layers=2, nhead=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CellDecoderBlock(q_dim, kv_dim, nhead, head_dim, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(q_dim, 1)

    def forward(self, Q_cell, H_enc, key_padding_mask=None, cell_pad=None):
        h = Q_cell
        for blk in self.blocks:
            h = blk(h, H_enc, key_padding_mask)
        if cell_pad is not None:
            h = h.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return self.head(h)  # [B,N2,1]


class MapSequenceEncoder(nn.Module):
    """
    Maps -> per-frame embeddings -> temporal sequence memory
    Input : imgs [B, T, 3, H, W], time_valid [B, T] (True=valid)
    Output: H_enc [B, T, D]
    """
    def __init__(self, d_model=256, nhead=4, enc_layers=2, pretrained=True):
        super().__init__()
        self.frame = FrameEncoder(d_model=d_model, pretrained=pretrained)
        self.temporal = TemporalEncoder(d_model=d_model, nhead=nhead, layers=enc_layers)

    def forward(self, imgs, time_seq_pad):
        imgs = imgs.permute(0, 1, 4, 2, 3)  # [B, T, 3, H, W]
        seq_emb = self.frame(imgs)                 # [B, T, D]
        H_enc  = self.temporal(seq_emb, time_seq_pad)  # [B, T, D]
        return H_enc


class SpatioTemporalEncoder(nn.Module):
    """
    Full model: maps -> per-frame embeddings -> temporal sequence memory -> per-cell decoding
    Inputs:
      imgs       : [B, T, 3, H, W]
      time_valid : [B, T] bool   (True=valid timestep, False=pad)
      Q_cell     : [B, N2, D]    (cell queries)
      cell_pad   : [B, N2] bool  (True=pad) optional
    Output:
      logits     : [B, N2, 1]
    """
    def __init__(self, d_model=256, nhead=4, enc_layers=2, dec_layers=2,
                 pretrained=True, dropout=0.1):
        super().__init__()
        self.encoder = MapSequenceEncoder(d_model=d_model, nhead=nhead,
                                          enc_layers=enc_layers, pretrained=pretrained)
        self.decoder = CellDecoder(q_dim=32, kv_dim=d_model, n_layers=dec_layers, nhead=nhead, head_dim=32, dropout=0.1)

    def forward(self, imgs, Q_cell, time_seq_pad, cell_pad=None):
        H_enc = self.encoder(imgs, time_seq_pad)          # [B, T, D]
        logits = self.decoder(Q_cell, H_enc, time_seq_pad, cell_pad)  # [B, N2, 1]
        return logits

