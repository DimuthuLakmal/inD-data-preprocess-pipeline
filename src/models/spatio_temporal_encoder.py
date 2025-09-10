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

    def forward(self, seq_emb, time_valid):
        # PyTorch wants True=PAD → invert your True=valid
        src_key_padding_mask = ~time_valid  # [B, T], True=pad
        x = self.pe(seq_emb)                # [B, T, D]
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, D]
        return self.ln(h)


class CellDecoderCrossOnly(nn.Module):
    """
    Cells as queries; cross-attend to temporal sequence memory.
    Inputs:
      Q_cell     : [B, N2, D]
      H_enc      : [B, T, D]
      time_valid : [B, T] bool  (True=valid)
      cell_pad   : [B, N2] bool (True=pad)  optional
    Output:
      logits     : [B, N2, 1]
    """
    def __init__(self, d_model=256, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [nn.ModuleDict(dict(
                ln1 = nn.LayerNorm(d_model),
                cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                ln2 = nn.LayerNorm(d_model),
                ffn = nn.Sequential(
                    nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model)
                ),
            ))]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d_model, 1)

    def forward(self, Q_cell, H_enc, time_valid, cell_pad=None):
        key_padding_mask = ~time_valid  # [B, T], True=pad for keys/values
        h = Q_cell
        for b in self.blocks:
            q = b["ln1"](h)
            attn_out, _ = b["cross"](q, H_enc, H_enc, key_padding_mask=key_padding_mask)
            h = h + attn_out
            h = h + b["ffn"](b["ln2"](h))
        if cell_pad is not None:
            h = h.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return self.head(h)  # [B, N2, 1]


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

    def forward(self, imgs, time_valid):
        imgs = imgs.permute(0, 1, 4, 2, 3)  # [B, T, 3, H, W]
        seq_emb = self.frame(imgs)                 # [B, T, D]
        H_enc  = self.temporal(seq_emb, time_valid)  # [B, T, D]
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
        self.decoder = CellDecoderCrossOnly(d_model=d_model, nhead=nhead,
                                            layers=dec_layers, dropout=dropout)

    def forward(self, imgs, time_valid, Q_cell, cell_pad=None):
        H_enc = self.encoder(imgs, time_valid)          # [B, T, D]
        logits = self.decoder(Q_cell, H_enc, time_valid, cell_pad)  # [B, N2, 1]
        return logits

