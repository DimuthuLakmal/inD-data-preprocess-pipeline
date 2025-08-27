import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoExpertGatedFusion(nn.Module):
    """
    Learnable softmax gate over two expert outputs (y1, y2).
    Shapes supported: (B, ..., F)  [any number of middle dims].
    Returns: fused, alphas  where alphas has shape (B, ..., 2).
    """
    def __init__(self, feat_dim, hidden=128, temperature=1.0, layernorm=True):
        super().__init__()
        self.temperature = temperature
        self.layernorm = layernorm
        if layernorm:
            self.ln1 = nn.LayerNorm(feat_dim)
            self.ln2 = nn.LayerNorm(feat_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2)
        )
        # Start near-equal mixing (bias 0 -> logits ~ 0 -> softmax ~ [0.5, 0.5])
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, y1, y2, mask=None):
        """
        y1, y2: tensors with identical shape (B, ..., F)
        mask: optional boolean/0-1 mask with shape (B, ...) for padded positions
        """
        assert y1.shape == y2.shape, "y1 and y2 must have the same shape"
        B, f = y1.shape[0], y1.shape[-1]

        if self.layernorm:
            y1 = self.ln1(y1)
            y2 = self.ln2(y2)

        # Flatten tokens: (B, tokens, F)
        tokens = int(torch.tensor(y1.shape[1:-1]).prod()) if y1.ndim > 2 else 1
        y1f = y1.view(B, tokens, f)
        y2f = y2.view(B, tokens, f)

        x = torch.cat([y1f, y2f], dim=-1)             # (B, tokens, 2F)
        logits = self.gate(x)                         # (B, tokens, 2)
        alphas = F.softmax(logits / self.temperature, dim=-1)

        a1 = alphas[..., 0].unsqueeze(-1)             # (B, tokens, 1)
        a2 = alphas[..., 1].unsqueeze(-1)

        fused = a1 * y1f + a2 * y2f

        return fused