import torch.nn as nn
import torch.nn.functional as F


# This code defines an attention pooling layer for the transformer model to output a single value instead a sequence.
class AttentionPool(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.attn = nn.Linear(dim_model, 1)

    def forward(self, x):
        # x -> B, N, T, F
        weights = F.softmax(self.attn(x), dim=2)  # B, N, T, 1
        out = (weights * x).sum(dim=2)           # B, N, F
        return out
