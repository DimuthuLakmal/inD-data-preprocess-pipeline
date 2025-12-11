import torch
import torch.nn as nn

class VehicleTemporalEncoderLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,          # (B, T, F)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x, hx=None):
        # out: (B, T, H * num_directions)
        # h_n: (num_layers * num_directions, B, H)
        B, N, T, F = x.size()
        x = x.view(B * N, T, F)
        out, (h_n, c_n) = self.lstm(x, hx)

        h_last = h_n[-self.num_directions:]          # (num_directions, B, H)
        h_last = h_last.transpose(0, 1).reshape(x.size(0), -1)
        h_last = h_last.view(B, N, -1)
        return h_last
