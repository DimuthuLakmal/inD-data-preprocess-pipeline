import torch
from torch import nn

from models.transformer.atten_pooling import AttentionPool
from models.transformer.encoder_block import EncoderBlock
from models.transformer.positional_encoding import PositionalEncoder


class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super(TemporalEncoder, self).__init__()

        self.device = config['device']

        num_heads = config['num_heads']
        num_layers = config['num_layers']
        out_dim = config['dim_model']
        dim_model = config['dim_model']
        input_dim = config['input_dim']
        seq_len = config['seq_len']

        # embedding and positional encoder
        self.hist_emb = nn.Linear(input_dim, dim_model)
        self.positional_encoder = PositionalEncoder(seq_len + 1, dim_model, matrix_dim=4) # for CLS token

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim_model))

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=num_heads,
                src_dropout=.1,
                ff_dropout=0.2,
                expansion_factor=4,
                mask=False
            ) for i in range(num_layers)])

        # self.conv_q_layers = nn.ModuleList(
        #     [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
        #      for _ in range(num_layers)])
        #
        # self.conv_k_layers = nn.ModuleList(
        #     [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
        #      for _ in range(num_layers)])

        self.attn_pool = AttentionPool(dim_model)
        self.fc_out = nn.Linear(dim_model, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        nn.init.uniform_(self.hist_emb.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.fc_out.weight, a=-1.0, b=1.0)
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, a=-1.0, b=1.0)
        # for conv_q, conv_k in zip(self.conv_q_layers, self.conv_k_layers):
        #     torch.nn.init.xavier_uniform_(conv_q.weight)
        #     torch.nn.init.xavier_uniform_(conv_k.weight)

    def forward(self, x, mask=None):
        x_adjacent_hist = x['historical_adjacent_obs']
        B, N, T, C = x_adjacent_hist.shape  # B, N, T, C

        x = self.hist_emb(x_adjacent_hist)

        cls_tokens = self.cls_token.expand(B, N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        # Increase the mask size for CLS token
        if mask is not None:
            cls_mask = torch.zeros((B, N, 1), dtype=torch.bool, device=self.device)
            mask = torch.cat((cls_mask, mask), dim=2)


        out_e = self.positional_encoder(x)
        out_e_shp = out_e.shape

        for enc_layer in self.layers:
            q, v, k = out_e, out_e, out_e
            out_e = enc_layer(q, k, v, mask)

        # out_e = self.attn_pool(out_e)

        return out_e[:, :, 0, :]
