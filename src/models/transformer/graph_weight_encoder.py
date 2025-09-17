import torch
from torch import nn

from src.models.transformer.encoder_block import EncoderBlock
from src.models.transformer.positional_encoding import PositionalEncoder


class GraphWeightEncoder(nn.Module):
    def __init__(self, config):
        super(GraphWeightEncoder, self).__init__()

        self.device = config['device']

        num_heads = config['num_heads']
        num_layers = config['num_layers']
        out_dim = config['dim_model']
        dim_model = config['dim_model']
        input_dim = config['input_dim']
        seq_len = config['seq_len']

        # embedding and positional encoder
        self.hist_emb = nn.Linear(input_dim, dim_model)
        self.positional_encoder = PositionalEncoder(seq_len, dim_model, matrix_dim=4)

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

        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(num_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(num_layers)])

        self.fc_out = nn.Linear(dim_model, out_dim)

    def forward(self, x, seq_mask=None):
        x_adjacent_hist = x['historical_adjacent_obs']
        # x_ego_hist = x['historical_ego_obs'].to(self.device)
        # x_hist = torch.cat((x_ego_hist.unsqueeze(dim=1), x_adjacent_hist), dim=1)

        x = self.hist_emb(x_adjacent_hist)
        out_e = self.positional_encoder(x)
        out_e_shp = out_e.shape

        for enc_layer, conv_q, conv_k in zip(self.layers, self.conv_q_layers, self.conv_k_layers):
            # output of temporal encoder layer
            out_e = out_e.view(-1, out_e_shp[2], out_e_shp[3])
            out_transposed = out_e.transpose(2, 1)
            q = conv_q(out_transposed).transpose(2, 1)
            k = conv_k(out_transposed).transpose(2, 1)
            v = out_e

            q = q.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            v = v.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            k = k.reshape(out_e_shp[0], out_e_shp[1], out_e_shp[2], out_e_shp[3])
            out_e = enc_layer(q, k, v, seq_mask)

        graph_w = out_e.permute(0, 2, 1, 3)

        return graph_w
