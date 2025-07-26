import torch
from torch import nn

from src.models.transformer.encoder_block import EncoderBlock
from src.models.transformer.positional_encoding import PositionalEncoder


class GraphWeightEncoder(nn.Module):
    def __init__(self,
            input_dim,
            dim_model,
            num_heads,
            num_encoder_layers,
            max_seq_len):
        super(GraphWeightEncoder, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_encoder_layers
        out_dim = dim_model

        # embedding and positional encoder
        self.hist_emb = nn.Linear(input_dim, dim_model)
        self.positional_encoder = PositionalEncoder(max_seq_len, dim_model, matrix_dim=4)

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=self.num_heads,
                src_dropout=.1,
                ff_dropout=0.2,
                expansion_factor=4,
                mask=False
            ) for i in range(self.num_layers)])

        self.conv_q_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.conv_k_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.fc_out = nn.Linear(dim_model, out_dim)

    def _get_edge_index(self, n_nodes):
        return [[x for x in range(n_nodes)], [0] * n_nodes]

    def forward(self, x):
        edge_weights = x['edge_weight']
        edge_idx = x['edge_index']
        x_hist = x['hist_feat']
        x = self.hist_emb(x_hist)
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
            out_e = enc_layer(q, k, v)

        graph_w = out_e.permute(0, 2, 1, 3)

        return graph_w