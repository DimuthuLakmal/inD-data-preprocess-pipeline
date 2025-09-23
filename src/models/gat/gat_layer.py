import torch
from torch import nn

from src.models.gat.gat import GATv2Conv
from src.models.transformer.cell_query_emb import CellQueryEmb


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()

        dim_adj_model = config['dim_adj_model']
        dim_cell_model = config['dim_cell_model']
        dim_model = config['dim_model']
        dim_edge = config['dim_edge']

        self.gat = GATv2Conv(in_channels=(dim_adj_model, dim_cell_model),
                             out_channels=dim_model,
                             heads=config['num_heads'],
                             dropout=config['dropout'],
                             concat=False,
                             edge_dim=dim_edge,
                             add_self_loops=False)

        self.cell_emb = CellQueryEmb(d_model=dim_cell_model, mode="mlp")

    def forward(self, x_te_batch, cell_batch, edge_attr_batch, edge_index_batch):
        x_cell_batch = self.cell_emb(cell_batch)

        gat_out_batch = []
        ls_loss_batch = 0
        for (x_te, x_cell, edge_index, edge_attr) in zip(x_te_batch, x_cell_batch, edge_index_batch, edge_attr_batch):
            edge_index = edge_index.to(x_te.device)
            edge_attr = edge_attr.to(x_te.device)

            gat_out, l2_loss = self.gat([x_te, x_cell], edge_index, edge_attr)
            gat_out_batch.append(gat_out)
            ls_loss_batch += l2_loss

        return torch.stack(gat_out_batch), ls_loss_batch
