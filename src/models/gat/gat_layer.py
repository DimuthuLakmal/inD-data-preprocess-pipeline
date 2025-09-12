import torch
from torch import nn

from models.gat.gat import GATv2Conv


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()

        dim_model = config['dim_model']
        dim_edge = config['dim_edge']

        self.gat = GATv2Conv(in_channels=(dim_model, dim_model),
                             out_channels=dim_model,
                             heads=config['num_heads'],
                             dropout=config['dropout'],
                             concat=False,
                             edge_dim=dim_edge,
                             add_self_loops=False)

        self.hoc_emb = nn.Linear(config['ogm_input_dim'], dim_model)

    def forward(self, x_te_batch, x_gwe_batch, x_batch, map_output):
        edge_attr_batch, edge_index_batch, hidden_ogm_cells = x_batch['edge_weights'], x_batch['edge_index'], x_batch[
            'hidden_ogm_cells']
        x_hoc_batch = self.hoc_emb(hidden_ogm_cells)
        map_output = torch.repeat_interleave(map_output.unsqueeze(dim=1), repeats=x_hoc_batch.shape[1], dim=1)
        # x_hoc_batch = torch.concat([x_hoc_batch, map_output], dim=-1)

        gat_out_batch = []
        for (x_te, x_gwe, x_hoc, edge_index, edge_attr) in zip(x_te_batch, x_gwe_batch, x_hoc_batch, edge_index_batch, edge_attr_batch):
            edge_index = edge_index.to(x_te.device)
            edge_attr = edge_attr.to(x_te.device)

            gat_out = self.gat([x_te, x_hoc], edge_index, edge_attr)
            gat_out_batch.append(gat_out)

        return torch.stack(gat_out_batch)
