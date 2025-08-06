from torch import nn

from src.models.gat.gat import GATv2Conv


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()

        dim_model = config['dim_model']

        self.gat = GATv2Conv(in_channels=(dim_model, dim_model),
                             out_channels=dim_model,
                             heads=config['num_heads'],
                             dropout=config['dropout'],
                             concat=False,
                             edge_dim=dim_model,
                             add_self_loops=False)

        self.hoc_emb = nn.Linear(config['ogm_input_dim'], dim_model)

    def forward(self, x_te_batch, x_gwe_batch, x_batch):
        edge_attr, edge_index, hidden_ogm_cells = x_batch['edge_weights'], x_batch['edge_index'], x_batch[
            'hidden_ogm_cells']
        x_hoc_batch = self.hoc_emb(hidden_ogm_cells)

        gat_out_batch = []
        for (x_te, x_gwe, x_hoc) in zip(x_te_batch, x_gwe_batch, x_hoc_batch):
            gat_out = self.gat([x_te, x_hoc], edge_index, edge_attr)
            gat_out.append(gat_out)

        return gat_out_batch
