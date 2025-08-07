import torch
from torch import nn

from src.models.gat.gat_layer import GATLayer
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder
from src.models.transformer.temporal_encoder import TemporalEncoder
from src.models.gat.gat import GATv2Conv


class SGATTransformer(nn.Module):
    def __init__(self, configs: dict):
        super(SGATTransformer, self).__init__()

        self.device = configs['device']

        gwe_configs = configs['graph_weight_encoder']
        gwe_configs['device'] = self.device
        self.gw_encoder = GraphWeightEncoder(gwe_configs)

        te_configs = configs['temporal_encoder']
        te_configs['device'] = self.device
        self.temporal_encoder = TemporalEncoder(te_configs)

        gat_configs = configs['gat']
        self.gat_layer = GATLayer(gat_configs)
        self.fc_gat_out = nn.Linear(gat_configs['dim_model'], 1)

    def forward(self, x):
        x_gwe = self.gw_encoder(x)
        x_te = self.temporal_encoder(x)

        gat_out = self.gat_layer(x_te, x_gwe, x)
        gat_out = self.fc_gat_out(gat_out)
        return gat_out
