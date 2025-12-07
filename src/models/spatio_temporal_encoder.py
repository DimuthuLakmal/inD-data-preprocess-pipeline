import torch
from torch import nn

from src.models.gat.gat_layer import GATLayer
from src.models.transformer.expert_gating import GatedFusion
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder
from src.models.transformer.vehicle_temporal_encoder import VehicleTemporalEncoder
from src.models.vision.base_models import FrameEncoder


class SGATTransformer(nn.Module):
    def __init__(self, configs: dict):
        super(SGATTransformer, self).__init__()

        self.device = configs['device']

        gwe_configs = configs['graph_weight_encoder']
        gwe_configs['device'] = self.device
        self.gw_encoder = GraphWeightEncoder(gwe_configs)

        te_configs = configs['temporal_encoder']
        te_configs['device'] = self.device
        self.temporal_encoder = VehicleTemporalEncoder(10, 32, nhead=4, num_layers=2, dropout=0.1)
        self.z_encoder = VehicleTemporalEncoder(10, 32, nhead=4, num_layers=2, dropout=0.1)

        gat_configs = configs['gat']
        self.gat_layer = GATLayer(gat_configs)

        self.map_encoder = FrameEncoder(d_model=64, pretrained=True, global_pool='avg')

        self.fusion = GatedFusion(d_veh=32, d_img=64, q_dim=32, use_cell_in_gate=False)

        self.fc_out = nn.Linear(32, 1)

    def reset_parameters(self):
        """Reset parameters of the model."""
        nn.init.uniform_(self.fc_gat_out.weight, a=-1.0, b=1.0)
        # TODO: since the classes are not balanced, the weights can be initialized as pos/total

    def forward(self, x):
        seq_mask = x['seq_mask']  # Sequence mask for the historical observations
        vehicle_mask = x['vehicle_mask']
        cell_feat = x['hidden_ogm_cells']
        veh_feat = x['historical_adjacent_obs']
        map_img = x['map_obs']

        x_te = self.temporal_encoder(veh_feat, seq_mask, vehicle_mask)
        z_te = self.temporal_encoder(veh_feat, seq_mask, vehicle_mask)
        gat_out, l2_loss, z_mask = self.gat_layer(x_te, z_te, cell_feat, x['edge_weights'], x['edge_index'])
        gat_out = gat_out.squeeze(1)

        # unet_out = self.unet(x)
        map_inputs = map_img.permute(0, 3, 1, 2)
        map_output = self.map_encoder(map_inputs)

        h_fused, gates = self.fusion(gat_out, map_output)

        out_fc = self.fc_out(h_fused)
        # out_fc = self.fc_out(gat_out)

        return out_fc, gates, l2_loss, z_mask
