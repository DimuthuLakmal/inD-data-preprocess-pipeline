from torch import nn

from src.models.gat.gat_layer import GATLayer
from src.models.gate.expert_gating import GatedFusion
from src.models.transformer.temporal_encoder import TemporalEncoder
from src.models.vision.map_encoder import MapEncoder


class VSTSBGT(nn.Module):
    def __init__(self, configs: dict):
        super(VSTSBGT, self).__init__()

        self.device = configs['device']

        te_configs = configs['obs_temporal_encoder']
        te_configs['device'] = self.device
        self.temporal_encoder = TemporalEncoder(d_in=te_configs['input_dim'],
                                                d_model=te_configs['dim_model'],
                                                nhead=te_configs['num_heads'],
                                                num_layers=te_configs['num_layers'],
                                                dropout=te_configs['dropout'])

        ze_configs = configs['z_temporal_encoder']
        ze_configs['device'] = self.device
        self.z_encoder = TemporalEncoder(d_in=te_configs['input_dim'],
                                         d_model=te_configs['dim_model'],
                                         nhead=te_configs['num_heads'],
                                         num_layers=te_configs['num_layers'],
                                         dropout=te_configs['dropout'])

        gat_configs = configs['gat']
        self.gat_layer = GATLayer(gat_configs)

        map_encoder_configs = configs['map_encoder']
        self.map_encoder = MapEncoder(d_model=map_encoder_configs['dim_model'],
                                      pretrained=map_encoder_configs['pretrained'],
                                      global_pool=map_encoder_configs['pooling'],
                                      architecture=map_encoder_configs['architecture'])

        self.fusion = GatedFusion(d_veh=gat_configs['dim_model'],
                                  d_img=map_encoder_configs['dim_model'],
                                  use_cell_in_gate=False)

        self.fc_out = nn.Linear(gat_configs['dim_model'], 1)

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
        z_te = self.z_encoder(veh_feat, seq_mask, vehicle_mask)
        gat_out, l2_loss, z_mask = self.gat_layer(x_te, z_te, cell_feat, x['edge_weights'], x['edge_index'])
        gat_out = gat_out.squeeze(1)

        # unet_out = self.unet(x)
        map_inputs = map_img.permute(0, 3, 1, 2)
        map_output = self.map_encoder(map_inputs)

        h_fused, gates = self.fusion(gat_out, map_output)

        # out_fc = self.fc_out(h_fused)
        out_fc = self.fc_out(gat_out)

        return out_fc, gates, l2_loss, z_mask
