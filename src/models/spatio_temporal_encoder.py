import torch
from torch import nn

from src.models.gat.gat_layer import GATLayer
from src.models.transformer.cell_guided_cross_attention import CellGuidedCrossAttention
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder
from src.models.transformer.temporal_encoder import TemporalEncoder
from src.models.unet.unet import UNet, AttU_Net


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
        # self.fc_gat_out = nn.Linear(gat_configs['dim_model'], 1)

        unet_configs = configs['unet']
        self.unet = AttU_Net(config=unet_configs)

        self.xattn = CellGuidedCrossAttention(unet_channels=unet_configs['output_ch'],
                                              q_dim=gat_configs['dim_model'],
                                              d_model=64,
                                              n_heads=4,
                                              out_dim=1,
                                              add_xy_to_q=False,
                                              xy_dim=2,
                                              use_pos_enc=True)

    def reset_parameters(self):
        """Reset parameters of the model."""
        torch.nn.init.xavier_uniform_(self.fc_gat_out.weight)
        # TODO: since the classes are not balanced, the weights can be initialized as pos/total

    def forward(self, x, mask):
        x_gwe = self.gw_encoder(x)
        x_te = self.temporal_encoder(x)

        gat_out = self.gat_layer(x_te, x_gwe, x)
        # gat_out = self.fc_gat_out(gat_out)

        unet_out = self.unet(x)

        cell_xy = x['hidden_ogm_cells']
        fused_cells, attn_maps = self.xattn(unet_out, gat_out, cell_xy=cell_xy, query_mask=mask)  # fused_cells: (B,M,256)

        return fused_cells
