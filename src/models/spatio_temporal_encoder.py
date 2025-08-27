import torch
from torch import nn

from src.models.gat.gat_layer import GATLayer
from src.models.transformer.cell_guided_cross_attention import CellGuidedCrossAttention
from src.models.transformer.expert_gating import TwoExpertGatedFusion
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder
from src.models.transformer.map_encoder_attn import MapEncoderAttention
from src.models.transformer.temporal_encoder import TemporalEncoder
from src.models.unet.unet import UNet, AttU_Net
from src.models.vision.map_encoder import MapEncoder


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
        self.fc_gat_out = nn.Linear(64, 1)

        unet_configs = configs['unet']
        self.unet = AttU_Net(config=unet_configs)

        # self.xattn = CellGuidedCrossAttention(unet_channels=unet_configs['output_ch'],
        #                                       q_dim=gat_configs['dim_model'],
        #                                       d_model=64,
        #                                       n_heads=4,
        #                                       out_dim=1,
        #                                       add_xy_to_q=False,
        #                                       xy_dim=2,
        #                                       use_pos_enc=True)

        self.expert_gating = TwoExpertGatedFusion(64)

        self.map_encoder_atten = MapEncoderAttention(256, 32, d=64)

        self.map_encoder = MapEncoder(model_arch='resnet50', input_image_shape=(4, 224, 224), global_feature_dim=64)

    def reset_parameters(self):
        """Reset parameters of the model."""
        torch.nn.init.xavier_uniform_(self.fc_gat_out.weight)
        # TODO: since the classes are not balanced, the weights can be initialized as pos/total

    def forward(self, x, seq_mask=None):
        x_gwe = self.gw_encoder(x, seq_mask)
        x_te = self.temporal_encoder(x, seq_mask)

        gat_out = self.gat_layer(x_te, x_gwe, x)
        gat_out_fc = self.fc_gat_out(gat_out)

        # unet_out = self.unet(x)
        map_output = self.map_encoder(x)[0]

        ### These are just testing lines, not concrete implementations
        map_output = map_output.unsqueeze(dim=1).repeat_interleave(gat_out.shape[1], dim=1)

        fused = self.expert_gating(map_output, gat_out)
        gat_out_fc = self.fc_gat_out(fused)

        # # combine map and GAT features
        # combined = torch.cat([map_output, gat_out], dim=-1)
        # gat_out_fc = self.fc_gat_out(map_output)
        ### end testing lines

        cell_xy = x['hidden_ogm_cells']
        # fused_cells, attn_maps = self.xattn(map_output, gat_out, cell_xy=cell_xy,
        #                                     query_mask=mask)  # fused_cells: (B,M,256)

        # _, fused_cells = self.map_encoder_atten(map_output, gat_out)
        # fused_cells_fc = self.fc_gat_out(fused_cells)

        return gat_out_fc
