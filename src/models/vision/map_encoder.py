import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from models.vision import base_models


class MapEncoder(nn.Module):
    """Encodes map, may output a global feature, feature map, or both."""
    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            global_feature_dim=None,
            grid_feature_dim=None,
    ) -> None:
        super(MapEncoder, self).__init__()
        self.return_global_feat = global_feature_dim is not None
        self.return_grid_feat = grid_feature_dim is not None
        encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim
        )
        self.input_image_shape = input_image_shape
        # build graph for extracting intermediate features
        feat_nodes = {
            'map_model.layer1': 'layer1',
            'map_model.layer2': 'layer2',
            'map_model.layer3': 'layer3',
            'map_model.layer4': 'layer4',
            'map_model.fc' : 'fc',
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)
        # if self.return_grid_feat:
        #     encoder_channels = list(encoder.feature_channels().values())
        #     input_shape_scale = encoder.feature_scales()["layer4"]
        #     self.decoder = MapGridDecoder(
        #         input_shape=(encoder_channels[-1], input_image_shape[1]*input_shape_scale, input_image_shape[2]*input_shape_scale),
        #         encoder_channels=encoder_channels[:-1],
        #         output_channel=grid_feature_dim,
        #         batchnorm=True,
        #     )
        self.encoder_feat_scales = list(encoder.feature_scales().values())

    def feat_map_out_dim(self, H, W):
        dim_scale = self.encoder_feat_scales[-4] # decoder has 3 upsampling
        return (H * dim_scale, W * dim_scale )

    def forward(self, x, encoder_feats=None):
        if encoder_feats is None:
            encoder_feats = self.encoder_heads(x)
        fc_out = encoder_feats['fc'] if self.return_global_feat else None
        encoder_feats = [encoder_feats[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
        feat_map_out = None
        # if self.return_grid_feat:
        #     feat_map_out = self.decoder.forward(feat_to_decode=encoder_feats[-1],
        #                                         encoder_feats=encoder_feats[:-1])
        return fc_out, feat_map_out