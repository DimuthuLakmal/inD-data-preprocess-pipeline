import torch
import torch.nn as nn

from src.models.vision.convnext_map_encoder import (
    ConvNeXtMapEncoder,
)

from src.models.vision.cell_map_sampler import (
    CellConditionedMultiScaleSampler,
)

from src.models.vision.cell_node_encoder import (
    CellNodeEncoder,
)


class SemanticContextEncoder(nn.Module):

    def __init__(
        self,
        num_semantic_classes: int,
        map_context_dim: int = 64,
        cell_node_dim: int = 64,
        projection_dim: int = 32,
        pretrained: bool = True,
        stem_init: str = "random",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.map_encoder = (
            ConvNeXtMapEncoder(
                in_channels=(
                    num_semantic_classes
                ),
                pretrained=pretrained,
                stem_init=stem_init,
            )
        )

        self.sampler = (
            CellConditionedMultiScaleSampler(
                in_channels=(
                    self.map_encoder
                    .out_channels
                ),
                projection_dim=(
                    projection_dim
                ),
                output_dim=(
                    map_context_dim
                ),
                dropout=dropout,
            )
        )

        self.cell_encoder = (
            CellNodeEncoder(
                map_context_dim=(
                    map_context_dim
                ),
                output_dim=(
                    cell_node_dim
                ),
                dropout=dropout,
            )
        )

    def forward(
        self,
        semantic_map: torch.Tensor,
        cell_xy: torch.Tensor,
        cell_mask=None,
    ):
        """
        semantic_map:
            [B,K,H,W]

        cell_xy:
            [B,Nc,2]
        """

        pyramid = self.map_encoder(
            semantic_map
        )

        map_context = self.sampler(
            pyramid=pyramid,
            cell_xy=cell_xy,
            cell_mask=cell_mask,
        )

        cell_embedding = (
            self.cell_encoder(
                cell_xy=cell_xy,
                map_context=map_context,
            )
        )

        return {
            "cell_embedding":
                cell_embedding,

            "map_context":
                map_context,

            "map_pyramid":
                pyramid,
        }