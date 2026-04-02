from typing import Any

import torch
import torch.nn as nn

from alphazx.models.homogeneous.gps import GPS, FeatureEmbeddingLayer


class RepresentationNetwork(nn.Module):

    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 node_embedding_out_channels: int,
                 node_out_channels: int,
                 num_edge_embeddings: int,
                 edge_embedding_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 bias: bool = True,
                 gps_num_layers: int = 2,  # Reduced from 4 to prevent over-smoothing
                 gps_heads: int = 4,
                 gps_dropout: float = 0.1,
                 gps_act: str = 'relu',
                 gps_act_kwargs: dict[str, Any] = None,
                 gps_norm: str = 'batch_norm',
                 gps_norm_kwargs: dict[str, Any] = None,
                 gps_attn_type: str = 'multihead',
                 gps_attn_kwargs: dict[str, Any] = None,
                 mlp_hidden_channels: int = 128,
                 mlp_num_layers: int = 2) -> None:
        super(RepresentationNetwork, self).__init__()
        self.emb = FeatureEmbeddingLayer(num_node_types * num_possible_phases, node_embedding_out_channels,
                                         num_edge_embeddings, edge_embedding_out_channels, pe_in_channels,
                                         pe_out_channels, bias)
        self.gps = GPS(node_embedding_out_channels + pe_out_channels,
                       node_out_channels,
                       edge_embedding_out_channels,
                       gps_num_layers,
                       gps_heads,
                       gps_dropout,
                       gps_act,
                       gps_act_kwargs,
                       gps_norm,
                       gps_norm_kwargs,
                       gps_attn_type,
                       gps_attn_kwargs,
                       mlp_hidden_channels,
                       mlp_num_layers)

    def reset_parameters(self):
        self.emb.reset_parameters()
        self.gps.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor,
                pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_attr = self.emb(x, edge_attr, pe)
        return self.gps(x, edge_index, edge_attr, batch), edge_attr
