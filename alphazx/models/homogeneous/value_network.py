from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg
from alphazx.models.homogeneous.gps import GPS


class ValueNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 node_embedding_channels: int,
                 gps_channels: int,
                 gps_edge_in_channels: int,
                 gps_edge_out_channels: int,
                 gps_pe_in_channels: int,
                 gps_pe_out_channels: int,
                 gps_num_layers: int,
                 gps_bias: bool,
                 gps_num_attn_heads: int,
                 gps_attn_type: str,
                 gps_attn_kwargs: dict[str, Any],
                 gps_mlp_hidden_channels: int,
                 gmt_num_encoder_blocks: int,
                 gmt_num_heads: int,
                 gmt_layer_norm: bool,
                 gmt_dropout: float) -> None:
        super(ValueNetwork, self).__init__()
        self.gps = GPS(num_node_types * num_possible_phases,
                       gps_channels,
                       node_embedding_channels,
                       gps_edge_in_channels,
                       gps_edge_out_channels,
                       gps_pe_in_channels,
                       gps_pe_out_channels,
                       gps_num_layers,
                       gps_bias,
                       gps_num_attn_heads,
                       gps_attn_type,
                       gps_attn_kwargs,
                       gps_mlp_hidden_channels)
        self.pool = pyg.nn.GraphMultisetTransformer(node_embedding_channels, 1, gmt_num_encoder_blocks, gmt_num_heads, gmt_layer_norm, gmt_dropout)

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        x = self.gps(data.x, data.pe, data.edge_index, data.batch)
        x = self.pool(x, data.batch)
        return x
