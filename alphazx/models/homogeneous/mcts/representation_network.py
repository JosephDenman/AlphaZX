from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg
from alphazx.models.homogeneous.gps import GPS


class RepresentationNetwork(nn.Module):

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
                 gps_mlp_hidden_channels: int) -> None:
        super(RepresentationNetwork, self).__init__()
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

    def reset_parameters(self):
        self.gps.reset_parameters()

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        return self.gps(data.x, data.pe, data.edge_index, data.batch)
