from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg
from alphazx.models.homogeneous.gps import GPS


class RepresentationNetwork(nn.Module):

    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 embedding_out_channels: int,
                 node_out_channels: int,
                 edge_in_channels: int,
                 edge_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 num_layers: int,
                 bias: bool,
                 num_attn_heads: int,
                 attn_type: str,
                 attn_kwargs: dict[str, Any],
                 mlp_hidden_channels: int) -> None:
        super(RepresentationNetwork, self).__init__()
        self.gps = GPS(num_node_types * num_possible_phases,
                       embedding_out_channels,
                       node_out_channels,
                       edge_in_channels,
                       edge_out_channels,
                       pe_in_channels,
                       pe_out_channels,
                       num_layers,
                       bias,
                       num_attn_heads,
                       attn_type,
                       attn_kwargs,
                       mlp_hidden_channels)

    def reset_parameters(self):
        self.gps.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        return self.gps(x, edge_index, batch, pe)
