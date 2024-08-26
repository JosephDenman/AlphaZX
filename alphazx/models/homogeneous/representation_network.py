from typing import Any

import torch
import torch.nn as nn

from alphazx.models.homogeneous.gps import GPS


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
                 num_layers: int,
                 bias: bool,
                 num_attn_heads: int,
                 attn_type: str,
                 attn_kwargs: dict[str, Any],
                 mlp_hidden_channels: int) -> None:
        super(RepresentationNetwork, self).__init__()
        self.gps = GPS(num_node_types * num_possible_phases,
                       node_embedding_out_channels,
                       node_out_channels,
                       num_edge_embeddings,
                       edge_embedding_out_channels,
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

    def sparse_parameters(self):
        return self.gps.sparse_parameters()

    def dense_parameters(self):
        return self.gps.dense_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor,
                pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gps(x, edge_index, edge_attr, batch, pe)
