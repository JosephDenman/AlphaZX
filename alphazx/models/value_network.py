import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.typing import NodeType, EdgeType

from alphazx.diagram.match import METADATA
from alphazx.models.hgt import HGT


class ValueNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hgt_hidden_dim: int,
                 hgt_out_dim: int,
                 hgt_heads: int,
                 hgt_layers: int,
                 encoder_blocks: int,
                 encoder_attn_heads: int,
                 encoder_feedforward_dim: int,
                 encoder_dropout: float,
                 encoder_activation: str,
                 encoder_layer_norm_eps: float,
                 encoder_bias: bool,
                 encoder_norm_first: bool,
                 pooling_encoder_blocks: int,
                 pooling_heads: int,
                 pooling_layer_norm: bool,
                 pooling_dropout: float):
        super(ValueNetwork, self).__init__()
        self.hgt = HGT(METADATA, input_dim, hgt_hidden_dim, hgt_out_dim, hgt_heads, hgt_layers)
        self.node_encoder = NodeEncoder(encoder_blocks, hgt_out_dim, encoder_attn_heads, encoder_feedforward_dim,
                                        encoder_dropout, encoder_activation, encoder_layer_norm_eps, encoder_bias,
                                        encoder_norm_first)
        self.pool = pyg.nn.GraphMultisetTransformer(hgt_out_dim, 1, pooling_encoder_blocks, pooling_heads,
                                                    pooling_layer_norm, pooling_dropout)

    def forward(self,
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> torch.Tensor:
        x_dict = self.hgt(x_dict, edge_index_dict)
        x_dict = self.node_encoder(x_dict, edge_index_dict)
        # TODO: The next two assignments are probably inefficient.
        data = pyg.data.HeteroData(x_dict.update(edge_index_dict)).to_homogeneous(node_attrs=['phase'],
                                                                                  add_node_type=True,
                                                                                  add_edge_type=True)
        data = data.sort(False)
        h = self.pool(data.x, data.edge_index)
        return h