import torch
import torch.nn as nn
from torch_geometric.typing import NodeType, EdgeType

from alphazx.diagram.match import METADATA


class RepresentationNetwork(nn.Module):

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 embed_channels: int,
                 attn_heads: int,
                 layers: int) -> None:
        super(RepresentationNetwork, self).__init__()
        self.hgt = HGT(METADATA, input_channels, hidden_channels, embed_channels, attn_heads, layers)

    def forward(self,
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.hgt(x_dict, edge_index_dict)
