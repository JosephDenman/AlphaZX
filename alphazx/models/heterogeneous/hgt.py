import torch
import torch_geometric as pyg
from torch_geometric.nn import HGTConv
from torch_geometric.typing import Metadata, NodeType, EdgeType


class HGT(torch.nn.Module):
    def __init__(self,
                 metadata: Metadata,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 num_layers: int):
        super(HGT, self).__init__()
        self.lin_dict_in = pyg.nn.HeteroDictLinear(in_channels, hidden_channels, metadata[0])
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
        self.lin_dict_out = pyg.nn.HeteroDictLinear(hidden_channels, out_channels, metadata[0])

    def forward(self,
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[NodeType, torch.Tensor]:
        x_dict = self.lin_dict_in(x_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict = self.lin_dict_out(x_dict)
        return x_dict
