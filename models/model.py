import torch
from torch.functional import
from torch_geometric.nn import HeteroDictLinear, HGTConv
from torch_geometric.typing import EdgeType, NodeType


MetaData = tuple[list[str], list[tuple[str, str, str]]]


# TODO - We need 'meta_data' to be the same each time. It should be the most complex form of ZXMatchDiagram metadata().
class Model(torch.nn.Module):
    def __init__(self, meta_data: MetaData, hidden_channels: int, out_channels: int, num_heads: int, num_layers: int):
        super().__init__()
        self.heads, self.out_channels = num_heads, out_channels
        self.lin_dict0 = HeteroDictLinear(-1, hidden_channels, types=meta_data[0])
        self.convs = torch.nn.ModuleList([HGTConv(hidden_channels, hidden_channels, meta_data,
                                                  num_heads, group='sum') for _ in range(num_layers)])
        self.lin_dict1 = HeteroDictLinear(hidden_channels, out_channels, types=meta_data[0])

    def forward(self, x_dict: dict[NodeType, torch.Tensor], edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[
            str, torch.Tensor]:
        x_dict = {ntype: torch.functional.()(tensor.relu_()) for ntype, tensor in self.lin_dict0(x_dict).items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return self.lin_dict1(x_dict)
