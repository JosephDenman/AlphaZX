import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroDictLinear, HGTConv
from torch_geometric.typing import EdgeType, NodeType

from matching.match import Match

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

    # TODO - Idea: Is it possible to pass a dictionary of dictionaries to represent the higher-order graph.
    #        (dict[NodeType, torch.Tensor], dict[EdgeType, torch.Tensor])
    """
    Run the original graph through a HGNN to get node embeddings.
    Take node embeddings as features in each subgraph.
    The graph of graphs: dict[NodeType, tuple[dict[NodeType, torch.Tensor], dict[EdgeType, torch.Tensor]]]
    """

    MatchType = str

    def forward(self, matches: dict[MatchType, list[dict[NodeType, torch.Tensor]]], hdata: HeteroData) -> dict[
        str, torch.Tensor]:
        x_dict = hdata.collect('x')
        edge_index_dict = hdata.collect('edge_index')

        # TODO: Factor this section into independent HGT module.
        x_dict = {ntype: F.relu(tensor) for ntype, tensor in self.lin_dict0(x_dict).items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict = self.lin_dict1(x_dict)

        for match_type, matches in matches.items():
            for match in matches:
                sub_hdata = hdata.subgraph(match)

        sub_hdata = [hdata.subgraph(match) for match_type, matches in matches.items()]
        return x_dict


"""

{ 
    'f_right_match' : {
        'z': {
            'embeddings': torch.tensor([[...], ..., [...]])
        },
        ('z', 'to', 'z'): torch.tensor(...)
    },
    'b_right_match' : {
        'z': {
            'embeddings': torch.tensor([[...], ..., [...]])
        },
        'x': {
            'embeddings': torch.tensor([[...], ..., [...]])
        },
        ('z', 'to', 'x'): torch.tensor(...),
        ('x', 'to', 'z'): torch.tensor(...),
    }
}


"""
