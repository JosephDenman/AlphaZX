from collections import defaultdict
from typing import Literal

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroDictLinear, HGTConv
from torch_geometric.typing import NodeType, EdgeType

MetaData = tuple[list[str], list[tuple[str, str, str]]]

MatchType = Literal['f_right_z'] | Literal['f_right_x'] | Literal['f_left_z'] | Literal['f_left_x'] \
            | Literal['b_right'] | Literal['b_left'] | Literal['y_left_z'] | Literal['y_left_x'] \
            | Literal['y_right_z'] | Literal['y_right_x']

HGTParams = tuple[int, int, int, int, str]


class HGT(torch.nn.Module):
    def __init__(self,
                 metadata: MetaData,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 num_layers: int,
                 group: str = 'sum'):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.group = group
        self.linear_dict_0 = HeteroDictLinear(-1, self.hidden_channels, types=self.metadata[0])
        self.hgt_convolutions = torch.nn.ModuleList([HGTConv(self.hidden_channels, self.hidden_channels, self.metadata,
                                                             self.num_heads, group=self.group) for _ in
                                                     range(self.num_layers)])
        self.linear_dict_1 = HeteroDictLinear(self.hidden_channels, self.out_channels, types=self.metadata[0])

    def forward(self,
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_dict = {ntype: F.relu(tensor) for ntype, tensor in self.linear_dict_0(x_dict).items()}
        for conv in self.hgt_convolutions:
            x_dict = conv(x_dict, edge_index_dict)
        return self.linear_dict_1(x_dict)


class MatchModule(torch.nn.Module):
    def __init__(self,
                 metadata: MetaData,
                 hgt_params: HGTParams):
        super().__init__()
        self.metadata = metadata
        self.hgt_params = hgt_params
        self.hgt_module = HGT(self.metadata, *self.hgt_params)
        self.pooling_module = None

    def forward(self,
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> torch.Tensor:
        return self.pooling_module(self.hgt_module(x_dict, edge_index_dict), edge_index_dict)


class HGTPolicy(torch.nn.Module):
    def __init__(self,
                 diagram_metadata: MetaData,
                 diagram_hgt_params: HGTParams,
                 match_metadata_dict: dict[MatchType, MetaData],
                 match_hgt_params_dict: dict[MatchType, HGTParams]):
        super().__init__()
        self.diagram_metadata = diagram_metadata
        self.diagram_hgt_params = diagram_hgt_params
        self.match_metadata_dict = match_metadata_dict
        self.match_hgt_params_dict = match_hgt_params_dict
        self.diagram_module = HGT(self.diagram_metadata, *self.diagram_hgt_params)
        self.match_module_dict = torch.nn.ModuleDict({
            key: HGT(match_meta_data, *self.match_hgt_params_dict[key]) for key, match_meta_data in
            self.match_metadata_dict.items()
        })

    # TODO: Remove type annotations
    # TODO: Add matches as attributes on HeteroData.
    #       IDEA:
    # BLeftMatch(15, 17) => b_left_match -> 2, 8

    def forward(self,                   # Tensor[Tensor[2,n]] (would need to be squarified)
                matches: dict[MatchType, dict[NodeType, torch.Tensor]],
                x_dict: dict[NodeType, torch.Tensor],
                edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[MatchType, torch.Tensor]:

        x_dict = self.diagram_module(x_dict, edge_index_dict)

        # TODO: Does this keep device and computational graph of embedding tensors?
        embedded_node_mapping = {ntype: {'x': embeddings} for ntype, embeddings in x_dict.items()}
        edge_index_mapping = {etype: {'edge_index': edge_index} for etype, edge_index in edge_index_dict}
        hdata = HeteroData(embedded_node_mapping.update(edge_index_mapping))

        match_x_dicts = defaultdict(list)
        for match_type, matches in matches.items():
            for match in matches:
                match_hdata = hdata.subgraph(match)
                match_x_dicts[match_type].append(
                    self.match_module_dict[match_type](match_hdata.collect('x'), match_hdata.collect('edge_index')))

        match_embeddings_dict = {match_type: torch.stack(match_embeddings, dim=0) for
                                 match_type, match_embeddings in match_x_dicts.items()}

        return match_embeddings_dict
