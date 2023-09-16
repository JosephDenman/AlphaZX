from collections import defaultdict
from typing import Literal

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, DataLoader, InMemoryDataset
from torch_geometric.nn import HeteroDictLinear, HGTConv

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

    def forward(self, hdata: HeteroData) -> HeteroData:
        x_dict = hdata.collect('x')
        # print('x_dict = ', x_dict)
        edge_index_dict = hdata.collect('edge_index')
        # print('edge_index_dict = ', edge_index_dict)
        x_dict = self.linear_dict_0(x_dict)
        for ntype, out in x_dict.items():
            x_dict[ntype] = F.relu(out)
        for conv in self.hgt_convolutions:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict = self.linear_dict_1(x_dict)
        for ntype, out in x_dict.items():
            hdata[ntype]['x'] = out
        return hdata


class MatchModule(torch.nn.Module):
    def __init__(self,
                 metadata: MetaData,
                 hgt_params: HGTParams):
        super().__init__()
        self.metadata = metadata
        self.hgt_params = hgt_params
        self.hgt_module = HGT(self.metadata, *self.hgt_params)

    # TODO: Add pooling layer
    def forward(self, hdata: HeteroData) -> HeteroData:
        return self.hgt_module(hdata)


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
            key: HGT(diagram_metadata, *self.match_hgt_params_dict[key]) for key, match_meta_data in
            self.match_metadata_dict.items()
        })

    def forward(self, hdata: HeteroData) -> dict[MatchType, torch.Tensor]:

        print('hdata = ', hdata)

        hdata = self.diagram_module(hdata)

        print('new_hdata = ', hdata)

        match_x_dicts = defaultdict(list)
        for match_type, matches in hdata['matches'].items():
            for match in matches:
                print('match = ', match)
                sub_hdata = hdata.subgraph(match)
                del sub_hdata['matches']
                print('sub_new_hdata = ', sub_hdata)
                match_x_dicts[match_type].append(self.match_module_dict[match_type](sub_hdata))

        match_x_outs = defaultdict()
        for match_type, match_out in match_x_dicts.items():
            match_x_outs[match_type] = torch.stack(match_out, dim=0)

        return match_x_outs
