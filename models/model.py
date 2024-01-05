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


"""

There are three obvious possible representations for the state graph. Each has implications for the input to the 
prediction model.

------------------------------------------------------------------------------------------------------------------------
# Homogenous Direct Diagram Encoding

The first approach is to directly convert the `ZXDiagram` to a `Data` object. This requires that node feature dimensions 
are the same for all node types. The representations are the concatenation of an integer representing the type of the
vertex, the phase of the vertex (zero for input and output vertices), and the degree of the vertex.

    i_feat = [0] + [0] + [1]

    o_feat = [1] + [0] + [1]
    
    z_feat = [2] + [phase] + [degree]
    
    x_feat = [3] + [phase] + [degree]

The vertex types are passed to an embedding layer. This representation allows using GPS, which allows positional/structural 
encoding techniques that are generally not available to HGNNs.

The main obstacle the direct representation faces is that of action encodings. In AlphaZX, an action is a rewrite match
identifying subgraph of G. The node outputs of the GNN must be transformed into a probability distribution over matches. 
In contrast to the match diagram encodings, the direct encoding requires a mechanism to assemble match-level outputs 
from node-level outputs in an end-to-end fashion. The mechanism must also support batching. One possibility is to provide 
the match sets as inputs to the model:

    def forward(x, 
                edge_index, 
                edge_attr, 
                flz_matches, 
                flx_matches, 
                frz_matches, frx_matches, bl_matches, br_matches, ylz_matches, yrz_matches, ylx_matches, yrx_matches)
        return value, policy_params
                        
Each match `m` is a mask that selects the elements of `x` contained in `m`. The output features can then be passed to
a pooling layer:

    `h_S_i = Pool_t(S)(h[m])`

# Policy


    
------------------------------------------------------------------------------------------------------------------------
# Homogenous Match Diagram Encoding

    flz_feat = []
    
    flx_feat = []
    
    frz_feat = []
    
    frx_feat = []
    
    bl_feat = []
    
    br_feat = []
    
    ylz_feat = []
    
    yrz_feat = []
    
    ylx_feat = []
    
    yrx_feat = []

------------------------------------------------------------------------------------------------------------------------

Observations:

    O1: Z or X gates that (1) have zero phase, (2) are the only gates on a circuit layer, and (3) belong to a circuit layer
       that does not interact with other circuit layers can be removed from the graph. This is not true when the objective is to minimize the number
        of gates with a particular phase set.

    O2: Z or X gates that satisfy only (O1.2) and (O1.3) above can be excluded from the model, since expanding these spiders 
        always produce a circuit that is sub-optimally larger. This is not true when the objective is to minimize the number
        of gates with a particular phase set.

    O3: Single isolated vertices in a match diagram always describe vertices satisfying at least (O1.2) and (O1.3).
       
Challenges:

    P1: ZX diagrams typically have multiple connected components due to the way they are constructed. Circuits are essentially
        a set of stacked, disconnected, horizontal lines, each of which begins with an input node and ends with an output 
        node. To these horizontal lines, a small number of vertical lines are added at some nodes in the middle. As a result
        of diagrams not being completely connected, messages will not propagate to all nodes. Messages are only propagated
        within connected components.
       
    S1: For each connected component C_i in G, add a component-level vertex c_i of type CO and connect it to each vertex 
        in C_i with an edge of type e_CO-VE. Connect each component-level vertex c_i to each other component-level vertex
        c_j with an edge of type e_CO-CO. By connecting component-level vertices to each other rather than a single top-level
        vertex, messages coming from different components can be distinguished.
       
    S2: Given node representations produced from running GNNs on each component (no additional logic is required to restrict
        the GNN to each component, since this is handled by disconnectedness) apply a transformer encoder to all final node
        representations. Transformer encoders work pair-wise globally, meaning that features are not propagated through
        component-level vertex bottlenecks.


"""