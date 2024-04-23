from typing import Optional, Dict

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import Aggregation, GATv2Conv, HeteroConv
from torch_geometric.typing import EdgeType, NodeType

from alphazx.diagram.diagram_generators import clifford_pyg_hdata_zx_match_diagram, clifford_pyg_zx_match_diagram
from alphazx.diagram.match import SIMPLE_EDGE_METADATA, NODE_METADATA, SIMPLE_NODE_METADATA, EDGE_METADATA
from alphazx.models.gat.custom_gat_v2 import CustomGATv2Conv
from alphazx.models.utils import cat_aggregate

torch.use_deterministic_algorithms(True)


def base_node_subgraph(zx_match_diagram_hdata: pyg.data.HeteroData) -> pyg.data.HeteroData:
    subgraph_dict = {}
    for ntype in NODE_METADATA:
        subgraph_dict[ntype] = torch.squeeze(
            torch.full(zx_match_diagram_hdata[ntype]['x'].size(), ntype in SIMPLE_NODE_METADATA, dtype=torch.bool),
            dim=-1)
    print('subgraph_dict = ', subgraph_dict)
    return zx_match_diagram_hdata.subgraph(subgraph_dict)


class ConcatAggregation(Aggregation):
    def __init__(self):
        super(ConcatAggregation, self).__init__()

    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = 0,
            max_num_elements: Optional[int] = None,
    ) -> Tensor:
        return cat_aggregate(x, index)


class HeteroEdgeAttn(torch.nn.Module):

    def __init__(self):
        super(HeteroEdgeAttn, self).__init__()
        self.hetero_conv = HeteroConv({
            etype: GATv2Conv(1, 1, heads=1, concat=False, add_self_loops=False, aggr=ConcatAggregation()) for etype in
            SIMPLE_EDGE_METADATA
        }, 'sum')

    def forward(self, x_dict: Dict[NodeType, torch.Tensor], edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[
            NodeType, torch.Tensor]:
        return self.hetero_conv(x_dict, edge_index_dict)


class HomoEdgeAttn(torch.nn.Module):

    def __init__(self, in_channels: int):
        super(HomoEdgeAttn, self).__init__()
        self.conv = CustomGATv2Conv(in_channels, 1, heads=1, concat=False, add_self_loops=False, aggr=ConcatAggregation())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        print('x = ', x)
        print('edge_index = ', edge_index)
        return self.conv(x, edge_index)


zx_match_diagram = clifford_pyg_zx_match_diagram(30, 30)
# zx_match_diagram.sort(True)
edge_attn = HomoEdgeAttn(2)

print(edge_attn(zx_match_diagram.x, zx_match_diagram.edge_index))
