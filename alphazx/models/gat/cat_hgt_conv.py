from typing import Optional, Dict

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import Aggregation, GATv2Conv, HeteroConv
from torch_geometric.typing import EdgeType, NodeType

from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import SIMPLE_EDGE_METADATA, NODE_METADATA, SIMPLE_NODE_METADATA
from alphazx.models.gat.custom_gat_v2 import CustomGATv2Conv


def base_node_subgraph(zx_match_diagram_hdata: pyg.data.HeteroData) -> pyg.data.HeteroData:
    subgraph_dict = {}
    for ntype in NODE_METADATA:
        subgraph_dict[ntype] = torch.squeeze(
            torch.full(zx_match_diagram_hdata[ntype]['x'].size(), ntype in SIMPLE_NODE_METADATA, dtype=torch.bool),
            dim=-1)
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
        x = torch.squeeze(x, dim=2)
        index_count = torch.bincount(index)
        fill_count = index_count.max() - index_count
        fill_zeros = torch.zeros_like(x[0]).repeat(fill_count.sum(), *([1] * (len(x.shape) - 1)))
        fill_index = torch.arange(0, fill_count.shape[0]).repeat_interleave(fill_count)
        index_ = torch.cat([index, fill_index], dim=0)
        x_ = torch.cat([x, fill_zeros], dim=0)
        x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
        return x_


class HeteroEdgeAttn(torch.nn.Module):

    def __init__(self):
        super(HeteroEdgeAttn, self).__init__()
        self.hetero_conv = HeteroConv({
            etype: GATv2Conv(1, 1, heads=1, concat=False, add_self_loops=False, aggr=ConcatAggregation()) for etype in
            SIMPLE_EDGE_METADATA
        })

    def forward(self, x_dict: Dict[NodeType, torch.Tensor], edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[
        NodeType, torch.Tensor]:
        return self.hetero_conv(x_dict, edge_index_dict)


class HomoEdgeAttn(torch.nn.Module):

    def __init__(self, in_channels: int):
        super(HomoEdgeAttn, self).__init__()
        self.conv = CustomGATv2Conv(in_channels, 1, heads=1, concat=False, add_self_loops=False,
                                    aggr=ConcatAggregation())
        # self.hetero_conv = HeteroConv({
        #     etype: CustomGATv2Conv(in_channels, 1, heads=1, concat=False, add_self_loops=False, aggr=ConcatAggregation()) for etype in
        #     SIMPLE_EDGE_METADATA
        # }, 'sum')

    # def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    #     print('x = ', x)
    #     print('edge_index = ', edge_index)
    #     return self.conv(x, edge_index)

    def forward(self, x_dict: torch.Tensor,
                edge_index_dict: torch.Tensor) -> torch.Tensor:
        return self.conv(x_dict, edge_index_dict)


zx_match_diagram = clifford_pyg_zx_match_diagram(5, 5)
edge_attn = HomoEdgeAttn(2)
output = edge_attn(zx_match_diagram.x, zx_match_diagram.edge_index)
