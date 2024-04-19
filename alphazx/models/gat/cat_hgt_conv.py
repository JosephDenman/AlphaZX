from typing import Optional, Dict

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import Aggregation, HeteroConv, GATv2Conv
from torch_geometric.typing import EdgeType, NodeType

from alphazx.diagram.diagram_generators import clifford_pyg_hdata_zx_diagram, clifford_pyg_hdata_zx_match_diagram, \
    clifford_zx_diagram, clifford_pyg_zx_diagram, clifford_pyg_zx_match_diagram
from alphazx.diagram.match import SIMPLE_EDGE_METADATA, NODE_METADATA, SIMPLE_NODE_METADATA
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram
from alphazx.models.utils import cat_aggregate

torch.use_deterministic_algorithms(True)


def base_node_subgraph(zx_match_diagram_hdata: pyg.data.HeteroData) -> pyg.data.HeteroData:
    subgraph_dict = {}
    for ntype in NODE_METADATA:
        subgraph_dict[ntype] = torch.squeeze(
            torch.full(zx_match_diagram_hdata[ntype]['x'].size(), ntype in SIMPLE_NODE_METADATA, dtype=torch.bool), dim=-1)
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
        print('agg.x = ', x)
        print('agg.squeeze(x) = ', torch.squeeze(x))
        return cat_aggregate(torch.squeeze(x), index)


class EdgeAttn(torch.nn.Module):

    def __init__(self):
        super(EdgeAttn, self).__init__()
        self.hetero_conv = HeteroConv({
            etype: GATv2Conv(1, 1, heads=1, concat=False, add_self_loops=False, aggr=ConcatAggregation()) for etype in
            SIMPLE_EDGE_METADATA
        }, 'sum')

    def forward(self, x_dict: Dict[NodeType, torch.Tensor], edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[
            NodeType, torch.Tensor]:
        return self.hetero_conv(x_dict, edge_index_dict)


zx_match_diagram = clifford_pyg_hdata_zx_match_diagram(30, 30)
print('zx_match_diagram = ', zx_match_diagram)

edge_attn = EdgeAttn()
print(edge_attn(zx_match_diagram.x_dict, zx_match_diagram.edge_index_dict))
