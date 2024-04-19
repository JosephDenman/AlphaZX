from typing import Optional, Dict

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import Aggregation, HeteroConv, GATv2Conv
from torch_geometric.typing import EdgeType, NodeType

from alphazx.diagram.diagram_generators import clifford_pyg_hetero_zx_diagram, clifford_pyg_hetero_zx_match_diagram, \
    clifford_zx_diagram
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
        return cat_aggregate(x, index)


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


zx_diagram = clifford_zx_diagram(30, 30, True)
zx_diagram_x_nodes = zx_diagram.x_nodes()
zx_diagram_z_nodes = zx_diagram.z_nodes()
zx_diagram_b_nodes = zx_diagram.b_nodes()

# TODO: Edge indices are incorrect in the hdata result. No edges show up. Do the edge indices show up for ZXMatchDiagram?
edge_attn = EdgeAttn()
zx_diagram = to_zx_match_diagram(zx_diagram).to_pyg_hdata()
print(edge_attn(zx_diagram.x_dict, zx_diagram.edge_index_dict))
