from typing import Optional, Dict

import torch
from torch import Tensor
from torch_geometric.nn import HGTConv, Aggregation, HeteroConv, GATv2Conv
from torch_geometric.typing import EdgeType, NodeType

from alphazx.diagram.constants import B_ETYPE_NAME
from alphazx.diagram.diagram_generators import clifford_pyg_hetero_zx_match_diagram
from alphazx.diagram.match import METADATA, FRightZMatch, FRightXMatch
from alphazx.models.utils import cat_aggregate


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
        return cat_aggregate(torch.squeeze(x), index)


class EdgeAttn(torch.nn.Module):

    def __init__(self):
        super(EdgeAttn, self).__init__()
        self.hetero_conv = HeteroConv({
            (FRightZMatch.abbrev, B_ETYPE_NAME, FRightZMatch.abbrev): GATv2Conv(1, 8, concat=False, add_self_loops=False, aggr=ConcatAggregation()),
            (FRightZMatch.abbrev, B_ETYPE_NAME, FRightXMatch.abbrev): GATv2Conv(1, 8, concat=False, add_self_loops=False, aggr=ConcatAggregation()),
            (FRightXMatch.abbrev, B_ETYPE_NAME, FRightZMatch.abbrev): GATv2Conv(1, 8, concat=False, add_self_loops=False, aggr=ConcatAggregation()),
            (FRightXMatch.abbrev, B_ETYPE_NAME, FRightXMatch.abbrev): GATv2Conv(1, 8, concat=False, add_self_loops=False, aggr=ConcatAggregation()),
        }, 'cat')

    def forward(self, x_dict: Dict[NodeType, torch.Tensor], edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[
            NodeType, torch.Tensor]:
        return self.hetero_conv(x_dict, edge_index_dict)


zx_match_diagram_hdata = clifford_pyg_hetero_zx_match_diagram(100, 100, True)
edge_attn = EdgeAttn()
print('x_dict = ', zx_match_diagram_hdata.x_dict)
print('edge_index_dict = ', zx_match_diagram_hdata.edge_index_dict)
print(edge_attn(zx_match_diagram_hdata.x_dict, zx_match_diagram_hdata.edge_index_dict))
