import math
from abc import ABC
from typing import Optional, Dict, Union

import torch
from torch import Tensor
from torch_geometric.nn import HGTConv, Aggregation
from torch_geometric.typing import Adj, Metadata
from torch_geometric.utils import softmax

from alphazx.diagram.diagram_generators import clifford_pyg_hetero_zx_match_diagram
from alphazx.diagram.match import METADATA
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
        return cat_aggregate(x, index, max_num_elements)


class EdgeAttn(HGTConv, ABC):

    def __init__(self,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 metadata: Metadata,
                 heads: int = 1,
                 **kwargs):
        super(EdgeAttn, self).__init__(in_channels, out_channels, metadata, heads, **kwargs)

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        print('alpha = ', alpha)
        print('alpha_view = ', alpha.view(-1, self.heads, 1))
        out = v_j * alpha.view(-1, self.heads, 1)
        print('out = ', out)
        print('out_view = ', out.view(-1, self.out_channels))
        return out.view(-1, self.out_channels)


zx_match_diagram_hdata = clifford_pyg_hetero_zx_match_diagram(2, 2, True)
edge_attn = EdgeAttn(in_channels=1, out_channels=10, metadata=METADATA, heads=1)
print(edge_attn(zx_match_diagram_hdata.x_dict, zx_match_diagram_hdata.edge_index_dict))
