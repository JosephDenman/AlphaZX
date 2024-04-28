from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation, HeteroConv, GATv2Conv, Linear
from torch_geometric.typing import NodeType, EdgeType

from alphazx.diagram.diagram_generators import clifford_zx_match_diagram
from alphazx.diagram.match import SIMPLE_EDGE_METADATA, SIMPLE_NODE_METADATA, SIMPLE_METADATA
from alphazx.models.gat.custom_gat_v2 import CustomGATv2Conv


class ConcatAggregation(Aggregation):
    """
    Aggregates messages by concatenation. The order of the messages is given by the order of the nodes in the feature
    tensors of the graph. Each label in `index` corresponds to a destination node in the graph.
    """

    def __init__(self):
        super(ConcatAggregation, self).__init__()

    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            # This is just destination nodes - all input features with the same destination node are concatenated.
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = 0,
            max_num_elements: Optional[int] = None,
    ) -> Tensor:
        if x.size()[0] == 0:
            return x
        # x = torch.squeeze(x, dim=-1)
        index_count = torch.bincount(index)
        fill_count = index_count.max() - index_count
        fill_zeros = torch.zeros_like(x[0]).repeat(fill_count.sum(), *([1] * (len(x.shape) - 1)))
        fill_index = torch.arange(0, fill_count.shape[0]).repeat_interleave(fill_count)
        index_ = torch.cat([index, fill_index], dim=0)
        x_ = torch.cat([x, fill_zeros], dim=0)
        x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
        return x_


class HeteroEdgeAttn(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(HeteroEdgeAttn, self).__init__()
        # Attempt to use 'to_hetero' here.
        self.hetero_conv = HeteroConv({
            etype: GATv2Conv(in_channels, out_channels, heads=1, concat=False, add_self_loops=False,
                             aggr=ConcatAggregation()) for etype in
            SIMPLE_EDGE_METADATA
        })
        self.han_conv = HANConv(in_channels, out_channels, SIMPLE_METADATA, aggr=ConcatAggregation())

    def forward(self, x_dict: dict[NodeType, torch.Tensor], edge_index_dict: dict[EdgeType, torch.Tensor]) -> dict[
        NodeType, torch.Tensor]:
        print('x_dict = ', x_dict)
        print('edge_index_dict = ', edge_index_dict)
        return self.han_conv(x_dict, edge_index_dict)


class HomoEdgeAttn(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(HomoEdgeAttn, self).__init__()
        self.conv = CustomGATv2Conv(in_channels, out_channels, heads=2, concat=False, add_self_loops=False,
                                    aggr=ConcatAggregation())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


def test_hetero_attn(num_qubits: int, depth: int) -> torch.Tensor:
    zx_match_diagram_nx = clifford_zx_match_diagram(num_qubits, depth)
    zx_match_diagram = zx_match_diagram_nx.to_pyg_hdata().node_type_subgraph(SIMPLE_NODE_METADATA)
    edge_attn = HeteroEdgeAttn(2, 3)
    return edge_attn(zx_match_diagram.x_dict, zx_match_diagram.edge_index_dict)


def test_homo_attn(num_qubits: int, depth: int) -> torch.Tensor:
    zx_match_diagram_nx = clifford_zx_match_diagram(num_qubits, depth)
    zx_match_diagram = zx_match_diagram_nx.to_pyg_data()
    edge_attn = HomoEdgeAttn(2, 1)
    return edge_attn(zx_match_diagram.x, zx_match_diagram.edge_index)


print(test_hetero_attn(4, 4))
