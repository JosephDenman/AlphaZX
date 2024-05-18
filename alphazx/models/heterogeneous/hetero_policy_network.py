import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.diagram.match import METADATA
from alphazx.distributions.alpha_zx_dist import AZXDistributionParams
from alphazx.models.heterogeneous.hgt import HGT


# TODO: The critical question is how to apply global attention to a heterogeneous graph
# TODO: The dumb approach: convert to/from a hetero/homo graph as needed. HGT -> to_homo -> GPS -> to_hetero
class HeteroPolicyNetwork(nn.Module):
    def __init__(self,
                 hgt_in_channels: int,
                 hgt_hidden_channels: int,
                 hgt_out_channels: int,
                 hgt_num_heads: int,
                 hgt_num_layers: int):
        super(HeteroPolicyNetwork, self).__init__()
        self.hgt = HGT(METADATA, hgt_in_channels, hgt_hidden_channels, hgt_out_channels, hgt_num_heads, hgt_num_layers)

    def forward(self, x_dict: dict[str, torch.Tensor],
                edge_index_dict: dict[str, torch.Tensor]) -> AZXDistributionParams:
        x_dict = self.hgt(x_dict, edge_index_dict)
        data = pyg.data.HeteroData(x_dict=x_dict, edge_index_dict=edge_index_dict).to_homogeneous(node_attrs=['phase'],
                                                                                                  edge_attrs=['size'],
                                                                                                  add_node_type=True,
                                                                                                  add_edge_type=True,
                                                                                                  dummy_values=False)


pyg.nn.to_hetero(pyg.nn.GraphMultisetTransformer(2, 2, 2,
                                                 2, 2, 2), metadata=METADATA)
