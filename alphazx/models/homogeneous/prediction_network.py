import torch
import torch.nn as nn

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.policy_network import PolicyNetwork
from alphazx.models.homogeneous.value_network import ValueNetwork


class PredictionNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_in_channels: int,
                 edge_in_channels: int):
        super(PredictionNetwork, self).__init__()
        self.value_network = ValueNetwork(node_in_channels, edge_in_channels)
        self.policy_network = PolicyNetwork(num_node_types, num_possible_phases, num_possible_new_edges,
                                            node_in_channels, edge_in_channels)

    def reset_parameters(self):
        self.value_network.reset_parameters()
        self.policy_network.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor, graph_ids: torch.Tensor) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        policy = self.policy_network(x, edge_index, edge_attr, node_type, batch, graph_ids)
        value = self.value_network(x, edge_index, edge_attr, batch)
        return policy, value
