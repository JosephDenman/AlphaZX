import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.mcts.policy_network import PolicyNetwork
from alphazx.models.homogeneous.mcts.value_network import ValueNetwork


class PredictionNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_channels: int,
                 edge_embedding_channels: int,
                 policy_num_pooling_encoder_blocks: int,
                 policy_num_pooling_heads: int,
                 policy_pooling_layer_norm: bool,
                 policy_pooling_dropout: float,
                 value_hidden_channels: int):
        super(PredictionNetwork, self).__init__()
        self.value_network = ValueNetwork(node_embedding_channels,
                                          edge_embedding_channels,
                                          value_hidden_channels)
        self.policy_network = PolicyNetwork(num_node_types,
                                            num_possible_phases,
                                            num_possible_new_edges,
                                            node_embedding_channels,
                                            policy_num_pooling_encoder_blocks,
                                            policy_num_pooling_heads,
                                            policy_pooling_layer_norm,
                                            policy_pooling_dropout)

    def reset_parameters(self):
        self.value_network.reset_parameters()
        self.policy_network.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        policy = self.policy_network(x, edge_index, node_type, batch)
        value = self.value_network(x, edge_index, edge_attr, batch)
        return policy, value
