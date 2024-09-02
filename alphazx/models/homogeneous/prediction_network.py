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
                 node_embedding_channels: int,
                 edge_embedding_channels: int,
                 policy_rewrite_type_out_channels: int,
                 policy_node_out_channels: int,
                 policy_rts_num_layers: int,
                 policy_ns_num_layers: int,
                 policy_nps_num_layers: int,
                 policy_nes_num_layers: int,
                 policy_tes_num_pooling_encoder_blocks: int,
                 policy_tes_num_pooling_heads: int,
                 policy_tes_pooling_layer_norm: bool,
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
                                            policy_rewrite_type_out_channels,
                                            policy_node_out_channels,
                                            policy_rts_num_layers,
                                            policy_ns_num_layers,
                                            policy_nps_num_layers,
                                            policy_nes_num_layers,
                                            policy_tes_num_pooling_encoder_blocks,
                                            policy_tes_num_pooling_heads,
                                            policy_tes_pooling_layer_norm,
                                            policy_pooling_dropout)

    def reset_parameters(self):
        self.value_network.reset_parameters()
        self.policy_network.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        policy = self.policy_network(x, edge_index, node_type, batch)
        value = self.value_network(x, edge_index, edge_attr, batch)
        return policy, value
