from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.policy_network import PolicyNetwork
from alphazx.models.homogeneous.value_network import ValueNetwork


class PredictionNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 policy_node_embedding_channels: int,
                 policy_gps_channels: int,
                 policy_gps_edge_in_channels: int,
                 policy_gps_edge_out_channels: int,
                 policy_gps_pe_in_channels: int,
                 policy_gps_pe_out_channels: int,
                 policy_gps_num_layers: int,
                 policy_gps_bias: bool,
                 policy_gps_num_attn_heads: int,
                 policy_gps_attn_type: str,
                 policy_gps_attn_kwargs: dict[str, Any],
                 policy_gps_mlp_hidden_channels: int,
                 policy_num_pooling_encoder_blocks: int,
                 policy_num_pooling_heads: int,
                 policy_pooling_layer_norm: bool,
                 policy_pooling_dropout: float,
                 value_node_embedding_channels: int,
                 value_gps_channels: int,
                 value_gps_edge_in_channels: int,
                 value_gps_edge_out_channels: int,
                 value_gps_pe_in_channels: int,
                 value_gps_pe_out_channels: int,
                 value_gps_num_layers: int,
                 value_gps_bias: bool,
                 value_gps_num_attn_heads: int,
                 value_gps_attn_type: str,
                 value_gps_attn_kwargs: dict[str, Any],
                 value_gps_mlp_hidden_channels: int,
                 value_gmt_num_encoder_blocks: int,
                 value_gmt_num_heads: int,
                 value_gmt_layer_norm: bool,
                 value_gmt_dropout: float):
        super(PredictionNetwork, self).__init__()
        self.value_network = ValueNetwork(num_node_types,
                                          num_possible_phases,
                                          value_node_embedding_channels,
                                          value_gps_channels,
                                          value_gps_edge_in_channels,
                                          value_gps_edge_out_channels,
                                          value_gps_pe_in_channels,
                                          value_gps_pe_out_channels,
                                          value_gps_num_layers,
                                          value_gps_bias,
                                          value_gps_num_attn_heads,
                                          value_gps_attn_type,
                                          value_gps_attn_kwargs,
                                          value_gps_mlp_hidden_channels,
                                          value_gmt_num_encoder_blocks,
                                          value_gmt_num_heads,
                                          value_gmt_layer_norm,
                                          value_gmt_dropout)
        self.policy_network = PolicyNetwork(num_node_types,
                                            num_possible_phases,
                                            num_possible_new_edges,
                                            policy_node_embedding_channels,
                                            policy_gps_channels,
                                            policy_gps_edge_in_channels,
                                            policy_gps_edge_out_channels,
                                            policy_gps_pe_in_channels,
                                            policy_gps_pe_out_channels,
                                            policy_gps_num_layers,
                                            policy_gps_bias,
                                            policy_gps_num_attn_heads,
                                            policy_gps_attn_type,
                                            policy_gps_attn_kwargs,
                                            policy_gps_mlp_hidden_channels,
                                            policy_num_pooling_encoder_blocks,
                                            policy_num_pooling_heads,
                                            policy_pooling_layer_norm,
                                            policy_pooling_dropout)

    def forward(self, data: pyg.data.Data) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        policy = self.policy_network(data)
        value = self.value_network(data)
        return policy, value
