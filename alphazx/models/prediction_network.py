import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.policy_network import PolicyNetwork
from alphazx.models.value_network import ValueNetwork


class PredictionNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 value_hgt_hidden_dim: int,
                 value_hgt_out_dim: int,
                 value_hgt_heads: int,
                 value_hgt_layers: int,
                 value_encoder_blocks: int,
                 value_encoder_attn_heads: int,
                 value_encoder_feedforward_dim: int,
                 value_encoder_dropout: float,
                 value_encoder_activation: str,
                 value_encoder_layer_norm_eps: float,
                 value_encoder_bias: bool,
                 value_encoder_norm_first: bool,
                 value_pooling_encoder_blocks: int,
                 value_pooling_heads: int,
                 value_pooling_layer_norm: bool,
                 value_pooling_dropout: float,
                 policy_encoder_blocks: int,
                 policy_encoder_attn_heads: int,
                 policy_encoder_feedforward_dim: int,
                 policy_encoder_dropout: float,
                 policy_encoder_activation: str,
                 policy_encoder_layer_norm_eps: float,
                 policy_encoder_bias: bool,
                 policy_encoder_norm_first: bool):
        super(PredictionNetwork, self).__init__()
        self.value_network = ValueNetwork(input_dim,
                                          value_hgt_hidden_dim,
                                          value_hgt_out_dim,
                                          value_hgt_heads,
                                          value_hgt_layers,
                                          value_encoder_blocks,
                                          value_encoder_attn_heads,
                                          value_encoder_feedforward_dim,
                                          value_encoder_dropout,
                                          value_encoder_activation,
                                          value_encoder_layer_norm_eps,
                                          value_encoder_bias,
                                          value_encoder_norm_first,
                                          value_pooling_encoder_blocks,
                                          value_pooling_heads,
                                          value_pooling_layer_norm,
                                          value_pooling_dropout)
        self.policy_network = PolicyNetwork(input_dim,
                                            policy_encoder_blocks,
                                            policy_encoder_attn_heads,
                                            policy_encoder_feedforward_dim,
                                            policy_encoder_dropout,
                                            policy_encoder_activation,
                                            policy_encoder_layer_norm_eps,
                                            policy_encoder_bias,
                                            policy_encoder_norm_first)

    def forward(self, data: pyg.data.Data) -> tuple[AlphaZXDistributionParams, torch.Tensor]:

        policy = self.policy_network(x_dict, edge_index_dict)
        value = self.value_network(x_dict, edge_index_dict)
        return policy, value
