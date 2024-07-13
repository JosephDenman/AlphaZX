from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.mcts.prediction_network import PredictionNetwork
from alphazx.models.homogeneous.mcts.representation_network import RepresentationNetwork


class AlphaZXModel(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_out_channels: int,
                 node_out_channels: int,
                 num_edge_embeddings: int,
                 edge_embedding_out_channels: int,
                 edge_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 num_layers: int,
                 bias: bool,
                 num_attn_heads: int,
                 attn_type: str,
                 attn_kwargs: dict[str, Any],
                 mlp_hidden_channels: int,
                 policy_num_pooling_encoder_blocks: int,
                 policy_num_pooling_heads: int,
                 policy_pooling_layer_norm: bool,
                 policy_pooling_dropout: float,
                 value_gmt_num_encoder_blocks: int,
                 value_gmt_num_heads: int,
                 value_gmt_layer_norm: bool,
                 value_gmt_dropout: float):
        super(AlphaZXModel, self).__init__()
        self.representation_network = RepresentationNetwork(num_node_types,
                                                            num_possible_phases,
                                                            node_embedding_out_channels,
                                                            node_out_channels,
                                                            num_edge_embeddings,
                                                            edge_embedding_out_channels,
                                                            edge_out_channels,
                                                            pe_in_channels,
                                                            pe_out_channels,
                                                            num_layers,
                                                            bias,
                                                            num_attn_heads,
                                                            attn_type,
                                                            attn_kwargs,
                                                            mlp_hidden_channels)
        self.prediction_network = PredictionNetwork(num_node_types,
                                                    num_possible_phases,
                                                    num_possible_new_edges,
                                                    node_out_channels,
                                                    policy_num_pooling_encoder_blocks,
                                                    policy_num_pooling_heads,
                                                    policy_pooling_layer_norm,
                                                    policy_pooling_dropout,
                                                    value_gmt_num_encoder_blocks,
                                                    value_gmt_num_heads,
                                                    value_gmt_layer_norm,
                                                    value_gmt_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor, pe: torch.Tensor) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        x = self.representation_network(x, edge_index, edge_attr, batch, pe)
        policy, value = self.prediction_network(x, edge_index, node_type, batch)
        return policy, value

    def compute_policy_value(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor, pe: torch.Tensor) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        x = self.representation_network(x, edge_index, batch, pe)
        policy, value = self.prediction_network(x, edge_index, node_type, batch)
        return policy, value

    def compute_logp_value(self, data: pyg.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        pass
