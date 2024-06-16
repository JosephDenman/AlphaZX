from typing import Any

import torch.nn
import torch_geometric as pyg
from ding.utils import MODEL_REGISTRY

from alphazx.distributions import AlphaZXDistributionParams
from alphazx.models.homogeneous import RepresentationNetwork, ValueNetwork, PolicyNetwork


@MODEL_REGISTRY.register('alphazx_vac')
class AlphaZXVAC(torch.nn.Module):
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_channels: int,
                 repr_gps_channels: int,
                 repr_gps_edge_in_channels: int,
                 repr_gps_edge_out_channels: int,
                 repr_gps_pe_in_channels: int,
                 repr_gps_pe_out_channels: int,
                 repr_gps_num_layers: int,
                 repr_gps_bias: bool,
                 repr_gps_num_attn_heads: int,
                 repr_gps_attn_type: str,
                 repr_gps_attn_kwargs: dict[str, Any],
                 repr_gps_mlp_hidden_channels: int,
                 policy_num_pooling_encoder_blocks: int,
                 policy_num_pooling_heads: int,
                 policy_pooling_layer_norm: bool,
                 policy_pooling_dropout: float,
                 value_gmt_num_encoder_blocks: int,
                 value_gmt_num_heads: int,
                 value_gmt_layer_norm: bool,
                 value_gmt_dropout: float):
        super(AlphaZXVAC, self).__init__()
        self.encoder = RepresentationNetwork(num_node_types,
                                             num_possible_phases,
                                             node_embedding_channels,
                                             repr_gps_channels,
                                             repr_gps_edge_in_channels,
                                             repr_gps_edge_out_channels,
                                             repr_gps_pe_in_channels,
                                             repr_gps_pe_out_channels,
                                             repr_gps_num_layers,
                                             repr_gps_bias,
                                             repr_gps_num_attn_heads,
                                             repr_gps_attn_type,
                                             repr_gps_attn_kwargs,
                                             repr_gps_mlp_hidden_channels)
        self.critic = ValueNetwork(node_embedding_channels,
                                   value_gmt_num_encoder_blocks,
                                   value_gmt_num_heads,
                                   value_gmt_layer_norm,
                                   value_gmt_dropout)
        self.actor = PolicyNetwork(num_node_types,
                                   num_possible_phases,
                                   num_possible_new_edges,
                                   node_embedding_channels,
                                   policy_num_pooling_encoder_blocks,
                                   policy_num_pooling_heads,
                                   policy_pooling_layer_norm,
                                   policy_pooling_dropout)

    def forward(self, data: pyg.data.Data, mode: str) -> dict[str, AlphaZXDistributionParams | torch.Tensor]:
        assert mode in self.mode, 'Not support forward mode: {}/{}'.format(mode, self.mode)
        return getattr(self, mode)(data)

    def compute_actor(self, data: pyg.data.Data) -> dict[str, AlphaZXDistributionParams]:
        data.x = self.encoder(data)
        alphazx_dist_params = self.actor(data)
        return {'logit': alphazx_dist_params}

    def compute_critic(self, data: pyg.data.Data) -> dict[str, torch.Tensor]:
        data.x = self.encoder(data)
        value = self.critic(data)
        return {'value': value}

    def compute_actor_critic(self, data: pyg.data.Data) -> dict[str, AlphaZXDistributionParams | torch.Tensor]:
        data.x = self.encoder(data)
        alphazx_dist_params = self.actor(data)
        value = self.critic(data)
        return {'logit': alphazx_dist_params, 'value': value}
