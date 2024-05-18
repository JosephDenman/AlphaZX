from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.new_edge_selector import NewEdgeSelector
from alphazx.models.homogeneous.new_phase_selector import NewPhaseSelector
from alphazx.models.homogeneous.node_selector import NodeSelector
from alphazx.models.homogeneous.rewrite_type_selector import RewriteTypeSelector
from alphazx.models.homogeneous.transfer_edge_selector import TransferEdgeSelector
from alphazx.models.homogeneous.gps import GPS


class PolicyNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_channels: int,
                 gps_channels: int,
                 gps_edge_in_channels: int,
                 gps_edge_out_channels: int,
                 gps_pe_in_channels: int,
                 gps_pe_out_channels: int,
                 gps_num_layers: int,
                 gps_bias: bool,
                 gps_num_attn_heads: int,
                 gps_attn_type: str,
                 gps_attn_kwargs: dict[str, Any],
                 gps_mlp_hidden_channels: int,
                 num_pooling_encoder_blocks: int,
                 num_pooling_heads: int,
                 pooling_layer_norm: bool,
                 pooling_dropout: float):
        super(PolicyNetwork, self).__init__()
        self.num_node_types = num_node_types
        self.num_possible_phases = num_possible_phases
        self.num_possible_new_edges = num_possible_new_edges
        self.gps = GPS(num_node_types * num_possible_phases,
                       gps_channels,
                       node_embedding_channels,
                       gps_edge_in_channels,
                       gps_edge_out_channels,
                       gps_pe_in_channels,
                       gps_pe_out_channels,
                       gps_num_layers,
                       gps_bias,
                       gps_num_attn_heads,
                       gps_attn_type,
                       gps_attn_kwargs,
                       gps_mlp_hidden_channels)
        self.rewrite_type_selector = RewriteTypeSelector(node_embedding_channels, num_node_types,
                                                         num_pooling_encoder_blocks, num_pooling_heads,
                                                         pooling_layer_norm, pooling_dropout)
        self.node_selector = NodeSelector(node_embedding_channels, num_node_types)
        self.new_phase_selector = NewPhaseSelector(node_embedding_channels, num_possible_phases)
        self.new_edge_selector = NewEdgeSelector(node_embedding_channels, num_possible_new_edges)
        self.transfer_edge_selector = TransferEdgeSelector(node_embedding_channels, num_node_types)

    def forward(self, data: pyg.data.Data) -> AlphaZXDistributionParams:
        """
        TODO: Have the node, phase, and edge prob computations be autoregressive. Compute mixture probabilities last
              to incorporate intermediate embedding updates. Do we need to do layer norm / residual connection between each
              MLP?
        :param data: The pyg.data.Data object representing the ZX match diagram.
        :return: Parameters for the AlphaZXDistribution.
        """
        x = self.gps(data.x, data.pe, data.edge_index, data.batch)
        mixture_probs = self.rewrite_type_selector(x, data.node_type, data.batch)
        node_probs = self.node_selector(x, data.node_type, data.batch)
        phase_probs = self.new_phase_selector(x, data.node_type, data.batch)
        edge_probs = self.new_edge_selector(x, data.node_type, data.batch)
        transfer_edge_probs = self.transfer_edge_selector(x, data.edge_index, data.node_type, data.batch)
        return AlphaZXDistributionParams(mixture_probs,
                                         node_probs,
                                         phase_probs,
                                         edge_probs,
                                         transfer_edge_probs)


def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))
