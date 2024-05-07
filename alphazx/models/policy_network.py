from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch import index_select

from alphazx import concatenate_neighbor_features
from alphazx.diagram.match import BoundaryMatch, FRightZMatch, FRightXMatch
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.attention import SigmoidCrossAttention
from alphazx.models.gps import GPS

torch.set_printoptions(threshold=200)


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
        self.neighbor_aggr = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
                                                             num_node_types,
                                                             num_pooling_encoder_blocks,
                                                             num_pooling_heads,
                                                             pooling_layer_norm,
                                                             pooling_dropout)
        self.sigmoid_attn = SigmoidCrossAttention(node_embedding_channels)
        self.mixture_aggr = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
                                                            num_node_types,
                                                            num_pooling_encoder_blocks,
                                                            num_pooling_heads,
                                                            pooling_layer_norm,
                                                            pooling_dropout)
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels, 1])

    def _compute_transfer_edge_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                                     node_types: torch.Tensor) -> torch.Tensor:
        neighbor_x = concatenate_neighbor_features(x, edge_index)
        neighbor_pad_mask = (neighbor_x == torch.full_like(x[0], -torch.inf).unsqueeze(0).unsqueeze(0)).all(dim=-1)
        transfer_edge_params = self.sigmoid_attn(x.unsqueeze(dim=1), neighbor_x, neighbor_pad_mask).squeeze(dim=1)
        non_simple_node_mask = torch.logical_and(node_types != FRightZMatch.index, node_types != FRightXMatch.index)
        transfer_edge_params[non_simple_node_mask] = torch.zeros(transfer_edge_params.shape[1])
        return transfer_edge_params

    def _compute_new_edge_probs(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        new_edge_probs = torch.cat([torch.tensor([1.]), torch.zeros(self.num_possible_new_edges - 1)]).repeat(
            x.shape[0], 1)
        # new_edge_probs = torch.fill(torch.empty([x.shape[0], self.num_possible_new_edges]), 0.)
        simple_node_mask = torch.logical_or(node_types == FRightZMatch.index, node_types == FRightXMatch.index)
        new_edge_probs[simple_node_mask] = torch.softmax(x[simple_node_mask][:, 1 + self.num_possible_phases:], dim=-1)
        return new_edge_probs

    def _compute_phase_probs(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        phase_probs = torch.cat([torch.tensor([1.]), torch.zeros(self.num_possible_phases - 1)]).repeat(x.shape[0], 1)
        simple_node_mask = torch.logical_or(node_types == FRightZMatch.index, node_types == FRightXMatch.index)
        phase_probs[simple_node_mask] = torch.softmax(x[simple_node_mask][:, 1:1 + self.num_possible_phases], dim=-1)
        return phase_probs

    def _compute_node_probs(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        node_probs = torch.fill(torch.empty([self.num_node_types, x.shape[0]]), 0)
        for node_type in range(self.num_node_types):
            if node_type != BoundaryMatch.index:
                node_probs[node_type][node_types != node_type] = -torch.inf
                node_probs[node_type][node_types == node_type] = x[node_types == node_type][:, :1].squeeze(dim=-1)
                node_probs[node_type] = torch.softmax(node_probs[node_type], dim=-1)
        node_probs[node_probs.isnan()] = 0.
        return node_probs

    def _compute_mixture_probs(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        mixture_params = self.mixture_aggr(x, node_type)
        mixture_params = self.mixture_mlp(mixture_params).squeeze(dim=-1)
        mixture_params = F.pad(mixture_params, [0, self.num_node_types - mixture_params.shape[0]], mode='constant',
                               value=-torch.inf)
        mixture_pad_mask = torch.isnan(mixture_params)
        mixture_params[mixture_pad_mask] = -torch.inf
        mixture_params[BoundaryMatch.index] = -torch.inf
        mixture_params = torch.softmax(mixture_params, dim=-1)
        return mixture_params

    def forward(self, data: pyg.data.Data) -> AlphaZXDistributionParams:
        """
        TODO: Figure out batching. All of the edge index based operations should stay correct, since the batched graphs
              are disconnected. We just have to collect the result from the different connected components using 'data.batch'.
        TODO: Ensure that removing connected components from the ZXDiagram does not affect 'Data' batching.
        :param data: The pyg.data.Data object representing the ZXMatchDiagram.
        :return: Parameters for the AlphaZXDistribution. Each value in the returned dictionary is a batch of distribution
                 parameters.
        """
        x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        x = self.neighbor_aggr(index_select(x, 0, data.edge_index[0]), data.edge_index[1])
        return AlphaZXDistributionParams(self._compute_mixture_probs(x, data.node_type).unsqueeze(dim=0),
                                         self._compute_node_probs(x, data.node_type).unsqueeze(dim=0),
                                         self._compute_phase_probs(x, data.node_type).unsqueeze(dim=0),
                                         self._compute_new_edge_probs(x, data.node_type).unsqueeze(dim=0),
                                         self._compute_transfer_edge_probs(x, data.edge_index,
                                                                           data.node_type).unsqueeze(dim=0))


def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))
