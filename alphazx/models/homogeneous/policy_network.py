from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.gps import GPS
from alphazx.models.homogeneous.new_edge_selector import NewEdgeSelector
from alphazx.models.homogeneous.new_phase_selector import NewPhaseSelector
from alphazx.models.homogeneous.node_selector import NodeSelector
from alphazx.models.homogeneous.rewrite_type_selector import RewriteTypeSelector
from alphazx.models.homogeneous.transfer_edge_selector import TransferEdgeSelector


class PolicyNetwork(nn.Module):
    # Number of non-boundary match types (FRZ, FRX, FLZ, FLX, BR, BL, YRZ, YRX, YLZ, YLX)
    NUM_ACTION_TYPES = 10

    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_in_channels: int,
                 edge_in_channels: int,
                 gps_num_layers: int = 2,  # Reduced from 4 to prevent over-smoothing
                 gps_heads: int = 4,
                 gps_dropout: float = 0.1,
                 gps_act: str = 'relu',
                 gps_act_kwargs: dict[str, Any] = None,
                 gps_norm: str = 'batch_norm',
                 gps_norm_kwargs: dict[str, Any] = None,
                 gps_attn_type: str = 'multihead',
                 gps_attn_kwargs: dict[str, Any] = None,
                 gps_mlp_hidden_channels: int = 128,
                 gps_mlp_num_layers: int = 2,
                 rts_num_layers: int = 2,
                 ns_num_layers: int = 2,
                 nps_num_layers: int = 2,
                 nes_num_layers: int = 2,
                 tes_num_pooling_encoder_blocks: int = 1,
                 tes_num_pooling_heads: int = 1,
                 tes_pooling_layer_norm: bool = True,
                 dropout: float = 0.1,
                 num_scoring_heads: int = 8):
        super(PolicyNetwork, self).__init__()
        self.num_node_types = num_node_types
        self.num_possible_phases = num_possible_phases
        self.num_possible_new_edges = num_possible_new_edges
        T = self.NUM_ACTION_TYPES
        self.gps = GPS(node_in_channels,
                       node_in_channels,
                       edge_in_channels,
                       gps_num_layers,
                       gps_heads,
                       gps_dropout,
                       gps_act,
                       gps_act_kwargs,
                       gps_norm,
                       gps_norm_kwargs,
                       gps_attn_type,
                       gps_attn_kwargs,
                       gps_mlp_hidden_channels,
                       gps_mlp_num_layers)
        # RewriteTypeSelector outputs probabilities for action types (10 match types excluding boundary)
        self.rewrite_type_selector = RewriteTypeSelector(
            node_in_channels, T, rts_num_layers, dropout, num_scoring_heads,
        )
        # NodeSelector outputs probabilities for each node type (10 match types)
        self.node_selector = NodeSelector(
            node_in_channels, T, ns_num_layers, dropout, num_scoring_heads,
        )
        # Phase, edge, transfer selectors are conditioned on action type via
        # concatenated type embeddings (Approach 1).  Output shapes include
        # the T dimension: [B, T, N, ...].
        self.new_phase_selector = NewPhaseSelector(
            node_in_channels, T, num_possible_phases, nps_num_layers, dropout,
        )
        self.new_edge_selector = NewEdgeSelector(
            node_in_channels, T, num_possible_new_edges, nes_num_layers, dropout,
        )
        self.transfer_edge_selector = TransferEdgeSelector(
            node_in_channels, num_node_types, T,
            tes_num_pooling_encoder_blocks, tes_num_pooling_heads,
            tes_pooling_layer_norm, dropout,
        )

    def reset_parameters(self):
        self.gps.reset_parameters()
        self.rewrite_type_selector.reset_parameters()
        self.node_selector.reset_parameters()
        self.new_phase_selector.reset_parameters()
        self.new_edge_selector.reset_parameters()
        self.transfer_edge_selector.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor, graph_ids: torch.Tensor) -> AlphaZXDistributionParams:
        x = self.gps(x, edge_index, edge_attr, batch)
        mixture_probs = self.rewrite_type_selector(x, node_type, batch)
        node_probs = self.node_selector(x, node_type, batch)
        phase_probs = self.new_phase_selector(x, node_type, batch)
        edge_probs = self.new_edge_selector(x, node_type, batch)
        transfer_edge_probs = self.transfer_edge_selector(x, edge_index, node_type, batch)
        return AlphaZXDistributionParams(graph_ids,
                                         mixture_probs,
                                         node_probs,
                                         phase_probs,
                                         edge_probs,
                                         transfer_edge_probs)


def pad_or_strip(minibatch_actions: torch.Tensor, minibatch_obs: pyg.data.Batch) -> torch.Tensor:
    """
    Adds or removes padding (with fill value `0`) from the last dimension of `minibatch_actions` so that the last dimension
    of `minibatch_actions` is the same size as the maximum number of neighbors in `minibatch_obs` plus five. Both tensors
    are assumed to be three-dimensional.

    :param minibatch_actions: The actions to be padded.
    :param minibatch_obs: The batch used to calculate the maximum degree over all nodes in a batch.
    :return: The newly padded `minibatch_actions` tensor.
    """
    target_size = torch.max(pyg.utils.degree(minibatch_obs.edge_index[0])).int().item() + 5
    # Get the sizes of the source and target tensors
    minibatch_actions_size = minibatch_actions.size()
    # Calculate the size difference in the last dimension
    diff = target_size - minibatch_actions_size[-1]
    if diff > 0:
        # If the target's last dimension is larger, pad the source tensor
        pad_shape = list(minibatch_actions_size)
        pad_shape[-1] = diff
        padding = torch.full(pad_shape, 0, dtype=minibatch_actions.dtype, device=minibatch_actions.device)
        padded_minibatch_actions = torch.cat((minibatch_actions, padding), dim=-1)
    else:
        # If the target's last dimension is smaller or equal, slice the source tensor
        padded_minibatch_actions = minibatch_actions[..., :target_size]
    return padded_minibatch_actions
