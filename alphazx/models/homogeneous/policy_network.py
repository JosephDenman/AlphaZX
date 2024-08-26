import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.new_edge_selector import NewEdgeSelector
from alphazx.models.homogeneous.new_phase_selector import NewPhaseSelector
from alphazx.models.homogeneous.node_selector import NodeSelector
from alphazx.models.homogeneous.rewrite_type_selector import RewriteTypeSelector
from alphazx.models.homogeneous.transfer_edge_selector import TransferEdgeSelector


class PolicyNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_channels: int,
                 num_pooling_encoder_blocks: int,
                 num_pooling_heads: int,
                 pooling_layer_norm: bool,
                 dropout: float):
        super(PolicyNetwork, self).__init__()
        self.num_node_types = num_node_types
        self.num_possible_phases = num_possible_phases
        self.num_possible_new_edges = num_possible_new_edges
        self.rewrite_type_selector = RewriteTypeSelector(node_embedding_channels, num_node_types,
                                                         num_pooling_encoder_blocks, num_pooling_heads,
                                                         pooling_layer_norm, dropout)
        self.node_selector = NodeSelector(node_embedding_channels, num_node_types, dropout)
        self.new_phase_selector = NewPhaseSelector(node_embedding_channels, num_possible_phases, dropout)
        self.new_edge_selector = NewEdgeSelector(node_embedding_channels, num_possible_new_edges, dropout)
        self.transfer_edge_selector = TransferEdgeSelector(node_embedding_channels, num_node_types, num_pooling_encoder_blocks, num_pooling_heads, pooling_layer_norm, dropout)

    def reset_parameters(self):
        self.rewrite_type_selector.reset_parameters()
        self.node_selector.reset_parameters()
        self.new_phase_selector.reset_parameters()
        self.new_edge_selector.reset_parameters()
        self.transfer_edge_selector.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor, graph_ids: torch.Tensor) -> AlphaZXDistributionParams:
        """
        TODO: Have the node, phase, and edge prob computations be autoregressive. Compute mixture probabilities last
              to incorporate intermediate embedding updates. Do we need to do layer norm / residual connection between each
              MLP?
        :return: Parameters for the AlphaZXDistribution.
        """
        mixture_probs = self.rewrite_type_selector(x, edge_index, node_type, batch)
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


def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))


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

