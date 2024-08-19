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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> AlphaZXDistributionParams:
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


def pad_or_strip(source: torch.Tensor, target_size: int, fill_value: float = 0.) -> torch.Tensor:
    """
    Adds or removes padding (with fill value `fill_value`) from the last dimension of `source` so that the last dimension
    of `source` is the same size as `target_size`. Both tensors are assumed to be three-dimensional. It
    is possible that the last dimension of `source` is larger or smaller than the last dimension of `target`.

    :param source: The tensor to be padded.
    :param target_size: The length of the last dimension of `source` after padding.
    :param fill_value: The value to pad with.

    :return: The newly padded `source` tensor.
    """
    # Get the sizes of the source and target tensors
    source_size = source.size()
    # Calculate the size difference in the last dimension
    diff = target_size - source_size[-1]
    if diff > 0:
        # If the target's last dimension is larger, pad the source tensor
        pad_shape = list(source_size)
        pad_shape[-1] = diff
        padding = torch.full(pad_shape, fill_value, dtype=source.dtype, device=source.device)
        padded_source = torch.cat((source, padding), dim=-1)
    else:
        # If the target's last dimension is smaller or equal, slice the source tensor
        padded_source = source[..., :target_size]
    return padded_source

