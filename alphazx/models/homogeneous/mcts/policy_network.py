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
                 pooling_dropout: float):
        super(PolicyNetwork, self).__init__()
        self.num_node_types = num_node_types
        self.num_possible_phases = num_possible_phases
        self.num_possible_new_edges = num_possible_new_edges
        self.rewrite_type_selector = RewriteTypeSelector(node_embedding_channels, num_node_types,
                                                         num_pooling_encoder_blocks, num_pooling_heads,
                                                         pooling_layer_norm, pooling_dropout)
        self.node_selector = NodeSelector(node_embedding_channels, num_node_types)
        self.new_phase_selector = NewPhaseSelector(node_embedding_channels, num_possible_phases)
        self.new_edge_selector = NewEdgeSelector(node_embedding_channels, num_possible_new_edges)
        self.transfer_edge_selector = TransferEdgeSelector(node_embedding_channels, num_node_types)

    def reset_parameters(self):
        self.rewrite_type_selector.reset_parameters()
        self.node_selector.reset_parameters()
        self.new_phase_selector.reset_parameters()
        self.new_edge_selector.reset_parameters()
        self.transfer_edge_selector.reset_parameters()

    def forward(self, data: pyg.data.Data) -> AlphaZXDistributionParams:
        """
        TODO: Have the node, phase, and edge prob computations be autoregressive. Compute mixture probabilities last
              to incorporate intermediate embedding updates. Do we need to do layer norm / residual connection between each
              MLP?
        :param data: The pyg.data.Data object representing the ZX match diagram.
        :return: Parameters for the AlphaZXDistribution.
        """
        mixture_probs = self.rewrite_type_selector(data.x, data.edge_index, data.node_type, data.batch)
        node_probs = self.node_selector(data.x, data.node_type, data.batch)
        phase_probs = self.new_phase_selector(data.x, data.node_type, data.batch)
        edge_probs = self.new_edge_selector(data.x, data.node_type, data.batch)
        transfer_edge_probs = self.transfer_edge_selector(data.x, data.edge_index, data.node_type, data.batch)
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
