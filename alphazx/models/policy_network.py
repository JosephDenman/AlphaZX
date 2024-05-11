from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from alphazx.diagram.match import FRightZMatch, FRightXMatch
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.aggregation.transfer_edge_transformer import TransferEdgeTransformer
from alphazx.models.gps import GPS


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
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.mixture_trans = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
                                                             num_node_types,
                                                             num_pooling_encoder_blocks,
                                                             num_pooling_heads,
                                                             pooling_layer_norm,
                                                             pooling_dropout)
        self.node_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.phase_mlp = pyg.nn.MLP([node_embedding_channels, num_possible_phases])
        self.edge_mlp = pyg.nn.MLP([node_embedding_channels, num_possible_new_edges])
        self.transfer_edge_trans = TransferEdgeTransformer(node_embedding_channels, num_node_types)

    def _compute_transfer_edge_probs(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                                     batch: torch.Tensor) -> torch.Tensor:
        # Run transfer edge transformer
        transfer_edge_params = self.transfer_edge_trans(x, edge_index, batch)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = torch.logical_or(node_type_batch == FRightZMatch.index, node_type_batch == FRightXMatch.index)
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(transfer_edge_params.shape[2], device=transfer_edge_params.device)
        # Transfer edge selection probabilities should be zero for all non-simple nodes
        transfer_edge_params = torch.where(~node_type_mask.unsqueeze(-1).expand_as(transfer_edge_params), replacement_row, transfer_edge_params)
        return transfer_edge_params

    def _compute_new_edge_probs(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Gather node embeddings according to batch
        new_edge_probs = pyg.utils.to_dense_batch(x, batch)[0]
        # Project node embeddings to a vector representing the probabilities of selecting the number of new edges
        new_edge_probs = self.edge_mlp(new_edge_probs).squeeze(dim=-1)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = torch.logical_or(node_type_batch == FRightZMatch.index, node_type_batch == FRightXMatch.index)
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(self.num_possible_new_edges, device=new_edge_probs.device)
        replacement_row[0] = 1
        # Insert replacement row for each non-simple node
        new_edge_probs = torch.where(~node_type_mask.unsqueeze(-1).expand_as(new_edge_probs), replacement_row, new_edge_probs)
        # Softmax over probabilities for each simple node
        new_edge_probs[node_type_mask] = torch.softmax(new_edge_probs[node_type_mask], dim=-1)
        return new_edge_probs

    def _compute_phase_probs(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Exact same computation as the new edge probs computation
        # Gather node embeddings according to batch
        phase_probs = pyg.utils.to_dense_batch(x, batch)[0]
        # Project node embeddings to a vector representing the probabilities of selecting phases
        phase_probs = self.phase_mlp(phase_probs).squeeze(dim=-1)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = torch.logical_or(node_type_batch == FRightZMatch.index, node_type_batch == FRightXMatch.index)
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(self.num_possible_phases, device=phase_probs.device)
        replacement_row[0] = 1
        # Insert replacement row for each non-simple node
        phase_probs = torch.where(~node_type_mask.unsqueeze(-1).expand_as(phase_probs), replacement_row, phase_probs)
        # Softmax over probabilities for each simple node
        phase_probs[node_type_mask] = torch.softmax(phase_probs[node_type_mask], dim=-1)
        return phase_probs

    def _compute_node_probs(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # TODO: Add a neighbor aggregation
        # Gather node embeddings according to batch
        x, batch_mask = pyg.utils.to_dense_batch(x, batch)
        # Project node embeddings to a scalar representing an un-normalized probability of selecting the node
        x = self.node_mlp(x).squeeze(dim=-1)
        # Mask padding nodes from to_dense_batch with negative infinity
        x[~batch_mask] = -torch.inf
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Initialize the result tensor
        node_probs = torch.fill(torch.empty(x.shape[0], self.num_node_types, x.shape[1], device=x.device), -torch.inf)
        # Scatter selection probabilities to correct column and row, respecting batching
        node_probs = node_probs.scatter(1, node_type_batch.unsqueeze(dim=1), x.unsqueeze(dim=1))
        # Softmax over innermost rows
        node_probs = torch.softmax(node_probs, dim=-1)
        # Sometimes softmax produces NaN, so set those values to zero - no chance of selection.
        node_probs[torch.isnan(node_probs)] = 0.
        # Zero out boundary probabilities
        node_probs[:, 0, 0] = 0.
        return node_probs

    def _compute_mixture_probs(self, x: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # We need to compute the mixture probabilities for each batch separately, so we modify node types so that the
        # same node type in different batches is represented by a different index
        index = node_type + self.num_node_types * batch
        # Aggregate the node embeddings by node type
        mixture_params = self.mixture_trans(x, index)
        # Project the final embeddings to a scalar
        mixture_params = self.mixture_mlp(mixture_params).squeeze(dim=-1)
        # Replace nan values produced by the aggregation with negative infinities
        mixture_params[torch.isnan(mixture_params)] = -torch.inf
        # The last negative infinity in the tensor is the last element before the start of the last batch
        inf_indices = torch.where(mixture_params == -torch.inf)[0]
        if len(inf_indices) > 0:
            # Pad the last batch with negative infinity so that all batches have the same number of elements
            pad_length = self.num_node_types - (len(mixture_params) - inf_indices[-1] - 1)
            mixture_params = F.pad(mixture_params, (0, pad_length), mode='constant', value=-torch.inf)
        else:
            pad_length = self.num_node_types - len(mixture_params)
            mixture_params = F.pad(mixture_params, (0, pad_length), mode='constant', value=-torch.inf)

        # Reshape the tensor to have the batch dimension
        mixture_params = mixture_params.view(-1, self.num_node_types)
        # Probability of selecting a boundary node should be zero
        mixture_params[:, 0] = -torch.inf
        # Apply softmax to get the mixture probabilities
        mixture_params = torch.softmax(mixture_params, dim=-1)
        return mixture_params

    def forward(self, data: pyg.data.Data) -> AlphaZXDistributionParams:
        """
        TODO: Have the node, phase, and edge prob computations be autoregressive. Compute mixture probabilities last
              to incorporate intermediate embedding updates. Do we need to do layer norm / residual connection between each
              MLP?
        TODO: Ensure that removing connected components from the ZXDiagram does not affect 'Data' batching.
        :param data: The pyg.data.Data object representing the ZXMatchDiagram.
        :return: Parameters for the AlphaZXDistribution. Each value in the returned dictionary is a batch of distribution
                 parameters.
        """
        x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        mixture_probs = self._compute_mixture_probs(x, data.node_type, data.batch)
        node_probs = self._compute_node_probs(x, data.node_type, data.batch)
        phase_probs = self._compute_phase_probs(x, data.node_type, data.batch)
        edge_probs = self._compute_new_edge_probs(x, data.node_type, data.batch)
        transfer_edge_probs = self._compute_transfer_edge_probs(x, data.edge_index, data.node_type, data.batch)
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
