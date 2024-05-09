from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from alphazx.diagram.match import BoundaryMatch, FRightZMatch, FRightXMatch, NODE_METADATA
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.aggregation.transfer_edge_transformer import TransferEdgeTransformer
from alphazx.models.gps import GPS

torch.set_printoptions(threshold=10000)


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
        # self.neighbor_aggr = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
        #                                                      num_node_types,
        #                                                      num_pooling_encoder_blocks,
        #                                                      num_pooling_heads,
        #                                                      pooling_layer_norm,
        #                                                      pooling_dropout)
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.node_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.edge_mlp = pyg.nn.MLP([node_embedding_channels, num_possible_new_edges])
        self.phase_mlp = pyg.nn.MLP([node_embedding_channels, num_possible_phases])
        self.transfer_edge_trans = TransferEdgeTransformer(node_embedding_channels, num_node_types)
        self.mixture_trans = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
                                                             num_node_types,
                                                             num_pooling_encoder_blocks,
                                                             num_pooling_heads,
                                                             pooling_layer_norm,
                                                             pooling_dropout)

    def _compute_transfer_edge_probs(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                                     batch: torch.Tensor) -> torch.Tensor:
        transfer_edge_params = self.transfer_edge_trans(x, edge_index, batch)
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

    def _compute_phase_probs(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Gather node embeddings according to batch
        x, x_mask = pyg.utils.to_dense_batch(x, batch)
        # Project node embeddings to a vector representing the probabilities of selecting phases.
        x = self.phase_mlp(x).squeeze(dim=-1)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = torch.logical_or(node_type_batch == FRightZMatch.index, node_type_batch == FRightXMatch.index)
        phase_probs = x
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(self.num_possible_phases, device=x.device)
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
        node_probs = torch.fill(torch.empty(x.shape[0], self.num_node_types, x.shape[1]), -torch.inf)
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
        # Reshape the tensor to have the batch dimension
        mixture_params = mixture_params.view(-1, self.num_node_types)
        # Probability of selecting a boundary node should be zero
        mixture_params[:, 0] = -torch.inf
        # Apply softmax to get the mixture probabilities
        mixture_params = torch.softmax(mixture_params, dim=-1)
        return mixture_params

    def forward(self, data: pyg.data.Data) -> AlphaZXDistributionParams:
        """
        TODO: Ensure that removing connected components from the ZXDiagram does not affect 'Data' batching.
        :param data: The pyg.data.Data object representing the ZXMatchDiagram.
        :return: Parameters for the AlphaZXDistribution. Each value in the returned dictionary is a batch of distribution
                 parameters.
        """
        x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        mixture_probs = self._compute_mixture_probs(x, data.node_type, data.batch)
        node_probs = self._compute_node_probs(x, data.node_type, data.batch)
        phase_probs = self._compute_phase_probs(x, data.node_type, data.batch)
        print('phase_probs = ', phase_probs)
        transfer_edge_probs = self._compute_transfer_edge_probs(x, data.edge_index, data.node_type, data.batch)
        return AlphaZXDistributionParams(mixture_probs,
                                         node_probs,
                                         phase_probs,
                                         self._compute_new_edge_probs(x, data.node_type).unsqueeze(dim=0),
                                         transfer_edge_probs)


def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))


def batch_node_type_test():
    node_type = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 10, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 10, 10])
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    node_type = node_type + len(NODE_METADATA) * batch
    print(node_type)


# batch_node_type_test()

def reshape_test():
    num_node_types = 11
    mixture_params = torch.tensor(
        [0.3473, 0.3252, 0.3170, 0.3252, 0.2990, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan,
         0.3451, 0.3349, 0.3056, 0.3277, 0.3001, 0.3152])
    mixture_params[torch.isnan(mixture_params)] = -torch.inf
    inf_indices = torch.where(mixture_params == -torch.inf)[0]
    if len(inf_indices) > 0:
        pad_length = num_node_types - (len(mixture_params) - inf_indices[-1] - 1)
        mixture_params = torch.nn.functional.pad(mixture_params, (0, pad_length), mode='constant', value=-torch.inf)
    mixture_params = mixture_params.view(-1, num_node_types)
    print('mixture_params = ', mixture_params)
    mixture_params = torch.softmax(mixture_params, dim=-1)
    print('mixture_params = ', mixture_params)
    output = torch.tensor(
        [[0.3473, 0.3252, 0.3170, 0.3252, 0.2990, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
         [0.3451, 0.3349, 0.3056, 0.3277, 0.3001, 0.3152, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]])


# reshape_test()

def node_batch_test():
    num_node_types = 11
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    node_types = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 5, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4])
    node_type_batch, mask = pyg.utils.to_dense_batch(node_types, batch, torch.nan)
    print('node_type_batch = ', node_type_batch)
    x = torch.tensor([[-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325,
                       -0.1559, -0.1649, -0.1576, -0.1599],
                      [-0.1552, -0.1329, -0.1384, -0.1322, -0.1441, -0.1450, -0.1415, -0.1636, -0.1276, -0.1508,
                       -0.1276, -0.1355, -0.1708, -0.1495]])
    input = torch.zeros((x.shape[0], num_node_types, x.shape[1]))
    print('input = ', input)
    print(input.scatter(1, node_type_batch.unsqueeze(dim=-1), x.unsqueeze(dim=-1)))
    # output = torch.tensor([[[-0.1566, 0., ..., 0.],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                        [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599]],
    #                        [[-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599],
    #                         [-0.1566, -0.1319, -0.1401, -0.1364, -0.1452, -0.1705, -0.1439, -0.1300, -0.1368, -0.1325, -0.1559, -0.1649, -0.1576, -0.1599]]])


# node_batch_test()

def example():
    num_node_types = 7
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    node_types = torch.tensor([0, 1, 3, 4, 4, 0, 1, 1, 5])
    node_type_batch = pyg.utils.to_dense_batch(node_types, batch)[0]
    x = torch.tensor([[-0.1566, -0.1319, -0.1401, -0.1364, -0.1452],
                      [-0.1552, -0.1329, -0.1384, -0.1322, torch.nan]])
    print(torch.scatter(torch.zeros((x.shape[0], num_node_types, x.shape[1])), 1, node_type_batch.unsqueeze(dim=1), x.unsqueeze(dim=1)))
    # expected = torch.tensor([[[-0.1566, 0., 0., 0., 0.],
    #                           [0., -0.1319, 0., 0., 0.],
    #                           [0., 0., 0., 0., 0.],
    #                           [0., 0., -0.1401, 0., 0.],
    #                           [0., 0., 0., -0.1364, -0.1452],
    #                           [0., 0., 0., 0., 0.],
    #                           [0., 0., 0., 0., 0.]],
    #                          [[-0.1552, 0., 0., 0., 0.],
    #                           [0., -0.132, -0.1384, 0., 0.],
    #                           [0., 0., 0., 0., 0.],
    #                           [0., 0., 0., 0., 0.],
    #                           [0., 0., 0., 0., 0.],
    #                           [0., 0., 0., -0.1322, 0.],
    #                           [0., 0., 0., 0., 0.]]])


# example()


def test():
    mask = torch.tensor([[True, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True],
                         [True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True]])
    phase_probs = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [0.0523, 0.0621, 0.0808, 0.0615, 0.0542, 0.0853, 0.0325, 0.0522,
                            0.0576, 0.0574, 0.0550, 0.0856, 0.0497, 0.0573, 0.0993, 0.0573],
                           [0.0565, 0.0661, 0.0754, 0.0612, 0.0557, 0.0747, 0.0325, 0.0491,
                            0.0521, 0.0603, 0.0541, 0.0851, 0.0573, 0.0603, 0.1005, 0.0591],
                           [0.0560, 0.0606, 0.0836, 0.0609, 0.0572, 0.0839, 0.0341, 0.0486,
                            0.0578, 0.0570, 0.0521, 0.0896, 0.0513, 0.0556, 0.0974, 0.0543],
                           [0.0560, 0.0606, 0.0836, 0.0609, 0.0572, 0.0839, 0.0341, 0.0486,
                            0.0578, 0.0570, 0.0521, 0.0896, 0.0513, 0.0556, 0.0974, 0.0543],
                           [0.0545, 0.0599, 0.0816, 0.0607, 0.0564, 0.0864, 0.0341, 0.0490,
                            0.0592, 0.0553, 0.0544, 0.0883, 0.0506, 0.0558, 0.0984, 0.0556],
                           [0.0544, 0.0669, 0.0759, 0.0618, 0.0541, 0.0757, 0.0315, 0.0515,
                            0.0522, 0.0612, 0.0545, 0.0837, 0.0553, 0.0608, 0.1009, 0.0598],
                           [0.0541, 0.0629, 0.0802, 0.0601, 0.0551, 0.0826, 0.0329, 0.0504,
                            0.0561, 0.0580, 0.0551, 0.0858, 0.0522, 0.0583, 0.0984, 0.0578],
                           [0.0574, 0.0647, 0.0798, 0.0611, 0.0568, 0.0764, 0.0330, 0.0486,
                            0.0532, 0.0605, 0.0518, 0.0876, 0.0556, 0.0585, 0.0985, 0.0565],
                           [0.0574, 0.0647, 0.0798, 0.0611, 0.0568, 0.0764, 0.0330, 0.0486,
                            0.0532, 0.0605, 0.0518, 0.0876, 0.0556, 0.0585, 0.0985, 0.0565],
                           [0.0573, 0.0576, 0.0881, 0.0625, 0.0591, 0.0865, 0.0350, 0.0474,
                            0.0606, 0.0556, 0.0485, 0.0941, 0.0489, 0.0520, 0.0966, 0.0502],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                          [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [0.0561, 0.0607, 0.0905, 0.0555, 0.0572, 0.0870, 0.0346, 0.0479,
                            0.0572, 0.0581, 0.0533, 0.0889, 0.0507, 0.0567, 0.0910, 0.0546],
                           [0.0537, 0.0602, 0.0827, 0.0611, 0.0557, 0.0870, 0.0336, 0.0502,
                            0.0592, 0.0559, 0.0539, 0.0880, 0.0494, 0.0557, 0.0982, 0.0554],
                           [0.0549, 0.0621, 0.0795, 0.0610, 0.0560, 0.0824, 0.0334, 0.0494,
                            0.0567, 0.0571, 0.0544, 0.0870, 0.0526, 0.0574, 0.0993, 0.0569],
                           [0.0571, 0.0643, 0.0821, 0.0574, 0.0567, 0.0790, 0.0336, 0.0479,
                            0.0534, 0.0600, 0.0542, 0.0866, 0.0556, 0.0596, 0.0951, 0.0576],
                           [0.0521, 0.0622, 0.0809, 0.0616, 0.0540, 0.0854, 0.0324, 0.0525,
                            0.0576, 0.0574, 0.0550, 0.0854, 0.0495, 0.0574, 0.0993, 0.0573],
                           [0.0580, 0.0661, 0.0799, 0.0608, 0.0567, 0.0742, 0.0326, 0.0487,
                            0.0517, 0.0621, 0.0513, 0.0872, 0.0567, 0.0593, 0.0980, 0.0568],
                           [0.0561, 0.0662, 0.0755, 0.0613, 0.0554, 0.0748, 0.0323, 0.0495,
                            0.0521, 0.0605, 0.0541, 0.0849, 0.0570, 0.0604, 0.1006, 0.0592],
                           [0.0580, 0.0575, 0.0909, 0.0620, 0.0596, 0.0865, 0.0352, 0.0471,
                            0.0604, 0.0563, 0.0472, 0.0952, 0.0485, 0.0515, 0.0950, 0.0491],
                           [0.0539, 0.0671, 0.0760, 0.0619, 0.0537, 0.0759, 0.0313, 0.0521,
                            0.0521, 0.0614, 0.0546, 0.0833, 0.0549, 0.0609, 0.1009, 0.0600],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])