from typing import Callable, Any, Optional

import torch
import torch_geometric as pyg

from alphazx.models.utils import throw_on_nan, mask_non_basis_nodes, concatenate_neighbor_features, \
    assert_unique_elements, mask_non_basis_edges


def compute_actual_num_basis_nodes(transfer_probs: torch.Tensor) -> int:
    transfer_probs = transfer_probs.flatten(start_dim=0, end_dim=1)
    unique_rows = torch.unique(transfer_probs, dim=0).shape[0] - 1
    return unique_rows


class TransferEdgeSelector(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_node_types: int,
                 gmt_num_encoder_blocks: int = 1,
                 gmt_heads: int = 1,
                 gmt_layer_norm: bool = True,
                 gmt_dropout: float = 0.0,
                 mlp_hidden_channels: int = 2048,
                 mlp_num_layers: int = 1,
                 mlp_dropout: float | list[float] = 0.,
                 mlp_act: Optional[str | Callable] = "relu",
                 mlp_act_first: bool = False,
                 mlp_act_kwargs: Optional[dict[str, Any]] = None,
                 mlp_norm: Optional[str | Callable | None] = "layer_norm",
                 mlp_norm_kwargs: Optional[dict[str, Any]] = None,
                 mlp_plain_last: bool = True,
                 mlp_bias: bool | list[bool] = True):
        super().__init__()
        # self.nmt = NeighborMultisetTransformer(in_channels, gmt_k, gmt_num_encoder_blocks, gmt_heads, gmt_layer_norm,
        #                                        gmt_dropout)
        self.num_node_types = num_node_types
        self.neighbor_trans = pyg.nn.SetTransformerAggregation(in_channels,
                                                               num_node_types,
                                                               gmt_num_encoder_blocks,
                                                               gmt_num_encoder_blocks,
                                                               gmt_heads,
                                                               False,
                                                               gmt_layer_norm,
                                                               gmt_dropout)
        # self.neighbor_trans = pyg.nn.GraphMultisetTransformer(in_channels,
        #                                                       num_node_types,
        #                                                       gmt_num_encoder_blocks,
        #                                                       gmt_heads,
        #                                                       gmt_layer_norm,
        #                                                       gmt_dropout)
        self.mlp = pyg.nn.MLP(
            [in_channels, 1],
            dropout=mlp_dropout, act=mlp_act, act_first=mlp_act_first,
            act_kwargs=mlp_act_kwargs, norm=mlp_norm, norm_kwargs=mlp_norm_kwargs, plain_last=mlp_plain_last,
            bias=mlp_bias)

    def reset_parameters(self):
        self.nmt.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        throw_on_nan(x)
        assert_unique_elements(x)
        old_transfer_probs = self.old_forward(x, edge_index, node_types, batch)
        # print('x = ', x.shape)
        # print('edge_index = ', edge_index.shape)
        # print('edge_index = ', edge_index)
        # We only want to aggregate along edges between basis nodes
        masked_x, masked_edge_index, masked_node_types, masked_batch = mask_non_basis_nodes(x, edge_index, node_types,
                                                                                            batch)
        expected_num_basis_nodes = masked_x.shape[0]
        # print('masked_x = ', masked_x.shape)
        # print('masked_edge_index = ', masked_edge_index.shape)
        # print('masked_edge_index = ', masked_edge_index)
        # print('masked_x = ', masked_x.shape)
        # print('masked_edge_index = ', masked_edge_index)
        masked_x_0 = self.neighbor_trans(torch.index_select(x, 0, masked_edge_index[0]), masked_edge_index[1])[0]
        masked_x_0 = self.mlp(masked_x_0).squeeze(dim=-1)
        print('masked_edge_index = ', masked_edge_index)
        neighbor_fx, mask = concatenate_neighbor_features(masked_x_0, masked_edge_index, max_num_nodes=batch.shape[0])
        row_mask = torch.any(mask, dim=1)
        # print('neighbor_x[row_mask] = ', neighbor_fx[row_mask])
        neighbor_fx[row_mask] = torch.sigmoid(neighbor_fx[row_mask])
        print('neighbor_fx = ', neighbor_fx.shape)
        print('batch = ', batch.shape)
        # print('mask = ', mask)
        transfer_probs = pyg.utils.to_dense_batch(neighbor_fx, batch)[0]
        print('transfer_probs = ', transfer_probs)
        print('old_transfer_probs = ', old_transfer_probs)
        actual_num_basis_nodes = compute_actual_num_basis_nodes(transfer_probs)
        assert actual_num_basis_nodes == expected_num_basis_nodes, f'Actual number of basis nodes {actual_num_basis_nodes} != expected number of basis nodes {expected_num_basis_nodes}'
        # assert torch.allclose(transfer_probs,
        #                       old_transfer_probs), f'Expected new transfer probs:\n\n{transfer_probs}\n\nto equal\n\n{old_transfer_probs}\n\nbut\n\n{old_transfer_probs == transfer_probs}\n\n{transfer_probs.dtype} and {old_transfer_probs.dtype}'
        return transfer_probs
        # print('selected = ', pyg.utils.to_dense_batch(torch.index_select(neighbor_fx, 0, masked_edge_index[0])), masked_edge_index[1])
        # print('neighbor_fx = ', neighbor_fx.shape)
        # print('row_mask = ', torch.any(mask, dim=1))
        # print('masked_node_types = ', masked_node_types.shape)
        # self.neighbor_trans_2(masked_selected_x, masked_edge_index[1])
        # print('concat = ', concatenate_neighbor_features(masked_x_0, edge_index))
        # # masked_x_0 = self.mlp(masked_x_0)
        # print('masked_x_0 = ', masked_x_0.shape)
        # print('mask_0 = ', mask.shape)
        # print('mask_0 = ', mask)
        #
        # print('mask[torch.any(mask, dim=1)] = ', mask[torch.any(mask, dim=1)].shape)
        # print('masked_masked_x_0 = ', masked_x_0[mask].shape)
        # print('masked_masked_x_0 = ', masked_x_0[mask])
        # print('unique_masked_edge_targets = ', torch.unique(masked_edge_index[1]))

        # masked_x_1 = self.neighbor_trans_2(masked_selected_x, masked_edge_index[1])
        # masked_x_1 = self.mlp(masked_x_1)
        # print('masked_x_1 = ', masked_x_1.shape)
        # Aggregate along edges between basis nodes
        # neighbor_x = self.nmt(x, edge_index)
        # # Concatenate the neighbor features according to the target nodes
        # neighbor_x, mask = concatenate_neighbor_features(neighbor_x, edge_index)
        # # print('neighbor_fx_masked = ', self.mlp(neighbor_x[mask], batch).squeeze(dim=-1))
        # neighbor_fx_masked = torch.full((neighbor_x.shape[0], neighbor_x.shape[1]), 0., device=neighbor_x.device,
        #                                 dtype=neighbor_x.dtype)
        # # Project the concatenated features to a scalar
        # neighbor_fx_masked[mask] = self.mlp(neighbor_x[mask], batch).squeeze(-1)
        # # Apply sigmoid to produce valid bernoulli parameters
        # neighbor_fx_masked = torch.sigmoid(neighbor_fx_masked).squeeze(-1)
        # # Create the output tensor
        # transfer_probs = torch.zeros(x.shape[0], neighbor_fx_masked.shape[1], device=x.device)
        # # Identify the target basis nodes - each one should correspond to an aggregated neighborhood
        # # TODO: There must be a more efficient way to do this...
        # target_basis_nodes = torch.unique(edge_index[1])
        # # Insert the computed probabilities for each target basis node into the output tensor
        # transfer_probs[mask] = neighbor_fx_masked
        # # Assert that the output tensor has the same number of nodes as the feature tensor
        # assert transfer_probs.shape[0] == x.shape[
        #     0], f'Expected transfer probabilities {transfer_probs} to have same dimension as {x}, found {transfer_probs.shape[0]} != {x.shape[0]}'
        # # Gather the transfer probabilities according to the node batch
        # transfer_probs = pyg.utils.to_dense_batch(transfer_probs, batch)[0]
        # # old_transfer_probs = self.old_forward(x, edge_index, node_types, batch)
        # # assert torch.allclose(transfer_probs, old_transfer_probs), f'Expected new transfer probs:\n\n{transfer_probs}\n\nto equal\n\n{old_transfer_probs}\n\nbut\n\n{old_transfer_probs == transfer_probs}\n\n{transfer_probs.dtype} and {old_transfer_probs.dtype}'
        # return transfer_probs

    def old_forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # We only want to aggregate along edges between basis nodes
        edge_index = mask_non_basis_edges(edge_index, node_types)
        # Aggregate along edges between basis nodes
        neighbor_x = self.neighbor_trans(torch.index_select(x, 0, edge_index[0]), edge_index[1])[0]
        # Concatenate the neighbor features according to the target nodes
        neighbor_x, mask = concatenate_neighbor_features(neighbor_x, edge_index)
        row_mask = torch.any(mask, dim=1)
        neighbor_x = neighbor_x[row_mask]

        # Project the concatenated features to a scalar
        neighbor_x = self.mlp(neighbor_x, batch).squeeze(dim=-1)
        # Replace NaN padding with -inf
        neighbor_x[neighbor_x.isnan()] = -torch.inf
        # Apply sigmoid to produce valid bernoulli parameters
        neighbor_x = torch.sigmoid(neighbor_x).squeeze(dim=-1)
        # Create the output tensor
        transfer_probs = torch.zeros(x.shape[0], neighbor_x.shape[1], device=x.device)
        # Identify the target basis nodes - each one should correspond to an aggregated neighborhood
        target_basis_nodes = torch.unique(edge_index[1])
        # Insert the computed probabilities for each target basis node into the output tensor
        transfer_probs[target_basis_nodes] = neighbor_x
        # Assert that the output tensor has the same number of nodes as the feature tensor
        assert transfer_probs.shape[0] == x.shape[
            0], f'Expected transfer probabilities {transfer_probs} to have same dimension as {x}, found {transfer_probs.shape[0]} != {x.shape[0]}'
        # Gather the transfer probabilities according to the node batch
        transfer_probs = pyg.utils.to_dense_batch(transfer_probs, batch)[0]
        return transfer_probs
