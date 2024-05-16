from typing import Callable, Any, Optional

import torch
import torch_geometric as pyg

from alphazx import mask_non_basis_edges, concatenate_neighbor_features
from alphazx.models.aggregation.neighbor_multiset_transformer import NeighborMultisetTransformer


class TransferEdgeSelector(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 gmt_k: int,
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
        self.nmt = NeighborMultisetTransformer(in_channels, gmt_k, gmt_num_encoder_blocks, gmt_heads, gmt_layer_norm,
                                               gmt_dropout)
        self.mlp = pyg.nn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=1,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout, act=mlp_act, act_first=mlp_act_first,
            act_kwargs=mlp_act_kwargs, norm=mlp_norm, norm_kwargs=mlp_norm_kwargs, plain_last=mlp_plain_last,
            bias=mlp_bias)

    def reset_parameters(self):
        self.nmt.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        # We only want to aggregate along edges between basis nodes
        edge_index = mask_non_basis_edges(edge_index, node_types)
        # Aggregate along edges between basis nodes
        neighbor_x = self.nmt(x, edge_index)
        # Concatenate the neighbor features according to the target nodes
        neighbor_x = concatenate_neighbor_features(neighbor_x, edge_index)
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
