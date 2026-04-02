from typing import Callable, Any, Optional

import torch
import torch_geometric as pyg

from alphazx.models.aggregation.set_transformer import SetTransformerAggregation
from alphazx.models.utils import concatenate_neighbor_features, \
    compute_non_simple_node_mask


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
                 mlp_hidden_channels: int = 64,
                 mlp_num_layers: int = 2,
                 mlp_dropout: float | list[float] = 0.1,
                 mlp_act: Optional[str | Callable] = "relu",
                 mlp_act_first: bool = False,
                 mlp_act_kwargs: Optional[dict[str, Any]] = None,
                 mlp_norm: Optional[str | Callable | None] = "layer_norm",
                 mlp_norm_kwargs: Optional[dict[str, Any]] = None,
                 mlp_plain_last: bool = True,
                 mlp_bias: bool | list[bool] = True):
        super(TransferEdgeSelector, self).__init__()
        self.num_node_types = num_node_types
        self.neighbor_trans = SetTransformerAggregation(in_channels,
                                                        num_node_types,
                                                        gmt_num_encoder_blocks,
                                                        gmt_num_encoder_blocks,
                                                        gmt_heads,
                                                        False,
                                                        gmt_layer_norm,
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
        self.neighbor_trans.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        # TODO: Once https://github.com/pytorch/pytorch/issues/41508 is fixed, mask the non-simple edges at the start.
        x = self.neighbor_trans(torch.index_select(x, 0, edge_index[0]), edge_index[1])[0]
        x = self.mlp(x).squeeze(dim=-1)
        neighbor_x, mask = concatenate_neighbor_features(x, edge_index, batch_size=batch.shape[0])
        neighbor_x[mask] = torch.sigmoid(neighbor_x[mask])
        non_simple_node_mask = compute_non_simple_node_mask(node_types)
        neighbor_x[~non_simple_node_mask] = torch.zeros_like(neighbor_x[~non_simple_node_mask], dtype=x.dtype,
                                                             device=x.device)
        transfer_probs = pyg.utils.to_dense_batch(neighbor_x, batch)[0]
        return transfer_probs

    # def old_forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
    #             batch: torch.Tensor) -> torch.Tensor:
    #     throw_on_nan(x)
    #     masked_edge_index = mask_non_simple_edges(edge_index, node_types)
    #     # TODO: Move MLP to sigmoid section
    #     masked_x = self.neighbor_trans(torch.index_select(x, 0, masked_edge_index[0]), masked_edge_index[1])[0]
    #     masked_x = self.mlp(masked_x).squeeze(dim=-1)
    #     masked_neighbor_x, mask = concatenate_neighbor_features(masked_x, masked_edge_index, batch_size=batch.shape[0])
    #     masked_neighbor_x[mask] = torch.sigmoid(masked_neighbor_x[mask])
    #     assert_unique_elements(masked_neighbor_x)
    #     print('masked_neighbor_x = ', masked_neighbor_x)
    #     transfer_probs = pyg.utils.to_dense_batch(masked_neighbor_x, batch)[0]
    #     return transfer_probs
    #
    # def old_old_forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
    #                 batch: torch.Tensor) -> torch.Tensor:
    #     throw_on_nan(x)
    #     masked_edge_index = mask_non_simple_edges(edge_index, node_types)
    #     masked_x, mask = self.neighbor_trans(torch.index_select(x, 0, masked_edge_index[0]), masked_edge_index[1])
    #     print('masked_x.is_nan() = ', masked_x.isnan())
    #     print('masked_x = ', masked_x[torch.any(mask, dim=1)])
    #     assert_unique_elements(masked_x[torch.any(mask, dim=1)])
    #     row_mask = torch.any(mask, dim=1)
    #     masked_x[mask] = self.mlp(masked_x[mask])
    #     masked_neighbor_x, mask = concatenate_neighbor_features(masked_x, masked_edge_index, batch_size=batch.shape[0])
    #     masked_neighbor_x[mask] = torch.sigmoid(masked_neighbor_x[mask])
    #     transfer_probs = pyg.utils.to_dense_batch(masked_neighbor_x, batch)[0]
    #     print('transfer_probs = ', transfer_probs)
    #     return transfer_probs
