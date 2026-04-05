from typing import Callable, Any, Optional

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.models.aggregation.set_transformer import SetTransformerAggregation
from alphazx.models.utils import concatenate_neighbor_features, \
    compute_non_simple_node_mask


def compute_actual_num_basis_nodes(transfer_probs: torch.Tensor) -> int:
    transfer_probs = transfer_probs.flatten(start_dim=0, end_dim=1)
    unique_rows = torch.unique(transfer_probs, dim=0).shape[0] - 1
    return unique_rows


class TransferEdgeSelector(nn.Module):
    """Predicts P(transfer_edges | node, action_type) with Bernoulli outputs.

    Conditioning on action_type is injected after the SetTransformer
    aggregation: the type embedding is concatenated to each node's aggregated
    neighbour features before the scoring MLP.  The MLP is run once over all
    (node, type) pairs — not T separate times — so the cost is a single
    matmul with an expanded batch dimension.

    Output shape: [B, T, N, E_trans] (Bernoulli probability per neighbour
    edge, per action type, per node, per graph).
    """

    def __init__(
        self,
        in_channels: int,
        num_node_types: int,
        num_action_types: int = 10,
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
        mlp_bias: bool | list[bool] = True,
        type_emb_dim: int = 16,
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_action_types = num_action_types
        self.type_emb = nn.Embedding(num_action_types, type_emb_dim)
        self.neighbor_trans = SetTransformerAggregation(
            in_channels, num_node_types,
            gmt_num_encoder_blocks, gmt_num_encoder_blocks,
            gmt_heads, False, gmt_layer_norm, gmt_dropout,
        )
        self.mlp = pyg.nn.MLP(
            in_channels=in_channels + type_emb_dim,
            hidden_channels=mlp_hidden_channels,
            out_channels=1,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout, act=mlp_act, act_first=mlp_act_first,
            act_kwargs=mlp_act_kwargs, norm=mlp_norm, norm_kwargs=mlp_norm_kwargs,
            plain_last=mlp_plain_last, bias=mlp_bias,
        )

    def reset_parameters(self):
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.neighbor_trans.reset_parameters()
        self.mlp.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        :return: [B, T, N, E_trans] transfer-edge Bernoulli probabilities.
        """
        T = self.num_action_types

        # --- Shared SetTransformer aggregation (type-independent) ---
        agg_x = self.neighbor_trans(
            torch.index_select(x, 0, edge_index[0]), edge_index[1],
        )[0]                                                      # [num_nodes, C]

        num_nodes = agg_x.shape[0]
        C = agg_x.shape[1]

        # --- Type-conditioned scoring ---
        # Expand to [num_nodes, T, C+D] then MLP → [num_nodes, T, 1] → [num_nodes, T]
        expanded_x = agg_x.unsqueeze(1).expand(num_nodes, T, C)   # [V, T, C]
        type_embs = self.type_emb.weight                           # [T, D]
        expanded_t = type_embs.unsqueeze(0).expand(num_nodes, T, -1)
        conditioned = torch.cat([expanded_x, expanded_t], dim=-1)  # [V, T, C+D]
        scores = self.mlp(conditioned).squeeze(dim=-1)             # [V, T]

        # --- Per-type neighbour rearrangement ---
        non_simple_node_mask = compute_non_simple_node_mask(node_types)  # [V]
        transfer_per_type = []
        for t in range(T):
            score_t = scores[:, t]                                 # [V]
            nbr_x, nbr_mask = concatenate_neighbor_features(
                score_t, edge_index, batch_size=batch.shape[0],
            )                                                      # [V, max_deg], [V, max_deg]
            nbr_x[nbr_mask] = torch.sigmoid(nbr_x[nbr_mask])
            nbr_x[~non_simple_node_mask] = 0.0
            tp = pyg.utils.to_dense_batch(nbr_x, batch)[0]        # [B, N, E_trans]
            transfer_per_type.append(tp)

        transfer_probs = torch.stack(transfer_per_type, dim=1)     # [B, T, N, E_trans]
        return transfer_probs
