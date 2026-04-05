import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.models import softmax_nonzero_entries, throw_on_nan


class RewriteTypeSelector(nn.Module):
    """Selects which rewrite (action) type to apply, using multi-head scoring.

    Each super node (one per action type per graph) is scored by an MLP that
    outputs ``num_scoring_heads`` criterion channels.  A ReLU + linear
    combination collapses them to a single logit per super node, then softmax
    over action types gives the mixture probabilities.
    """

    def __init__(self,
                 node_in_channels: int,
                 num_node_types: int,
                 num_layers: int,
                 pooling_dropout: float,
                 num_scoring_heads: int = 8):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_scoring_heads = num_scoring_heads
        self.mlp = pyg.nn.MLP(
            in_channels=node_in_channels, hidden_channels=node_in_channels,
            out_channels=num_scoring_heads, num_layers=num_layers,
            dropout=pooling_dropout, norm='layer_norm',
        )
        self.head_combine = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_scoring_heads, 1, bias=False),
        )

    def reset_parameters(self):
        self.mlp.reset_parameters()
        for m in self.head_combine:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        head_scores = self.mlp(x)                                   # [num_nodes, H]
        x = self.head_combine(head_scores).squeeze(dim=-1)          # [num_nodes]
        # Filter for super nodes (indices 12-21)
        super_node_mask = (node_types >= 12) & (node_types <= 21)
        masked_x = x[super_node_mask]
        masked_batch = batch[super_node_mask]
        masked_node_type = node_types[super_node_mask]

        B = torch.max(batch) + 1
        # Initialize with -inf so softmax gives 0 for unset positions
        mixture_logits = torch.full([B, self.num_node_types], float('-inf'),
                                    dtype=x.dtype, device=x.device)

        # Convert super node indices (12-21) to action type indices (0-9)
        action_type_indices = masked_node_type - 12
        mixture_logits[masked_batch, action_type_indices] = masked_x

        # CRITICAL: Mask out action types that have no valid match nodes
        # This ensures we never sample an action type with no selectable nodes
        match_node_mask = (node_types >= 1) & (node_types <= 10)
        if match_node_mask.any():
            match_batches = batch[match_node_mask]
            match_types = node_types[match_node_mask] - 1  # Convert type 1-10 to index 0-9
            # Create valid (batch, type) indicator using linear indexing
            valid_indices = match_batches * self.num_node_types + match_types
            valid_types_flat = torch.zeros(B * self.num_node_types, dtype=torch.bool, device=x.device)
            valid_types_flat[valid_indices.long()] = True
            valid_types_mask = valid_types_flat.view(B, self.num_node_types)
        else:
            valid_types_mask = torch.zeros(B, self.num_node_types, dtype=torch.bool, device=x.device)

        # Set logits to -inf for action types with no valid match nodes
        mixture_logits = torch.where(valid_types_mask, mixture_logits,
                                     torch.full_like(mixture_logits, float('-inf')))

        # Apply softmax to get probabilities
        mixture_probs = torch.nn.functional.softmax(mixture_logits, dim=-1)
        # Handle NaN from all-inf rows (shouldn't happen in valid diagrams)
        mixture_probs = torch.nan_to_num(mixture_probs, 0.0)

        throw_on_nan(mixture_probs)
        return mixture_probs

    # def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
    #             batch: torch.Tensor) -> torch.Tensor:
    #     x = self.mlp(x).squeeze(-1)
    #     valid_type_mask = (node_types >= 12) & (node_types <= 21)
    #     x[~valid_type_mask] = 0.
    #     dense_x, _ = pyg.utils.to_dense_batch(x, batch)
    #
    #     T = self.num_node_types
    #     B, N = dense_x.shape
    #     dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
    #                                                    fill_value=torch.nan)
    #     rewrite_probs = torch.zeros([B, T], device=x.device, dtype=x.dtype)
    #     non_super_node_mask = compute_non_super_node_mask(node_type)
    #     # masked_x = x[non_super_node_mask]
    #     # masked_batch = batch[non_super_node_mask]
    #     # masked_node_type = node_type[non_super_node_mask]
    #     # mixture_probs = torch.zeros([torch.max(batch) + 1, len(METADATA.super_node_type_indices) - 1],
    #     #                             dtype=x.dtype,
    #     #                             device=x.device)
    #     # mixture_probs[masked_batch, masked_node_type - len(METADATA.super_node_type_indices)] = masked_x
    #     # mixture_probs = softmax_nonzero_entries(mixture_probs)
    #     # return mixture_probs

    # def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     x = self.mlp(x).squeeze(-1)
    #     valid_type_mask = (node_types >= 1) & (node_types <= 10)
    #     x[~valid_type_mask] = 0.
    #     dense_x, _ = pyg.utils.to_dense_batch(x, batch)
    #     T = self.num_node_types
    #     B, N = dense_x.shape
    #     dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
    #                                                    fill_value=torch.nan)
    #     node_probs = torch.zeros([B, T, N], device=x.device, dtype=x.dtype)
    #     valid_type_broadcast = (dense_node_types.unsqueeze(1) == torch.arange(0, T, device=x.device).view(1, -1, 1))
    #     node_probs = torch.masked_scatter(node_probs, valid_type_broadcast, dense_x)
    #     node_probs = softmax_nonzero_entries(node_probs, dim=-1)
    #     # node_probs = node_probs[:, 1:11, :]
    #     throw_on_nan(node_probs)
    #     return node_probs
