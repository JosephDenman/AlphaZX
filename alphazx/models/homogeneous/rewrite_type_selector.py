import torch.nn
import torch_geometric as pyg

from alphazx.diagram import METADATA
from alphazx.models import softmax_nonzero_entries, compute_non_super_node_mask, throw_on_nan


class RewriteTypeSelector(torch.nn.Module):
    def __init__(self,
                 node_embedding_channels: int,
                 num_node_types: int,
                 pooling_encoder_blocks: int,
                 pooling_heads: int,
                 pooling_layer_norm: bool,
                 pooling_dropout: float):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP(in_channels=node_embedding_channels, hidden_channels=node_embedding_channels,
                              out_channels=1, num_layers=2, dropout=pooling_dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mixture_mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x).squeeze(dim=-1)
        print('x = ', x)
        print('edge_index = ', edge_index)
        print('node_types = ', node_types)
        print('batch = ', batch)
        non_super_node_mask = (node_types >= 12) & (node_types <= 21)
        masked_x = x[non_super_node_mask]
        masked_batch = batch[non_super_node_mask]
        masked_node_type = node_types[non_super_node_mask]
        mixture_probs = torch.zeros([torch.max(batch) + 1, len(METADATA.super_node_type_indices) - 1],
                                    dtype=x.dtype,
                                    device=x.device)
        print('mixture_probs.shape = ', mixture_probs.shape)

        mixture_probs[masked_batch, masked_node_type - len(METADATA.super_node_type_indices)] = masked_x
        mixture_probs = softmax_nonzero_entries(mixture_probs)
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

    def node_selector_forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x).squeeze(-1)
        valid_type_mask = (node_types >= 1) & (node_types <= 10)
        x[~valid_type_mask] = 0.
        dense_x, _ = pyg.utils.to_dense_batch(x, batch)
        T = self.num_node_types
        B, N = dense_x.shape
        dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
                                                       fill_value=torch.nan)
        node_probs = torch.zeros([B, T, N], device=x.device, dtype=x.dtype)
        valid_type_broadcast = (dense_node_types.unsqueeze(1) == torch.arange(0, T, device=x.device).view(1, -1, 1))
        node_probs = torch.masked_scatter(node_probs, valid_type_broadcast, dense_x)
        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        node_probs = node_probs[:, 1:11, :]
        throw_on_nan(node_probs)
        return node_probs
