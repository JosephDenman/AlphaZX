import torch.nn
import torch_geometric as pyg

from alphazx.diagram import METADATA
from alphazx.models import softmax_nonzero_entries, compute_non_super_node_mask


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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x).squeeze(dim=-1)
        non_super_node_mask = compute_non_super_node_mask(node_type)
        masked_x = x[non_super_node_mask]
        masked_batch = batch[non_super_node_mask]
        masked_node_type = node_type[non_super_node_mask]
        mixture_probs = torch.zeros([torch.max(batch) + 1, len(METADATA.super_node_type_indices) - 1],
                                    dtype=x.dtype,
                                    device=x.device)
        mixture_probs[masked_batch, masked_node_type - len(METADATA.super_node_type_indices)] = masked_x
        mixture_probs = softmax_nonzero_entries(mixture_probs)
        return mixture_probs

    # def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor,
    #             batch: torch.Tensor) -> torch.Tensor:
    #     throw_on_nan(x)
    #     masked_x, masked_edge_index, masked_node_types, masked_batch = mask_non_super_nodes(node_type, edge_index,
    #                                                                                         node_type, batch)
    #     masked_x, mask = self.mixture_trans(torch.index_select(x, 0, masked_edge_index[0]), masked_edge_index[1])
    #     masked_x = masked_x[torch.any(mask, dim=1)]
    #     masked_x = self.mixture_mlp(masked_x).squeeze(dim=-1)
    #     mixture_probs = torch.full([torch.max(batch) + 1, len(METADATA.super_node_type_indices) - 1],
    #                                0.,
    #                                dtype=x.dtype,
    #                                device=x.device)
    #     mixture_probs[masked_batch, masked_node_types - len(METADATA.super_node_type_indices)] = masked_x
    #     mixture_probs = softmax_nonzero_entries(mixture_probs)
    #     return mixture_probs
