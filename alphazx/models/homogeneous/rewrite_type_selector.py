import torch.nn
import torch_geometric as pyg

from alphazx.diagram import METADATA
from alphazx.models import mask_non_super_nodes


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
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels * self.num_node_types, 1])
        self.mixture_trans = pyg.nn.SetTransformerAggregation(node_embedding_channels,
                                                              num_node_types,
                                                              pooling_encoder_blocks,
                                                              pooling_encoder_blocks,
                                                              pooling_heads,
                                                              True,
                                                              pooling_layer_norm,
                                                              pooling_dropout)

    def reset_parameters(self):
        self.mixture_mlp.reset_parameters()
        self.mixture_trans.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        masked_x, masked_edge_index, masked_node_types, masked_batch = mask_non_super_nodes(node_type, edge_index,
                                                                                            node_type, batch)
        masked_selected_x = torch.index_select(x, 0, masked_edge_index[0])
        masked_x = self.mixture_trans(masked_selected_x, masked_edge_index[1])
        masked_x = self.mixture_mlp(masked_x).squeeze(dim=-1)
        mixture_probs = torch.full([torch.max(batch) + 1, len(METADATA.super_node_type_indices)], 0., dtype=x.dtype,
                                   device=x.device)
        mixture_probs[masked_batch, masked_node_types - len(METADATA.super_node_type_indices)] = masked_x
        return mixture_probs
