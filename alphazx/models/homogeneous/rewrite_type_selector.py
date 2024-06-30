import torch.nn
import torch.nn.functional as F
import torch_geometric as pyg

from alphazx.models import throw_on_nan


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
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.mixture_trans = pyg.nn.GraphMultisetTransformer(node_embedding_channels,
                                                             num_node_types,
                                                             pooling_encoder_blocks,
                                                             pooling_heads,
                                                             pooling_layer_norm,
                                                             pooling_dropout)

    def reset_parameters(self):
        self.mixture_mlp.reset_parameters()
        self.mixture_trans.reset_parameters()

    def forward(self, x: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        throw_on_nan(x)
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
