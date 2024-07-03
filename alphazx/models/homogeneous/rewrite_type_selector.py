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
        self.mixture_mlp = pyg.nn.MLP([node_embedding_channels * self.num_node_types, 1])
        self.mixture_trans = pyg.nn.SetTransformerAggregation(node_embedding_channels,
                                                              num_node_types,
                                                              pooling_encoder_blocks,
                                                              pooling_encoder_blocks,
                                                              pooling_heads,
                                                              True,
                                                              pooling_layer_norm,
                                                              pooling_dropout)
        self.sag = pyg.nn.SAGPooling(node_embedding_channels)

    def reset_parameters(self):
        self.mixture_mlp.reset_parameters()
        self.mixture_trans.reset_parameters()

    # def forward(self, x: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     # TODO: An error in this function is producing non-zero values for rewrite types that do not exist in the graph.
    #     #       The distribution samples the rewrite type and since the probability of sample a node of that type is an
    #     #       all zero tensor, the softmax produces NaNs. Need to insert the MLP output into a tensor that
    #     #       contains all zeros for node types that don't exist.
    #     # TODO: Many of the rows in the input tensor are identical. The positional encoding from GPS should prevent this.
    #     throw_on_nan(x)
    #     # We need to compute the mixture probabilities for each batch separately, so we modify node types so that the
    #     # same node type in different batches is represented by a different index
    #     index = node_type + self.num_node_types * batch
    #
    #     print('batch.shape = ', batch.shape)
    #     print('batch = ', batch)
    #
    #     print('node_type.shape = ', node_type.shape)
    #     print('node_type = ', node_type)
    #
    #     print('index.shape = ', index.shape)
    #     print('index = ', index)
    #
    #     print('index.unique.shape = ', len(torch.unique(index)))
    #     print('index.unique = ', torch.unique_consecutive(index))
    #
    #     print('x.shape = ', x.shape)
    #     # print('x = ', x)
    #
    #     print('node_type.shape = ', torch.unique_consecutive(node_type).shape)
    #     print('node_type = ', torch.unique_consecutive(node_type))



    def forward(self, x: torch.Tensor, node_type: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # TODO: An error in this function is producing non-zero values for rewrite types that do not exist in the graph.
        #       The distribution samples the rewrite type and since the probability of sample a node of that type is an
        #       all zero tensor, the softmax produces NaNs. Need to insert the MLP output into a tensor that
        #       contains all zeros for node types that don't exist.
        # TODO: Many of the rows in the input tensor are identical. The positional encoding from GPS should prevent this.
        throw_on_nan(x)
        # We need to compute the mixture probabilities for each batch separately, so we modify node types so that the
        # same node type in different batches is represented by a different index
        index = node_type + self.num_node_types * batch

        print('batch.shape = ', batch.shape)
        print('batch = ', batch)

        print('node_type.shape = ', node_type.shape)
        print('node_type = ', node_type)

        print('index.shape = ', index.shape)
        print('index = ', index)

        print('index.unique.shape = ', len(torch.unique(index)))
        print('index.unique = ', torch.unique_consecutive(index))

        print('x.shape = ', x.shape)
        # print('x = ', x)

        print('node_type.shape = ', torch.unique_consecutive(node_type).shape)
        print('node_type = ', torch.unique_consecutive(node_type))

        # Aggregate the node embeddings by node type
        # Do not index into the result of this function with 'index' - it results in duplicate values
        mixture_params = self.mixture_trans(x, index)
        print('mixture_params.shape = ', mixture_params.shape)
        print('mixture_params = ', mixture_params)
        # print('mixture_params = ', mixture_params)
        # Project the final embeddings to a scalar
        mixture_params = self.mixture_mlp(mixture_params).squeeze(dim=-1)
        throw_on_nan(mixture_params)
        print('mixture_params.shape = ', mixture_params.shape)
        print('mixture_params = ', mixture_params)

        batch_uniques, indices = torch.unique_consecutive(batch, return_counts=True)
        mixture_params, mask = pyg.utils.to_dense_batch(mixture_params, batch_uniques)
        row_mask = torch.any(mask, dim=1)
        mixture_params = mixture_params[row_mask]
        mask = mask[row_mask]

        # print('mixture_params.shape = ', mixture_params.shape)
        # print('mixture_params = ', mixture_params)
        # mixture_params = torch.scatter(mixture_params, 0, node_type)
        # print('mixture_params.shape = ', mixture_params.shape)
        # # print('mixture_params.mask = ', mask)
        # print('mixture_params = ', mixture_params)
        # row_mask = torch.any(mask, dim=1)
        # mixture_params = mixture_params[row_mask]
        # mask = mask[row_mask]
        # print('mixture_params.shape = ', mixture_params.shape)
        # print('mixture_params = ', mixture_params)
        # Replace nan values produced by the aggregation with negative infinities
        # mixture_params[torch.isnan(mixture_params)] = -torch.inf
        # The last negative infinity in the tensor is the last element before the start of the last batch
        # inf_indices = torch.where(mixture_params == -torch.inf)[0]
        # if len(inf_indices) > 0:
        #     # Pad the last batch with negative infinity so that all batches have the same number of elements
        #     pad_length = self.num_node_types - (len(mixture_params) - inf_indices[-1] - 1)
        #     mixture_params = F.pad(mixture_params, (0, pad_length), mode='constant', value=-torch.inf)
        # else:
        # pad_length = self.num_node_types - len(mixture_params[0])
        # mixture_params = F.pad(mixture_params, (0, pad_length), mode='constant', value=-torch.inf)
        # # TODO: Within batches, each node type is assigned the same embedding, meaning that every node will be equally
        # #       Likely to be sampled. This is a problem.
        # # print('mixture_params.shape = ', mixture_params[mask].shape)
        # # print('mixture_params = ', mixture_params[mask])
        # # Reshape the tensor to have the batch dimension
        # mixture_params = mixture_params.view(-1, self.num_node_types)
        # # Probability of selecting a boundary node should be zero
        # mixture_params[:, 0] = -torch.inf
        # Apply softmax to get the mixture probabilities
        mixture_params = torch.softmax(mixture_params, dim=-1)
        # print('rts.batch = ', batch)
        # print('rts.mixture_params = ', mixture_params)
        return mixture_params
