import torch.nn
import torch_geometric as pyg

from alphazx.models import assert_not_all_zero, softmax_nonzero_entries, throw_on_nan


def count_unique_2d_tensors(tensor: torch.Tensor) -> int:
    """
    Counts the number of unique inner 2D tensors in a 3D tensor of booleans.

    Args:
    tensor (torch.Tensor): A 3D tensor of booleans with shape (N, H, W).

    Returns:
    int: The number of unique inner 2D tensors.
    """
    # Reshape the 3D tensor to a 2D tensor where each row is a flattened 2D tensor
    N, H, W = tensor.shape
    flattened = tensor.view(N, -1)

    # Find unique rows
    unique_flattened = torch.unique(flattened, dim=0)

    # Return the number of unique rows
    return unique_flattened.size(0)

class NodeSelector(torch.nn.Module):
    def __init__(self, node_embedding_channels: int, num_node_types: int, dropout: float):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP(in_channels=node_embedding_channels, hidden_channels=node_embedding_channels,
                              out_channels=1, num_layers=2, dropout=dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mlp.reset_parameters()

    # def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     print('x.shape = ', x.shape)
    #     B = batch.max().item() + 1
    #     T = self.num_node_types
    #
    #     # Compute node scores using the MLP
    #     node_scores = self.mlp(x).squeeze(-1)
    #     print('x.shape = ', x.shape)
    #     # Create a mask for valid node types (1 to 10)
    #     valid_types_mask = (node_types >= 1) & (node_types <= 10)
    #
    #     # Filter out invalid nodes
    #     valid_node_scores = node_scores[valid_types_mask]
    #     valid_node_types = node_types[valid_types_mask]
    #     valid_batch = batch[valid_types_mask]
    #
    #     # Use to_dense_batch to get dense representation
    #     dense_scores, mask = pyg.utils.to_dense_batch(valid_node_scores, valid_batch)
    #     dense_types, _ = pyg.utils.to_dense_batch(valid_node_types, valid_batch)
    #
    #     B_max_nodes = dense_scores.size(1)
    #
    #     # Initialize node_probs tensor
    #     node_probs = torch.zeros((B, T, B_max_nodes), device=x.device, dtype=x.dtype)
    #     node_indices = torch.arange(x.size(0), device=x.device)
    #
    #
    #    # torch.scatter()
    #
    #
    #
    #
    #
    #
    #
    #     print('B = ', B)
    #     print('T = ', T)
    #     print('B_max_nodes = ', B_max_nodes)
    #
    #     print('node_probs = ', node_probs)
    #     print('node_probs.shape = ', node_probs.shape)
    #     print('node_probs[valid_batch, valid_node_types].shape = ', node_probs[valid_batch, valid_node_types].shape)
    #     print('node_probs[valid_batch, valid_node_types] = ', node_probs[valid_batch, valid_node_types])
    #
    #     # torch.tensor([[["batch_1_node_type_1_node_1", "batch_1_node_type_1_node_2", "batch_1_node_type_1_node_3"],
    #     #                ["batch_1_node_type_2_node_1", "batch_1_node_type_2_node_2", "batch_1_node_type_2_node_3"]],
    #     #               [["batch_2_node_type_1_node_1", "batch_2_node_type_1_node_2", "batch_2_node_type_1_node_3"],
    #     #                ["batch_2_node_type_2_node_1", "batch_2_node_type_2_node_2", "batch_2_node_type_2_node_3"]]])
    #
    #     # Create a mask for valid node types
    #     valid_type_indices = (dense_types >= 1) & (dense_types <= 10)
    #
    #     # Compute softmax probabilities for valid nodes
    #     node_probs = node_probs.masked_scatter(valid_type_indices, 0.)
    #     node_probs = softmax_nonzero_entries(node_probs, dim=1)
    #     print('masked_probs = ', softmax_probs[mask])
    #     # print('softmax_probs = ', softmax_probs)
    #     # # # Scatter the probabilities to the appropriate positions
    #     # scatter_indices = dense_types - 1  # Node types are 1-based, so subtract 1 for 0-based indexing
    #     # node_probs.scatter_add_(1, scatter_indices.unsqueeze(-1).expand(-1, -1, B_max_nodes),
    #     #                         softmax_probs.unsqueeze(1))
    #     #
    #     # # Set probabilities for invalid node types to zero
    #     # invalid_type_indices = ~((dense_types >= 0) & (dense_types < 10))
    #     # node_probs[:, invalid_type_indices, :] = 0
    #
    #     return node_probs

    def scatter_to_node_type_position(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :param node_types:
        :param batch:
        :return:
        """
        print('x.shape = ', x)
        print('node_types = ', node_types)
        print('batch = ', batch)

        valid_type_mask = (node_types >= 1) & (node_types <= 10)
        x[~valid_type_mask] = 0.

        dense_x, mask = pyg.utils.to_dense_batch(x, batch)
        print('x.shape = ', dense_x.shape)

        T = self.num_node_types
        B, N = dense_x.shape

        print('T = ', T)
        print('B = ', B)
        print('N = ', N)

        dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch)
        target = torch.zeros([B, T, N])
        valid_type_broadcast = (dense_node_types.unsqueeze(1) == torch.arange(0, T, device=x.device).view(1, -1, 1))

        print('valid_type_broadcast.shape = ', valid_type_broadcast.shape)
        print('valid_type_broadcast = ', valid_type_broadcast)

        for i in range(len(x)):
            actual = valid_type_broadcast[batch[i], node_types[i], i].item()
            expected = True
            assert actual == expected, f'For {i}, actual {actual} != expected {expected}, batch[i] = {batch[i]}, node_types[i] = {node_types[i]}'

        # print('valid_type_broadcast = ', valid_type_broadcast)

        target = torch.masked_scatter(target, valid_type_broadcast, dense_x)
        print('target = ', target)

        for i in range(len(batch)):
            actual = target[batch[i].item(), node_types[i].item(), i].item()
            expected = x[i].item()
            assert actual == expected, f'For {i}, actual {actual} != expected {expected}'

        pass

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """

            :param x:
            :param node_types:
            :param batch:
            :return:
            """
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

        b = 0
        idx = 0
        for i in range(len(batch)):
            if batch[i] != b:
                idx = 0
            actual = valid_type_broadcast[batch[idx], node_types[idx], idx].item()
            expected = True
            assert actual == expected, f'For {idx}, actual {actual} != expected {expected}, batch[i] = {batch[idx]}, node_types[i] = {node_types[idx]}'
            idx += 1

        node_probs = torch.masked_scatter(node_probs, valid_type_broadcast, dense_x)

        b = 0
        idx = 0
        for i in range(len(batch)):
            if batch[i] != b:
                idx = 0
            actual = node_probs[batch[idx], node_types[idx], idx].item()
            expected = x[idx].item()
            assert actual == expected, f'For {idx}, actual {actual} != expected {expected}'

        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        node_probs = node_probs[:, 1:11, :]
        throw_on_nan(node_probs)
        return node_probs