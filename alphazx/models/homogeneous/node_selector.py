import torch.nn
import torch_geometric as pyg

from alphazx.models import softmax_nonzero_entries, throw_on_nan


class NodeSelector(torch.nn.Module):
    def __init__(self, node_embedding_channels: int, num_node_types: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP(in_channels=node_embedding_channels, hidden_channels=node_embedding_channels,
                              out_channels=1, num_layers=num_layers, dropout=dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x).squeeze(-1)

        # Mask invalid node types and set their logits to 0
        valid_types_mask = (node_types >= 1) & (node_types <= 10)
        x[~valid_types_mask] = 0.

        # Convert node logits and node types to dense batch form
        dense_logits = pyg.utils.to_dense_batch(x, batch)[0]
        B, N = dense_logits.shape

        dense_node_types = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
                                                    fill_value=torch.nan)[0].long()

        # Create a tensor for node probabilities with zeros
        node_probs = torch.zeros((B, self.num_node_types, N), device=x.device)

        # Scatter the logits to the correct type dimension
        node_probs.scatter_(1, dense_node_types.unsqueeze(1), dense_logits.unsqueeze(1))

        # Apply softmax to non-zero entries
        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        throw_on_nan(node_probs)

        return node_probs

    # forward_working_primitive
    def working_inefficient_forward(self, x: torch.Tensor, node_types: torch.Tensor,
                                    batch: torch.Tensor) -> torch.Tensor:
        """
        Transforms node features, node types, and batch assignments of all nodes in a graph into a
        [B, T, max_nodes_in_a_batch] shaped tensor indicating the probability of selecting a node
        n of type t in batch b. In other words, the innermost dimension of the result tensor
        is a categorical distribution over possible nodes of type t in the graph.

        Supposing there are only two node types (0, 1, and 2), and three nodes total in the batch such that
        x = [1, 2, 3, 4], node_type = [0, 2, 1, 1], batch = [0, 0, 1, 1], the output tensor is:

        result = torch.tensor([[1., 0.], [0., 0.], [0., 1.]], [[0, 0], [0.3, 0.7], [0, 0]])

        For each batch, the index of each node in the third dimension corresponds to the index of the nodes in
        the batch tensor. For example, the third index for the first two nodes in the batch tensor is 0 and 1,
        therefore the probability of selecting node 0 is result[0, 0, 0] and the probability of selecting node
        1 is result[0, 2, 0 + 1] = result[0, 2, 1].

        N = the number of nodes in the batch
        T = the number of possible types a node can have (most often 22)
        max_nodes_in_a_batch = the maximum number of nodes across all batches passed in

        :param x: [N, F] shaped tensor of node features
        :param node_types: [N] shaped tensor of node types
        :param batch: [N] shaped tensor indicating the batch of each node

        :return:[B, T, max_nodes_in_a_batch] shaped tensor indicating the probability of selecting a node
                n of type t in batch b.
        """
        x = self.mlp(x).squeeze(-1)
        # Create masks for valid node types (1 to 10 inclusive)
        valid_types_mask = (node_types >= 1) & (node_types <= 10)
        x[~valid_types_mask] = 0.

        # Convert node logits and node types to dense batch form
        dense_logits, _ = pyg.utils.to_dense_batch(x, batch)
        T = 22
        B, N = dense_logits.shape
        dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
                                                       fill_value=torch.nan)

        node_probs = torch.zeros((B, T, N), device=x.device)
        batch_offsets = torch.zeros(B, device=x.device, dtype=torch.int64)
        for i in range(x.shape[0]):
            b = batch[i]
            t = node_types[i]
            node_probs[b, t, batch_offsets[b]] = x[i]
            batch_offsets[b] = batch_offsets[b] + 1

        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        throw_on_nan(node_probs)
        return node_probs
