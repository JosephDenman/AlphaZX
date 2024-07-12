import torch.nn
import torch_geometric as pyg

from alphazx.diagram import METADATA, BoundaryMatch


class NodeSelector(torch.nn.Module):
    def __init__(self, node_embedding_channels: int, num_node_types: int):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP([node_embedding_channels, 1])

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Produces a tensor indicating the probability of selecting each node in the graph. Each innermost row of the output
        contains the parameters for a categorical distribution.

        N = number of nodes in the graph
        B = number of batches
        T = number of possible node types = 21
        B_max_nodes = the maximum number of nodes across all batches

        :param x: An [N] shaped tensor indicating the feature of each node. The entry at position i is the feature of the
                  i-th node of the graph.
        :param node_types: An [N] shaped tensor indicating the type of each node. The entry at position i is the type of the i-th
                           node of the graph. Not necessarily sorted, and types may not be continuous, i.e, there may be
                           node types 1 and 3 but not 2. Node types begin at 0 and end at 21.
        :param batch: An [N] shaped tensor indicating the batch each node belongs to. The entry at position i is the batch that
                     the i-th node of the graph belongs to. This tensor is always sorted in ascending order. The batch numbers
                     are continuous, i.e., batch[j] >= batch[i] + 1. Each batch can have a different number of nodes.
        :return node_probs: A [B, T, B_max_nodes] shaped tensor where each node_probs[b, t] has non-zero entries for each
                            node in batch b of type t and zero-entries everywhere else. When there is at least one node
                            in batch b of type t, the entries in node_probs[b, t] should be normalized so that they
                            represent the parameters of a categorical distribution. If there are no nodes in batch b
                            of type t then node_probs[b, t] should be an all-zero tensor. For all t not between 1 and 10
                            (inclusive) node_probs[b, t] should be all zeros. In other words, only nodes with types between
                            1 and 10 should be samplable.
        """
        x = self.mlp(x).squeeze(dim=-1)
        # Compute mask for nodes that do not have types between 1 and 10 (inclusive)
        valid_type_mask = (node_types >= 1) & (node_types <= 10)

        # Set features of invalid nodes to zero
        x[~valid_type_mask] = 0.0
        # Convert to dense batch format
        x, mask = pyg.utils.to_dense_batch(x, batch)
        node_types, _ = pyg.utils.to_dense_batch(node_types, batch)
        # Initialize the result tensor
        B, N = x.shape
        T = self.num_node_types
        node_probs = torch.zeros((B, T, N), dtype=x.dtype, device=x.device)
        # Create a mask for valid node types between 1 and 10
        valid_type_broadcast = (node_types.unsqueeze(1) == torch.arange(1, 11, device=x.device).view(1, -1, 1))
        # Broadcast x to match the shape of valid_type_broadcast and scatter the probabilities
        x_broad_casted = x.unsqueeze(1).expand(-1, 10, -1)
        node_probs[:, 1:11, :] = torch.where(valid_type_broadcast, x_broad_casted, torch.tensor(-torch.inf, device=x.device))
        # Identify rows that are not all negative infinity and have at least one non-zero entry
        not_neg_inf_mask = (node_probs[:, 1:11, :] > -torch.inf).sum(dim=-1) > 0
        non_zero_mask = node_probs[:, 1:11, :].sum(dim=-1) != 0
        # Combine the masks
        apply_softmax_mask = not_neg_inf_mask & non_zero_mask
        # Apply softmax only to the identified rows
        for i in range(1, 11):
            node_probs[:, i, :][apply_softmax_mask[:, i - 1]] = torch.softmax(
                node_probs[:, i, :][apply_softmax_mask[:, i - 1]], dim=-1)
        return node_probs
