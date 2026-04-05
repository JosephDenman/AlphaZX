import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.models import softmax_nonzero_entries, throw_on_nan


class NodeSelector(nn.Module):
    """Selects which node to apply a rewrite to, using multi-head scoring.

    The MLP produces ``num_scoring_heads`` independent criterion scores per
    node.  A ReLU non-linearity followed by a learned linear combination
    collapses them into a single logit.  Because each head can independently
    activate or stay silent, the combiner can learn which *patterns* of head
    activations indicate a good rewrite target — making this strictly more
    expressive than a single-output MLP of equivalent depth.
    """

    def __init__(self, node_in_channels: int, num_node_types: int, num_layers: int,
                 dropout: float, num_scoring_heads: int = 8):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_scoring_heads = num_scoring_heads
        self.mlp = pyg.nn.MLP(
            in_channels=node_in_channels, hidden_channels=node_in_channels,
            out_channels=num_scoring_heads, num_layers=num_layers,
            dropout=dropout, norm='layer_norm',
        )
        # Combine H head scores → 1 logit via ReLU + learned weights.
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
        """
        Compute node selection probabilities for each action type.

        Output shape: [B, T, N] where:
        - B = batch size
        - T = num_node_types (10 match types)
        - N = max nodes in batch

        For action type t, node_probs[b, t, :] is a valid probability distribution
        over nodes of type t+1 in batch b. If no such nodes exist, it's all zeros.
        """
        head_scores = self.mlp(x)                                   # [num_nodes, H]
        x = self.head_combine(head_scores).squeeze(-1)              # [num_nodes]

        # Convert to dense batch form
        dense_logits, mask = pyg.utils.to_dense_batch(x, batch)
        B, N = dense_logits.shape

        dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
                                                       fill_value=0)

        # Build type mask: [B, T, N] where type_mask[b, t, n] = True iff node n in batch b has type t+1
        # Using one-hot encoding for efficiency.
        # IMPORTANT: Only match node types 1-10 are valid for selection. Super nodes (11-21),
        # boundary nodes (0), and padding (0) must map to one-hot index 0, which gets excluded
        # by the [:, :, 1:] slice below. We zero out any type outside [1, T] rather than
        # clamping, because clamping would alias super nodes onto the last match type.
        safe_types = dense_node_types.clone()
        safe_types[(safe_types < 1) | (safe_types > self.num_node_types)] = 0
        # One-hot gives [B, N, T+1], we want [B, T, N] for types 1-10 (indices 1 to T in one-hot)
        one_hot = torch.nn.functional.one_hot(safe_types, self.num_node_types + 1)  # [B, N, T+1]
        type_mask = one_hot[:, :, 1:].permute(0, 2, 1).bool()  # [B, T, N], excluding index 0
        # Also mask padding
        type_mask = type_mask & mask.unsqueeze(1)

        # Expand logits to [B, T, N] - same logit for each type dimension
        # The type_mask ensures only correct type gets non-masked value
        expanded_logits = dense_logits.unsqueeze(1).expand(-1, self.num_node_types, -1)

        # Masked softmax: set invalid positions to large negative (not -inf to avoid NaN in backward)
        masked_logits = torch.where(type_mask, expanded_logits,
                                    torch.full_like(expanded_logits, -1e9))

        # Softmax per type (dim=-1 is over nodes)
        node_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Zero out invalid positions (softmax gave them tiny values)
        node_probs = node_probs * type_mask.float()

        # Re-normalize to get valid distributions
        probs_sum = node_probs.sum(dim=-1, keepdim=True)
        node_probs = node_probs / (probs_sum + 1e-10)
        # Zero out types with no valid nodes
        node_probs = node_probs * (probs_sum > 0).float()

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
        x = self.head_combine(self.mlp(x)).squeeze(-1)
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
