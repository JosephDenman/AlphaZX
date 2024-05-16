import torch.nn
import torch_geometric as pyg


class NodeSelector(torch.nn.Module):
    def __init__(self, node_embedding_channels: int, num_node_types: int):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP([node_embedding_channels, 1])

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # TODO: Add a neighbor aggregation
        # Gather node embeddings according to batch
        x, batch_mask = pyg.utils.to_dense_batch(x, batch)
        # Project node embeddings to a scalar representing an un-normalized probability of selecting the node
        x = self.mlp(x).squeeze(dim=-1)
        # Mask padding nodes from to_dense_batch with negative infinity
        x[~batch_mask] = -torch.inf
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch)[0]
        # Initialize the result tensor
        node_probs = torch.fill(torch.empty(x.shape[0], self.num_node_types, x.shape[1], device=x.device), -torch.inf)
        # Scatter selection probabilities to correct column and row, respecting batching
        node_probs = node_probs.scatter(1, node_type_batch.unsqueeze(dim=1), x.unsqueeze(dim=1))
        # Softmax over innermost rows
        node_probs = torch.softmax(node_probs, dim=-1)
        # Sometimes softmax produces NaN, so set those values to zero - no chance of selection.
        node_probs[torch.isnan(node_probs)] = 0.
        # Zero out boundary probabilities
        node_probs[:, 0, 0] = 0.
        return node_probs
