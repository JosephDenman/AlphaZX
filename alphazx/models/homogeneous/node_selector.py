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

    # forward_working_primitive
    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
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
