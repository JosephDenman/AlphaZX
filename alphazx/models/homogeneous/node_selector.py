import torch.nn
import torch_geometric as pyg

from alphazx.models import softmax_nonzero_entries, throw_on_nan


class NodeSelector(torch.nn.Module):
    def __init__(self, node_embedding_channels: int, num_node_types: int, dropout: float):
        super().__init__()
        self.num_node_types = num_node_types
        self.mlp = pyg.nn.MLP(in_channels=node_embedding_channels, hidden_channels=node_embedding_channels,
                              out_channels=1, num_layers=2, dropout=dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
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
        node_probs = torch.masked_scatter(node_probs, valid_type_broadcast, dense_x)
        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        # node_probs = node_probs[:, 1:11, :]
        throw_on_nan(node_probs)
        return node_probs

