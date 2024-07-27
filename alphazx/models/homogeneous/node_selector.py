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

    # def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     x = self.mlp(x).squeeze(-1)
    #     valid_type_mask = (node_types >= 1) & (node_types <= 10)
    #     valid_type_indices = torch.where(valid_type_mask, True, False)
    #     masked_x = x[valid_type_mask]
    #     masked_batch = batch[valid_type_mask]
    #     masked_node_type = node_types[valid_type_mask]
    #     node_probs = torch.zeros([torch.max(batch) + 1, self.num_node_types, x.shape[0]], device=x.device, dtype=x.dtype)
    #     node_probs[masked_batch, masked_node_type, valid_type_indices] = masked_x
    #     node_probs = softmax_nonzero_entries(node_probs, dim=-1)
    #     throw_on_nan(node_probs)
    #     return node_probs

    # def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     x = self.mlp(x).squeeze(-1)
    #     valid_type_mask = (node_types >= 1) & (node_types <= 10)
    #
    #     valid_type_indices = torch.where(valid_type_mask, True, False)
    #     #print('valid_type_indices = ', valid_type_indices)
    #     masked_x = x[valid_type_mask]
    #     #print('valid_type_indices = ', pyg.utils.to_dense_batch(masked_x, node_types[valid_type_mask], fill_value=torch.nan)[0])
    #     masked_batch = batch[valid_type_mask]
    #     masked_node_type = node_types[valid_type_mask]
    #     node_probs = torch.zeros([torch.max(batch) + 1, self.num_node_types, x.shape[0]], device=x.device, dtype=x.dtype)
    #     print('node_probs[masked_batch, masked_node_type, valid_type_indices] = ', node_probs[masked_batch, masked_node_type, valid_type_indices].shape)
    #     print('node_probs[masked_batch, masked_node_type] = ',
    #           node_probs[masked_batch, masked_node_type].shape)
    #     print('masked_x = ', masked_x.shape)
    #     node_probs = torch.masked_scatter(node_probs[masked_batch, masked_node_type], pyg.utils.to_dense_batch(valid_type_mask, node_types, fill_value=torch.nan)[0], masked_x)
    #     node_probs = softmax_nonzero_entries(node_probs, dim=-1)
    #     throw_on_nan(node_probs)
    #     return node_probs

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x).squeeze(-1)
        # Create masks for valid node types (1 to 10 inclusive)
        valid_types_mask = (node_types >= 1) & (node_types <= 10)
        masked_logits = x * valid_types_mask.float()

        # Convert node logits and node types to dense batch form
        dense_logits, _ = pyg.utils.to_dense_batch(masked_logits, batch)
        T = 22
        B, N = dense_logits.shape
        dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
                                                       fill_value=torch.nan)

        # Create output tensor
        node_probs = torch.zeros((B, T, N), device=x.device)

        # Mask out the logits of invalid types and apply softmax along the node dimension
        valid_types_mask_dense = (dense_node_types >= 1) & (dense_node_types <= 10)
        valid_logits = dense_logits * valid_types_mask_dense.float()

        # Generate the type-specific mask
        type_indices = torch.arange(1, 11, device=x.device).view(1, -1, 1)
        type_mask = (dense_node_types.unsqueeze(1) == type_indices).float()

        # Multiply probabilities with type mask to scatter into correct type slots
        expanded_probs = valid_logits.unsqueeze(1) * type_mask
        node_probs[:, 1:11, :] = expanded_probs
        node_probs = softmax_nonzero_entries(node_probs, dim=-1)
        throw_on_nan(node_probs)
        return node_probs

