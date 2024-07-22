import torch
import torch_geometric as pyg
from alphazx.diagram import METADATA
from alphazx.models import softmax_nonzero_entries

x = torch.tensor([-0.4647, -0.4772, -0.3914, -0.3798, -0.2297, -0.2597, -0.6460, -1.2113, -0.4302, -0.0211, -0.2275, -0.0835, -0.3567, -0.4225, -0.3966, -0.7417, -0.4364, -0.5343, -0.5240, -0.5887, -0.2446,
                  -0.7013, -0.0506, -0.3327, -0.4267, -0.4139, -0.2123, -0.4170, -0.2189, -0.4212, -0.3095, -1.1701, -0.2314, -0.2300, -0.5756, -0.5119, -0.2739, -0.8951, -0.6868, -0.2886, -0.9617, -0.5939,
                  -0.5226, -0.4528, -0.1946, -0.5202, -0.2774, -0.3862, -0.5099, -0.2337, -0.2783, -0.5421, -0.2420, -0.1387, -0.2715, -0.4884, -0.4538, -0.8763, -0.5970, -0.6677, -0.5461, -0.3417, -0.2896,
                  -0.2288, -0.3149, -0.4276, -1.6158, -0.5295, -0.5355, -0.9165, -0.8799, -0.3323, -0.4894, -0.3408, -0.2629, -0.1710, -0.6135, -0.3945, -0.6793, -0.3077, -0.5163, -0.2439, -0.4473, -0.6271,
                  -0.7259, -0.5078])
node_types = torch.tensor([0, 11, 0, 1, 12, 2, 13, 1, 2, 1, 2, 1, 2, 0, 0, 3, 14, 3, 4, 15, 4,
                           0, 11, 0, 1, 12, 2, 13, 1, 2, 2, 1, 2, 0, 0, 3, 14, 5, 16, 4, 15, 4,
                           0, 11, 0, 1, 12, 2, 13, 1, 2, 1, 2, 1, 2, 0, 0, 5, 16, 3, 14, 4, 15,
                           0, 11, 0, 1, 12, 2, 13, 1, 2, 1, 2, 1, 2, 0, 0, 3, 14, 5, 16, 4, 15,
                           3, 4])
batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                      3, 3])

#
# # def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
# #     x = self.mlp(x).squeeze(-1)
# #     valid_type_mask = (node_types >= 1) & (node_types <= 10)
# #     x[~valid_type_mask] = 0.
# #     dense_x, _ = pyg.utils.to_dense_batch(x, batch)
# #     T = self.num_node_types
# #     B, N = dense_x.shape
# #     dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N,
# #                                                    fill_value=torch.nan)
# #     node_probs = torch.zeros([B, T, N], device=x.device, dtype=x.dtype)
# #     valid_type_broadcast = (dense_node_types.unsqueeze(1) == torch.arange(0, T, device=x.device).view(1, -1, 1))
# #     node_probs = torch.masked_scatter(node_probs, valid_type_broadcast, dense_x)
# #     node_probs = softmax_nonzero_entries(node_probs, dim=-1)
# #     # node_probs = node_probs[:, 1:11, :]
# #     throw_on_nan(node_probs)
# #     return node_probs
#
#
def forward(x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    valid_type_mask = (node_types >= 12) & (node_types <= 21)
    x[~valid_type_mask] = 0.
    masked_x = x[valid_type_mask]
    masked_batch = batch[valid_type_mask]
    masked_node_type = node_types[valid_type_mask]

    dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=torch.nan)
    T = 22
    B, N = dense_node_types.shape
    node_probs = torch.zeros([B, T, N], device=x.device, dtype=x.dtype)
    dense_valid_node_type_idxs = torch.where((dense_node_types >= 12) & (dense_node_types <= 21))
    print('dense_node_types = ', dense_node_types)
    print('dense_valid_node_type_idxs = ', dense_valid_node_type_idxs)
    node_probs[masked_batch, masked_node_type, dense_valid_node_type_idxs]
    return

"""
dense_node_types =  tensor([[ 0, 11,  0,  1, 12,  2, 13,  1,  2,  1,  2,  1,  2,  0,  0,  3, 14,  3,
          4, 15,  4,  0,  0],
        [ 0, 11,  0,  1, 12,  2, 13,  1,  2,  2,  1,  2,  0,  0,  3, 14,  5, 16,
          4, 15,  4,  0,  0],
        [ 0, 11,  0,  1, 12,  2, 13,  1,  2,  1,  2,  1,  2,  0,  0,  5, 16,  3,
         14,  4, 15,  0,  0],
        [ 0, 11,  0,  1, 12,  2, 13,  1,  2,  1,  2,  1,  2,  0,  0,  3, 14,  5,
         16,  4, 15,  3,  4]])
"""

print(forward(x, node_types, batch))

# mixture_probs = torch.zeros([B, T], device=x.device, dtype=x.dtype)
# dense_x, _ = pyg.utils.to_dense_batch(x, batch)
# print('dense_x.shape = ', dense_x.shape)
# T = 22
# B, N = dense_x.shape
# dense_node_types, _ = pyg.utils.to_dense_batch(node_types, batch, batch_size=B, max_num_nodes=N, fill_value=torch.nan)
# print('dense_node_types.shape = ', dense_node_types.shape)
# mixture_probs = torch.zeros([B, T], device=x.device, dtype=x.dtype)
# print('mixture_probs.shape = ', mixture_probs.shape)
# valid_type_broadcast = ((dense_node_types >= 12) & (dense_node_types <= 21))
# print('valid_type_broadcast.shape = ', valid_type_broadcast.shape)
# print('valid_type_broadcast = ', valid_type_broadcast)
# torch.tensor([[0, 1, 2, 3, 4, 5, torch.nan],
#               []])
# mixture_probs = torch.masked_scatter(mixture_probs, dense_node_types, dense_x)
# print('mixture_probs = ', mixture_probs)
# mixture_probs = softmax_nonzero_entries(mixture_probs, dim=-1)