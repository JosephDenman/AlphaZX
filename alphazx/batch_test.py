import torch

import torch_geometric as pyg

edge_index = torch.tensor([[1, 8, 0, 3, 4, 1, 1, 0],
                           [0, 0, 1, 1, 1, 3, 4, 8]])

print(pyg.utils.to_dense_batch(edge_index[0], edge_index[1]))