import torch

from torch_geometric.nn import SetTransformerAggregation

trans_agg = SetTransformerAggregation(1)

x = torch.tensor([[1], [2], [3]]).float()
index = torch.tensor([0, 0, 1])

print(trans_agg(x, index, dim_size=3, dim=0))