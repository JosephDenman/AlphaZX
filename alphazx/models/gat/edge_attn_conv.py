import torch_geometric as pyg
from torch import Tensor


class EdgeAttnConv(pyg.nn.GATv2Conv):

    def update(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha
