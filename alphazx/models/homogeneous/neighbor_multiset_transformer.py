import torch
import torch_geometric as pyg

from alphazx.models import throw_on_nan


class NeighborMultisetTransformer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 k: int,
                 num_encoder_blocks: int = 1,
                 heads: int = 1,
                 layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.gmt = pyg.nn.SetTransformerAggregation(in_channels, k, num_encoder_blocks, num_encoder_blocks, heads,
                                                    # TODO: Reevaluate whether 'False' is correct.
                                                    False, layer_norm, dropout)

    def reset_parameters(self):
        self.gmt.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        throw_on_nan(x)
        # Get neighbor features
        x = torch.index_select(x, 0, edge_index[0])
        # Aggregate neighbor features according to the central node
        x = self.gmt(x, edge_index[1])
        return x
