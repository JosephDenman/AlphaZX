import torch
import torch.nn as nn
import torch_geometric as pyg


class ValueNetwork(nn.Module):
    def __init__(self,
                 node_embedding_channels: int,
                 gmt_num_encoder_blocks: int,
                 gmt_num_heads: int,
                 gmt_layer_norm: bool,
                 gmt_dropout: float) -> None:
        super(ValueNetwork, self).__init__()
        self.pool = pyg.nn.GraphMultisetTransformer(node_embedding_channels, 1, gmt_num_encoder_blocks, gmt_num_heads,
                                                    gmt_layer_norm, gmt_dropout)
        self.mlp = pyg.nn.MLP([node_embedding_channels, 1])

    def reset_parameters(self):
        self.pool.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.pool(x, batch)
        x = self.mlp(x, batch)
        return x
