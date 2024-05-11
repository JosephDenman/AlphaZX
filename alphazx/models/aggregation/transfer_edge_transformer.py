from typing import Callable, Any, Optional

import torch
import torch_geometric as pyg

from alphazx import concatenate_neighbor_features


class TransferEdgeTransformer(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 gmt_k: int,
                 gmt_num_encoder_blocks: int = 1,
                 gmt_heads: int = 1,
                 gmt_layer_norm: bool = True,
                 gmt_dropout: float = 0.0,
                 mlp_hidden_channels: int = 2048,
                 mlp_num_layers: int = 1,
                 mlp_dropout: float | list[float] = 0.,
                 mlp_act: Optional[str | Callable] = "relu",
                 mlp_act_first: bool = False,
                 mlp_act_kwargs: Optional[dict[str, Any]] = None,
                 mlp_norm: Optional[str | Callable | None] = "layer_norm",
                 mlp_norm_kwargs: Optional[dict[str, Any]] = None,
                 mlp_plain_last: bool = True,
                 mlp_bias: bool | list[bool] = True):
        super().__init__()
        self.gmt = pyg.nn.GraphMultisetTransformer(in_channels, gmt_k, gmt_num_encoder_blocks, gmt_heads,
                                                   gmt_layer_norm, gmt_dropout)
        self.mlp = pyg.nn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=1,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout, act=mlp_act, act_first=mlp_act_first,
            act_kwargs=mlp_act_kwargs, norm=mlp_norm, norm_kwargs=mlp_norm_kwargs, plain_last=mlp_plain_last,
            bias=mlp_bias)

    def reset_parameters(self):
        self.gmt.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        neighbor_x = torch.index_select(x, 0, edge_index[0])
        neighbor_x = self.gmt(neighbor_x, edge_index[1])
        neighbor_x = concatenate_neighbor_features(neighbor_x, edge_index)
        neighbor_x = self.mlp(neighbor_x, batch)
        neighbor_x[neighbor_x.isnan()] = -torch.inf
        neighbor_x = torch.sigmoid(neighbor_x).squeeze(dim=-1)
        neighbor_x = pyg.utils.to_dense_batch(neighbor_x, batch)[0]
        return neighbor_x
