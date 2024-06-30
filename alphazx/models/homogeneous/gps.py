from typing import Any

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GPSConv, TransformerConv

from alphazx.models import throw_on_nan


class GPS(torch.nn.Module):
    def __init__(self,
                 num_node_embeddings: int,
                 channels: int,
                 node_out_channels: int,
                 edge_in_channels: int,
                 edge_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 num_layers: int,
                 bias: bool,
                 num_attn_heads: int,
                 attn_type: str,
                 attn_kwargs: dict[str, Any],
                 mlp_hidden_channels: int):
        super().__init__()
        self.pe_norm = BatchNorm1d(pe_in_channels)
        self.node_emb = Embedding(num_node_embeddings, channels - pe_out_channels, dtype=torch.float)
        self.pe_lin = Linear(pe_in_channels, pe_out_channels, bias=bias)
        self.edge_lin = Linear(edge_in_channels, edge_out_channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(channels,
                        TransformerConv(channels,
                                        channels,
                                        heads=num_attn_heads),
                        attn_type=attn_type,
                        attn_kwargs=attn_kwargs,
                        heads=num_attn_heads,
                        norm='layer_norm'))
        self.mlp = Sequential(
            Linear(channels, mlp_hidden_channels),
            ReLU(),
            Linear(mlp_hidden_channels, mlp_hidden_channels),
            ReLU(),
            Linear(mlp_hidden_channels, node_out_channels),
        )

    def reset_parameters(self):
        self.pe_norm.reset_parameters()
        self.node_emb.reset_parameters()
        self.pe_lin.reset_parameters()
        self.edge_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        throw_on_nan(x)
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.long().squeeze(-1)), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return self.mlp(x)
