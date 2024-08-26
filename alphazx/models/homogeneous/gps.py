import itertools
from typing import Any

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
)
from torch_geometric.nn import GPSConv, TransformerConv, MLP


class GPS(torch.nn.Module):
    def __init__(self,
                 num_node_embeddings: int,
                 node_embedding_out_channels: int,
                 node_out_channels: int,
                 num_edge_embeddings: int,
                 edge_embedding_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 num_layers: int,
                 bias: bool,
                 num_attn_heads: int,
                 attn_type: str,
                 attn_kwargs: dict[str, Any],
                 mlp_hidden_channels: int):
        super().__init__()
        self.node_emb = Embedding(num_node_embeddings, node_embedding_out_channels, dtype=torch.float64)
        self.edge_emb = Embedding(num_edge_embeddings, edge_embedding_out_channels, dtype=torch.float64)
        self.pe_norm = BatchNorm1d(pe_in_channels)
        self.pe_lin = Linear(pe_in_channels, pe_out_channels, bias=bias)
        self.convs = ModuleList()
        node_in_channels = node_embedding_out_channels + pe_out_channels
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(node_in_channels,
                        TransformerConv(node_in_channels,
                                        node_in_channels,
                                        edge_dim=edge_embedding_out_channels,
                                        bias=bias,
                                        heads=1),
                        attn_type=attn_type,
                        attn_kwargs=attn_kwargs,
                        heads=num_attn_heads))
        self.mlp = MLP(in_channels=node_in_channels, hidden_channels=mlp_hidden_channels,
                       out_channels=node_out_channels, num_layers=4, dropout=0.1, norm='layer_norm')

    def sparse_parameters(self):
        return itertools.chain(self.node_emb.parameters(), self.edge_emb.parameters())

    def dense_parameters(self):
        return itertools.chain(self.pe_norm.parameters(), self.pe_lin.parameters(), self.convs.parameters(),
                               self.mlp.parameters())

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.pe_norm.reset_parameters()
        self.pe_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.node_emb(x.long())
        edge_attr = self.edge_emb(edge_attr.long())
        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)
        x = torch.cat((x, pe), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.mlp(x), edge_attr
