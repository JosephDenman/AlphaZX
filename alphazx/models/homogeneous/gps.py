
from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg


class FeatureEmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 num_node_embeddings: int,
                 node_embedding_out_channels: int,
                 num_edge_embeddings: int,
                 edge_embedding_out_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int,
                 bias: bool = True):
        super(FeatureEmbeddingLayer, self).__init__()
        self.node_emb = nn.Embedding(num_node_embeddings, node_embedding_out_channels)
        self.edge_emb = nn.Embedding(num_edge_embeddings, edge_embedding_out_channels)
        self.pe_norm = nn.BatchNorm1d(pe_in_channels)
        self.pe_lin = nn.Linear(pe_in_channels, pe_out_channels, bias=bias)

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.pe_norm.reset_parameters()
        self.pe_lin.reset_parameters()

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.node_emb(x.long())
        edge_attr = self.edge_emb(edge_attr.long())
        # Cast pe to match the model's parameter dtype.  The model's dtype is
        # determined by torch.get_default_dtype() at construction time, while
        # pe's dtype comes from preprocessing (with_random_walk_pe uses explicit
        # dtype=torch.float).  When these differ — e.g. if another module set
        # the global default to float64 before model creation — BatchNorm raises
        # a mixed-dtype error.  Adapting the input to the model's own parameter
        # dtype is the standard robust pattern.
        pe = pe.to(dtype=self.pe_norm.weight.dtype)
        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)
        x = torch.cat((x, pe), 1)
        return x, edge_attr


class GPS(torch.nn.Module):
    def __init__(self,
                 node_in_channels: int,
                 node_out_channels: int,
                 edge_in_channels: int,
                 gps_num_layers: int = 4,
                 gps_heads: int = 4,
                 gps_dropout: float = 0.1,
                 gps_act: str = 'relu',
                 gps_act_kwargs: dict[str, Any] = None,
                 gps_norm: str = 'batch_norm',
                 gps_norm_kwargs: dict[str, Any] = None,
                 gps_attn_type: str = 'multihead',
                 gps_attn_kwargs: dict[str, Any] = None,
                 mlp_hidden_channels: int = 128,
                 mlp_num_layers: int = 2):
        super(GPS, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(gps_num_layers):
            self.convs.append(
                pyg.nn.GPSConv(node_in_channels,
                               pyg.nn.ResGatedGraphConv(node_in_channels, node_in_channels, edge_dim=edge_in_channels),
                               heads=gps_heads,
                               dropout=gps_dropout,
                               act=gps_act,
                               act_kwargs=gps_act_kwargs,
                               norm=gps_norm,
                               norm_kwargs=gps_norm_kwargs,
                               attn_type=gps_attn_type,
                               attn_kwargs=gps_attn_kwargs))
        self.mlp = pyg.nn.MLP(in_channels=node_in_channels, hidden_channels=mlp_hidden_channels,
                              out_channels=node_out_channels, num_layers=mlp_num_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x_res = x  # Residual connection
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            # Add residual connection to improve gradient flow
            x = x + x_res
        x = self.mlp(x)
        return x
