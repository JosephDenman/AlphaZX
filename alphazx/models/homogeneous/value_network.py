from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.models.homogeneous import GPS


class ValueNetwork(nn.Module):
    def __init__(self,
                 node_in_channels: int,
                 edge_in_channels: int,
                 hidden_channels: int = 128,
                 gps_num_layers: int = 2,  # Reduced from 4 to prevent over-smoothing
                 gps_heads: int = 4,
                 gps_dropout: float = 0.1,
                 gps_act: str = 'relu',
                 gps_act_kwargs: dict[str, Any] = None,
                 gps_norm: str = 'batch_norm',
                 gps_norm_kwargs: dict[str, Any] = None,
                 gps_attn_type: str = 'multihead',
                 gps_attn_kwargs: dict[str, Any] = None,
                 gps_mlp_hidden_channels: int = 128,
                 gps_mlp_num_layers: int = 2) -> None:
        super(ValueNetwork, self).__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.gps = GPS(node_in_channels,
                       node_in_channels,
                       edge_in_channels,
                       gps_num_layers,
                       gps_heads,
                       gps_dropout,
                       gps_act,
                       gps_act_kwargs,
                       gps_norm,
                       gps_norm_kwargs,
                       gps_attn_type,
                       gps_attn_kwargs,
                       gps_mlp_hidden_channels,
                       gps_mlp_num_layers)
        self.global_attention = pyg.nn.GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            ),
            nn=nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                             nn.Linear(hidden_channels, hidden_channels), nn.ReLU()),
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_features=1),
        )

    def reset_parameters(self):
        self.gps.reset_parameters()
        self.global_attention.reset_parameters()
        self.ff.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.node_in_channels
        assert edge_attr.shape[
                   -1] == self.edge_in_channels, f'Expected {self.edge_in_channels}, received {edge_attr.shape}'
        x = self.gps(x, edge_index, edge_attr, batch)
        x = self.global_attention(x, batch)
        x = self.ff(x)
        return x
