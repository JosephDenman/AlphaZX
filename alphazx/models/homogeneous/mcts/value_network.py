import torch
import torch.nn as nn
import torch_geometric as pyg


class ValueNetwork(nn.Module):
    def __init__(self,
                 node_embedding_channels: int,
                 edge_embedding_channels: int,
                 hidden_channels: int) -> None:
        super(ValueNetwork, self).__init__()
        self.node_embedding_channels = node_embedding_channels
        self.edge_embedding_channels = edge_embedding_channels
        self.gnn = pyg.nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    pyg.nn.GATv2Conv(node_embedding_channels, hidden_channels, edge_dim=edge_embedding_channels, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(node_embedding_channels, hidden_channels, edge_dim=edge_embedding_channels, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(node_embedding_channels, hidden_channels, edge_dim=edge_embedding_channels, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(node_embedding_channels, hidden_channels, edge_dim=edge_embedding_channels, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(node_embedding_channels, hidden_channels, edge_dim=edge_embedding_channels, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
            ],
        )
        self.global_attention = pyg.nn.GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            ),
            nn=nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels), nn.ReLU()),
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_features=1),
        )

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.global_attention.reset_parameters()
        self.ff.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.node_embedding_channels
        assert edge_attr.shape[-1] == self.edge_embedding_channels, f'Expected {self.edge_embedding_channels}, received {edge_attr.shape}'
        x = self.gnn(x, edge_index, edge_attr)
        x = self.global_attention(x, batch)
        x = self.ff(x)
        return x
