import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool, to_hetero


class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, heads: int = 4, attn_type: str = 'multihead',
                 attn_kwargs=None):
        super().__init__()

        if attn_kwargs is None:
            attn_kwargs = {}
        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=heads, attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.mlp(x)


FLZ = 'flz'
FRZ = 'frz'
FLX = 'flx'
FRX = 'frx'
BL = 'bl'
BR = 'br'
YZL = 'yzl'
YZR = 'yzr'
YXL = 'yxl'
YXR = 'yxr'

I = 'include'
B = 'bridge'

node_types = [FRZ, FLZ, FRX, FLX]
edge_types = [(FLZ, I, FRZ), (FRZ, I, FLZ), (FLX, I, FRX), (FRX, I, FLX), (FRZ, B, FRX), (FRX, B, FRZ)]
meta_data = (node_types, edge_types)

gps = GPS(64, 4, 2)
hetero_gps = to_hetero(gps, meta_data, aggr='sum')
