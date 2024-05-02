from alphazx.diagram.match import METADATA
from torch_geometric.nn import to_hetero, GPSConv, GATv2Conv

gps = GPSConv(64, GATv2Conv(-1, 3))
hetero_gps = to_hetero(gps, METADATA, aggr='sum', debug=True)
print(hetero_gps.meta)
