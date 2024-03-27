from torch_geometric.nn import to_hetero, SAGEConv

from diagram.match import METADATA
from models.gps.gps_model import GPSModel

gps = GPSModel(64, 4)
hetero_gps = to_hetero(gps, METADATA, aggr='sum')
print(hetero_gps.meta)
