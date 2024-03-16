from torch_geometric.nn import to_hetero

from models.gps.gps_model import GPSModel

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

gps = GPSModel(64, 4)
hetero_gps = to_hetero(gps, meta_data, aggr='sum')
print(hetero_gps.meta)
