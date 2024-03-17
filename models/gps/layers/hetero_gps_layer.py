from torch import nn
from torch_geometric.nn import to_hetero

from diagram.match import METADATA
from models.gps.gps_model import FeatureEncoder
from models.gps.layers.gps_layer import GPSLayer

# Have to figure out 'cfg' to make the following work.
dim_in = 60
f_encoder = FeatureEncoder(dim_in)
hetero_feature_encoder_layer = to_hetero(f_encoder, METADATA)

# This works without 'cfg'.
hidden_dim = 50
local_gnn_type = 'GAT'
global_model_type = 'Transformer'
num_heads = 10
gps = GPSLayer(hidden_dim, local_gnn_type, global_model_type, num_heads, act=nn.GELU)
hetero_gps_layer = to_hetero(gps, METADATA, debug=True)
