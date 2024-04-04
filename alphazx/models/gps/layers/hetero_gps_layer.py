from diagram.match import METADATA
from torch_geometric.nn import to_hetero, GPSConv, TransformerConv

# Have to figure out 'cfg' to make the following work.
"""dim_in = 60
f_encoder = FeatureEncoder(dim_in)
hetero_feature_encoder_layer = to_hetero(f_encoder, METADATA)"""

in_channels = 10
out_channels = 20
message_passing = TransformerConv(in_channels, out_channels)

channels = 10
gps = GPSConv(channels, message_passing)
hetero_gps_layer = to_hetero(gps, METADATA, debug=True)
