import torch
from torch_geometric.nn import to_hetero, GAT

from diagram.diagram_generators import clifford_pyg_hetero_zx_match_diagram
from diagram.match import METADATA


md = clifford_pyg_hetero_zx_match_diagram(10, 10, True, True)


gat = GAT(64, 64, 5, v2=True, add_self_loops=False)
hetero_gps = to_hetero(gat, METADATA, aggr='max')
print(hetero_gps.graph)
