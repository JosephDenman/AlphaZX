from torch_geometric.nn import to_hetero, GAT

from diagram.diagram_generators import clifford_zx_match_diagram
from diagram.match import METADATA

zx_match_diagram = clifford_zx_match_diagram(10, 10, True).to_pyg_hetero_data()

gat = GAT(-1, 64, 5, 4, v2=True, add_self_loops=False)
hetero_gps = to_hetero(gat, METADATA, aggr='max')
print(zx_match_diagram.x_dict)
print(hetero_gps(zx_match_diagram.x_dict, zx_match_diagram.edge_index_dict))
