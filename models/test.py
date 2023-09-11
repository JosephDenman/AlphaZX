import torch

from game.zx_game import ZXGame
from models.match_metadata import match_metadata_dict
from models.model import HGTPolicy

game = ZXGame(100, 100)
hdata = game.reset()

print('hdata = ', hdata)

diagram_hgt_params = (64, 4, 2, 1, 'sum')

model = HGTPolicy(hdata.metadata(), diagram_hgt_params, match_metadata_dict,
                  {key: diagram_hgt_params for key in match_metadata_dict.keys()})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hdata, model = hdata.to(device.type), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(hdata.collect('x'), hdata.collect('edge_index'))
    print('out = ', out)

"""
import torch_geometric as pyg
import networkx as nx
from diagram.pyzx_nx_conv import nx_to_pyg_hetero

g = nx.MultiGraph()
g.add_node(0, type='A', phase=1.5)
g.add_node(1, type='B', phase=-1.5)
g.add_edge(0, 1)

hdata = nx_to_pyg_hetero(g, 'type')

top_hdata = pyg.data.HeteroData({ 'a': hdata, 'b': hdata})

print('top_hdata = ', top_hdata)
print('top_hdata.index = ', top_hdata['a'].x)
"""

"""
import torch_geometric as pyg
import networkx as nx
from diagram.pyzx_nx_conv import nx_to_pyg_hetero

g = nx.DiGraph()

g0 = nx.MultiGraph()
g0.add_node(0, type='A')
g0.add_node(1, type='A')
g0.add_edges_from(4 * [(0, 1)])

g.add_node(g0, type='ARightMatch')

g1 = nx.MultiGraph()
g1.add_node(0, type='A')
g1.add_node(1, type='A')
g1.add_node(2, type='B')
g1.add_node(3, type='B')
g1.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 3)])

g.add_node(g1, type='BRightMatch')

g.add_edge(g0, g1, type=('ARightMatch', 'Bridge', 'BRightMatch'))
g.add_edge(g1, g0, type=('BRightMatch', 'Bridge', 'ARightMatch'))

hdata = nx_to_pyg_hetero(g, node_type_attribute='type', edge_type_attribute='type')
print('hdata = ', hdata['ARightMatch', 'Bridge', 'BRightMatch'])
print('hdata = ', hdata['BRightMatch', 'Bridge', 'ARightMatch'])
"""
