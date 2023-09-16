import torch

from game.zx_game import ZXGame
from models.match_metadata import match_metadata_dict
from models.model import HGTPolicy


import networkx as nx
from diagram.pyzx_nx_conv import nx_to_pyg_hetero

def model_test():
    game = ZXGame(10, 10)
    hdata = game.reset()

    diagram_hgt_params = (64, 4, 2, 1, 'sum')

    model = HGTPolicy(hdata.metadata(),
                      diagram_hgt_params,
                      match_metadata_dict,
                      {key: diagram_hgt_params for key in match_metadata_dict.keys()})

    hdata = hdata.to('cpu')
    model = model.to('cpu')

    with torch.no_grad():  # Initialize lazy modules.
        out = model(hdata)

def subgraph_test():
    g = nx.MultiGraph()
    g.add_node(0, type='A', phase=1.5)
    g.add_node(1, type='B', phase=-1.5)
    g.add_node(2, type='C', phase=0.5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    hdata = nx_to_pyg_hetero(g, 'type', group_node_attrs=['phase'])
    sub_hdata = hdata.subgraph({'A': torch.tensor([], dtype=torch.long),
                                'B': torch.tensor([], dtype=torch.long),
                                'C': torch.tensor([0], dtype=torch.long)})
    print('sub_hdata = ', sub_hdata)
    print('sub_hdata[A] = ', sub_hdata['A'])
    print('sub_hdata[B] = ', sub_hdata['B'])
    print('sub_hdata[C] = ', sub_hdata['C'])

model_test()