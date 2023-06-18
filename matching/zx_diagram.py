from collections.abc import Iterator

import networkx as nx

from graph.pyzx_nx_conv import nx_add_degree


class ZXDiagram(nx.MultiGraph):

    NTYPE = 'type'
    PHASE = 'phase'
    DEGREE = 'degrees'
    ETYPE = 'type'

    def __init__(self, nx_graph: nx.MultiGraph, **attr):
        nx_add_degree(nx_graph)
        super().__init__(nx_graph, **attr)

    @property
    def node_attrs(self):
        return [self.NTYPE, self.PHASE, self.DEGREE]

    @property
    def edge_attrs(self):
        return [self.ETYPE]