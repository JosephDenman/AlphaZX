import networkx as nx

from alphazx.diagram.match import is_boundary

PHASE = 'phase'
DEGREE = 'degree'
NTYPE = 'type'
COLUMN = 'x'
ROW = 'y'
CONNECTED_TO = 'connected_to'
ETYPE = 'type'


def nx_remove_position_attributes(nx_graph: nx.MultiGraph) -> None:
    for _, ndata in nx_graph.nodes(data=True):
        del ndata[ROW]
        del ndata[COLUMN]


def nx_remove_top_level_attributes(nx_graph: nx.MultiGraph) -> None:
    del nx_graph.graph['node_default']
    del nx_graph.graph['edge_default']


def nx_add_boundary_connected_to(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        if is_boundary(ndata[NTYPE]):
            ndata[CONNECTED_TO] = nx_graph.nodes[list(nx_graph.neighbors(n))[0]][NTYPE]


def nx_add_degree(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        ndata[DEGREE] = len(list(nx_graph.neighbors(n)))


def nx_remove_boundary_phase(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        if is_boundary(ndata[NTYPE]):
            del ndata[PHASE]


def nx_to_pyg_heterograph_pre_process(nx_graph: nx.MultiGraph) -> None:
    nx_remove_position_attributes(nx_graph)
    nx_remove_top_level_attributes(nx_graph)
    nx_add_boundary_connected_to(nx_graph)
    nx_add_degree(nx_graph)
    nx_remove_boundary_phase(nx_graph)
