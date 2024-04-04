import networkx as nx

B_NTYPE_INDEX = 0
B_NTYPE_NAME = 'b_node'
Z_NTYPE_INDEX = 1
Z_NTYPE_NAME = 'z_node'
X_NTYPE_INDEX = 2
X_NTYPE_NAME = 'x_node'
H_NTYPE_INDEX = 3
H_NTYPE_NAME = 'hadamard'
D_ETYPE_INDEX = 0
D_ETYPE_NAME = 'dummy'
S_ETYPE_INDEX = 1
S_ETYPE_NAME = 'simple'
H_ETYPE_INDEX = 2
H_ETYPE_NAME = 'hadamard'

NTYPE_NAMES = [B_NTYPE_NAME, Z_NTYPE_NAME, X_NTYPE_NAME, H_NTYPE_NAME]
ETYPE_NAMES = [D_ETYPE_NAME, S_ETYPE_NAME, H_ETYPE_NAME]

PHASE = 'phase'
DEGREE = 'degree'
NTYPE = 'type'
COLUMN = 'x'
ROW = 'y'
CONNECTED_TO = 'connected_to'
ETYPE = 'type'


def is_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str) or isinstance(ntype, int):
        return is_z_basis(ntype) or is_x_basis(ntype)
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_z_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == Z_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == Z_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_x_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == X_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == X_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_boundary(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == B_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == B_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


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
