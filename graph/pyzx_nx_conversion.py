from typing import Union

import networkx as nx
import torch
import torch_geometric as pyg

B_NTYPE_INDEX = 0
B_NTYPE_NAME = 'boundary'
Z_NTYPE_INDEX = 1
Z_NTYPE_NAME = 'z'
X_NTYPE_INDEX = 2
X_NTYPE_NAME = 'x'
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


def is_basis(ntype: Union[str, int]) -> bool:
    if isinstance(ntype, str):
        return ntype in [Z_NTYPE_NAME, X_NTYPE_NAME]
    elif isinstance(ntype, int):
        return ntype in [Z_NTYPE_INDEX, X_NTYPE_INDEX]
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_boundary(ntype: Union[str, int]) -> bool:
    if isinstance(ntype, str):
        return ntype == B_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == B_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_hadamard_edge(etype: Union[str, int]) -> bool:
    if isinstance(etype, str):
        return etype == H_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == H_ETYPE_INDEX
    else:
        raise Exception('Unexpected edge type representation ' + str(etype))


def is_simple_edge(etype: Union[str, int]) -> bool:
    return not is_hadamard_edge(etype)


PYG_ETYPE_NAMES = [
    (Z_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (Z_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME),
    (Z_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME)
]

PYG_ETYPE_NAMES_TO_INDICES = {name: i for i, name in enumerate(PYG_ETYPE_NAMES)}


def edge_type_index(nx_graph: nx.MultiGraph, u: int, v: int, etype: int) -> int:
    return PYG_ETYPE_NAMES_TO_INDICES[
        (NTYPE_NAMES[nx_graph.nodes[u][NTYPE]], ETYPE_NAMES[etype],
         NTYPE_NAMES[nx_graph.nodes[v][NTYPE]])]


def node_types(nx_graph: nx.MultiGraph) -> torch.Tensor:
    return torch.tensor([t for _, t in nx_graph.nodes(data=NTYPE)])


def edge_types(nx_graph: nx.MultiGraph) -> torch.Tensor:
    return torch.tensor([edge_type_index(nx_graph, u, v, t) for u, v, t in nx_graph.edges(data=NTYPE)])


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


def node_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def edge_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def nx_to_pyg_heterograph_post_process(hdata: pyg.data.HeteroData) -> None:
    node_type_to_one_hot(hdata)
    edge_type_to_one_hot(hdata)


# TODO: Use one-hot encoding for edge and node types
# TODO: Use one-hot encoding for node degree before training and inference, not in environment.
def nx_to_pyg_heterograph(nx_graph: nx.MultiGraph) -> pyg.data.HeteroData:
    assert not nx_graph.is_directed(), "Graph must be undirected"
    nx_to_pyg_heterograph_pre_process(nx_graph)
    hdata = pyg.utils.from_networkx(
        nx_graph,
        group_node_attrs=[NTYPE, PHASE, DEGREE, CONNECTED_TO],
        group_edge_attrs=[ETYPE]).to_heterogeneous(node_type=node_types(nx_graph),
                                                   edge_type=edge_types(nx_graph),
                                                   node_type_names=NTYPE_NAMES,
                                                   edge_type_names=PYG_ETYPE_NAMES)
    nx_to_pyg_heterograph_post_process(hdata)
    return hdata
