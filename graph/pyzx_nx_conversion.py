from fractions import Fraction
from typing import Union

import torch
import torch_geometric as pyg
import networkx as nx

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


def node_phases_to_ints(nx_graph: nx.MultiGraph) -> None:
    for node in nx_graph.nodes:
        phase_string = nx_graph.nodes[node][PHASE]
        phase_float = float(Fraction(phase_string))
        nx_graph.nodes[node][PHASE] = phase_float


PYG_ETYPE_NAMES_TO_INDICES = {
    (Z_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME): 0,
    (Z_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME): 1,
    (Z_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME): 2,
    (Z_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME): 3,
    (Z_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME): 4,
    (Z_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME): 5,
    (X_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME): 6,
    (X_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME): 7,
    (X_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME): 8,
    (X_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME): 9,
    (X_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME): 10,
    (X_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME): 11,
    (B_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME): 12,
    (B_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME): 13,
    (B_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME): 14,
    (B_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME): 15,
    (B_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME): 16,
    (B_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME): 17
}


def edge_type_index(nx_graph: nx.MultiGraph, u: int, v: int, etype: int) -> int:
    return PYG_ETYPE_NAMES_TO_INDICES[
        (NTYPE_NAMES[nx_graph.nodes[u][NTYPE]], ETYPE_NAMES[etype],
         NTYPE_NAMES[nx_graph.nodes[v][NTYPE]])]


def remove_attributes(nx_graph: nx.MultiGraph) -> None:
    for _, ndata in nx_graph.nodes(data=True):
        del ndata[ROW]
        del ndata[COLUMN]
    del nx_graph.graph['node_default']
    del nx_graph.graph['edge_default']


def add_connected_to(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        if is_boundary(ndata[NTYPE]):
            ndata[CONNECTED_TO] = nx_graph.nodes[list(nx_graph.neighbors(n))[0]][NTYPE]
        else:
            ndata[CONNECTED_TO] = 0


def add_degree(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        ndata[DEGREE] = len(list(nx_graph.neighbors(n)))


def node_types(nx_graph: nx.MultiGraph) -> torch.Tensor:
    return torch.tensor([t for _, t in nx_graph.nodes(data=NTYPE)])


def edge_types(nx_graph: nx.MultiGraph) -> torch.Tensor:
    return torch.tensor([edge_type_index(nx_graph, u, v, t) for u, v, t in nx_graph.edges(data=NTYPE)])


def remove_basis_connected_to(hdata: pyg.data.HeteroData) -> None:
    hdata.update_tensor(hdata[Z_NTYPE_NAME].x[:, :3], Z_NTYPE_NAME, 'x', 0)
    hdata.update_tensor(hdata[X_NTYPE_NAME].x[:, :3], X_NTYPE_NAME, 'x', 0)


def remove_boundary_phase(hdata: pyg.data.HeteroData) -> None:
    boundary_feat = hdata[B_NTYPE_NAME].x
    hdata.update_tensor(torch.cat((boundary_feat[:, :1], boundary_feat[:, 2:]), dim=1), B_NTYPE_NAME, 'x', 0)


def node_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def edge_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def post_process(hdata: pyg.data.HeteroData) -> None:
    remove_basis_connected_to(hdata)
    remove_boundary_phase(hdata)
    node_type_to_one_hot(hdata)
    edge_type_to_one_hot(hdata)
    # TODO: Remove boundary node phase
    # hdata.validate()
    # hdata.generate_ids()
    # boundary_store = hdata.re('boundary')
    # print('store = ', boundary_store)
    # boundary_store.apply(lambda t: t[:, :2], 'x')
    # print('store = ', boundary_store)


# TODO: Use one-hot encoding for edge and node types
# TODO: Use one-hot encoding for node degree before training and inference, not in environment.
def nx_to_pyg_heterograph(nx_graph: nx.MultiGraph) -> pyg.data.HeteroData:
    assert not nx_graph.is_directed(), "Graph must be undirected"
    hdata = pyg.utils.from_networkx(
        nx_graph,
        group_node_attrs=[NTYPE, PHASE, DEGREE, CONNECTED_TO],
        group_edge_attrs=[ETYPE]).to_heterogeneous(node_type=node_types(nx_graph),
                                                    edge_type=edge_types(nx_graph),
                                                    node_type_names=NTYPE_NAMES,
                                                    edge_type_names=list(
                                                        PYG_ETYPE_NAMES_TO_INDICES.keys()))
    post_process(hdata)
    return hdata
