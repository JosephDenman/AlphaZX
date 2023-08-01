from collections.abc import Iterator

import networkx as nx

from graph.pyzx_nx_conv import Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, ETYPE, Z_NTYPE_INDEX, X_NTYPE_INDEX
from matching.match_types import Basis, FRightZMatch, FLeftXMatch, FLeftZMatch, FRightXMatch, FLeftMatch, FRightMatch
from matching.zx_diagram import ZXDiagram


def basis_to_ntype_index(basis: Basis) -> int:
    return Z_NTYPE_INDEX if basis == Z_NTYPE_NAME else X_NTYPE_INDEX


def f_left_pattern(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([0, 1], type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def f_left_z_pattern() -> ZXDiagram:
    return f_left_pattern(Z_NTYPE_NAME)


def f_left_x_pattern() -> ZXDiagram:
    return f_left_pattern(X_NTYPE_NAME)


def f_left_matches(diagram: ZXDiagram) -> Iterator[FLeftMatch]:
    yield from f_left_z_matches(diagram)
    yield from f_left_x_matches(diagram)


def f_left_z_matches(diagram: ZXDiagram) -> Iterator[FLeftZMatch]:
    candidates = set()
    for s, t, edata in diagram.edges(data=True):
        if diagram.is_z_basis(s) and diagram.is_z_basis(t):
            candidates.add((s, t))
    for s, t in candidates:
        if all([edata[ETYPE] == S_ETYPE_INDEX for _, _, _, edata in diagram.edges_between(s, t, data=True)]):
            yield FLeftZMatch({s: 0, t: 1})


def f_left_x_matches(diagram: ZXDiagram) -> Iterator[FLeftXMatch]:
    candidates = set()
    for s, t, edata in diagram.edges(data=True):
        if diagram.is_x_basis(s) and diagram.is_x_basis(t):
            candidates.add((s, t))
    for s, t in candidates:
        if all([edata[ETYPE] == S_ETYPE_INDEX for _, _, _, edata in diagram.edges_between(s, t, data=True)]):
            yield FLeftXMatch({s: 0, t: 1})


def f_right_pattern(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    return ZXDiagram(nx_graph)


def f_right_z_pattern() -> ZXDiagram:
    return f_right_pattern(Z_NTYPE_NAME)


def f_right_x_pattern() -> ZXDiagram:
    return f_right_pattern(X_NTYPE_NAME)


def f_right_matches(diagram: ZXDiagram) -> Iterator[FRightMatch]:
    yield from f_right_z_matches(diagram)
    yield from f_right_x_matches(diagram)


def f_right_z_matches(diagram: ZXDiagram) -> Iterator[FRightZMatch]:
    return (FRightZMatch({z: 0}) for z in diagram.z_nodes())


def f_right_x_matches(diagram: ZXDiagram) -> Iterator[FRightXMatch]:
    return (FRightXMatch({x: 0}) for x in diagram.x_nodes())
