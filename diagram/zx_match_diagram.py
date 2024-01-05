from typing import Any, Iterator

import networkx as nx
from torch_geometric.data import HeteroData

from diagram.pyzx_nx_conv import ETYPE, nx_to_pyg_hetero
from diagram.zx_diagram import ZXDiagram
from diagram.match import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch

I_ETYPE_INDEX = 3
I_ETYPE_NAME = 'inclusion'
B_ETYPE_INDEX = 4
B_ETYPE_NAME = 'bridge'
NTYPES = 'types'


class ZXMatchDiagram(nx.Graph):
    NTYPE = 'type'
    NTYPES = 'types'
    PHASES = 'phases'

    def __init__(self, diagram: ZXDiagram, **attr):
        self.zx_diagram = diagram
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        super().__init__(nx.Graph(), **attr)


def sub_match_zx_match_diagram(diagram: ZXDiagram) -> ZXMatchDiagram:
    match_diagram = ZXMatchDiagram(diagram)
    matches = list(diagram.compute_matches())
    for match in matches:
        add_match(match_diagram, diagram, match)
    add_composition_edges(match_diagram, diagram)
    num_nodes = match_diagram.number_of_nodes()
    num_matches = len(matches)
    assert num_nodes == num_matches, f'Number of nodes {num_nodes} in match diagram != number of matches ' \
                                     f'{num_matches}'
    return match_diagram


def add_match(match_diagram: ZXMatchDiagram, diagram: ZXDiagram, match: Match) -> None:
    if not match_diagram.has_node(match):
        match_diagram.add_node(match, type=match.name,
                               **collect([diagram.nodes[node] for node in match], ['phase']))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not match_diagram.has_node(sub_match):
                add_match(match_diagram, diagram, sub_match)
            if not match_diagram.has_edge(sub_match, match):
                match_diagram.add_edge(match, sub_match, type=I_ETYPE_NAME)
    return


def add_composition_edges(match_diagram: ZXMatchDiagram, diagram: ZXDiagram) -> None:
    for u in diagram.basis_nodes():
        for v in basis_neighbors(diagram, u):
            u_match = f_right_match_from_ndata(diagram, u)
            v_match = f_right_match_from_ndata(diagram, v)
            if not connected(match_diagram, u_match, v_match):
                match_diagram.add_edge(u_match, v_match, type=B_ETYPE_NAME)


def basis_neighbors(diagram: ZXDiagram, n: int) -> set[int]:
    return {m for m in diagram.neighbors(n) if diagram.is_basis(m)}


def collect(dicts: list[dict[str, Any]], ks: list[str]) -> dict[str, list[Any]]:
    return {f'{k}s': [d[k] for d in dicts] for k in dicts[0] if k in ks}


def f_right_match_from_ndata(diagram: ZXDiagram, n: int) -> FRightMatch:
    if diagram.is_z_basis(n):
        return FRightZMatch(n)
    elif diagram.is_x_basis(n):
        return FRightXMatch(n)
    else:
        raise Exception(f'Unexpected node type {diagram.type(n)}')


def is_inclusion_edge(etype: str) -> bool:
    if isinstance(etype, str):
        return etype == I_ETYPE_NAME
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def has_i_edge(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_inclusion_edge(edata[ETYPE])
    return False


def inclusion_neighbors(match_diagram: ZXMatchDiagram, u_match: Match) -> Iterator[Match]:
    for u_neighbor in match_diagram.neighbors(u_match):
        if has_i_edge(match_diagram, u_match, u_neighbor):
            yield u_neighbor


def is_bridge_edge(etype: str) -> bool:
    if isinstance(etype, str):
        return etype == B_ETYPE_NAME
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def has_b_edge(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_bridge_edge(edata[ETYPE])
    return False


def is_match_neighbor(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    for u_neighbor in inclusion_neighbors(match_diagram, u_match):
        if v_match in inclusion_neighbors(match_diagram, u_neighbor):
            return True
    return False


def connected(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    return is_match_neighbor(match_diagram, u_match, v_match) or has_b_edge(match_diagram, u_match, v_match)


def intersection_zx_match_diagram(diagram: ZXDiagram) -> ZXMatchDiagram:
    match_diagram = ZXMatchDiagram(diagram)
    matches = list(diagram.compute_matches())
    for match in matches:
        match_diagram.add_node(match, type=match.name,
                               **collect([diagram.nodes[node] for node in match], ['phase']))
    for n1 in match_diagram.nodes:
        for n2 in match_diagram.nodes:
            if n1 != n2:
                n1_nodes = set(n1.nodes)
                n2_nodes = set(n2.nodes)
                if len(n1_nodes) == 1 and len(n2_nodes) == 1:
                    continue
                elif len(n1_nodes) == 1 and len(n2_nodes) > 1:
                    if n1.nodes[0] in n2_nodes:
                        match_diagram.add_edge(n1, n2, type=I_ETYPE_NAME)
                elif len(n1_nodes) > 1 and len(n2_nodes) == 1:
                    if n2.nodes[0] in n1_nodes:
                        match_diagram.add_edge(n1, n2, type=I_ETYPE_NAME)
                else:
                    if not n1_nodes.isdisjoint(n2_nodes):
                        match_diagram.add_edge(n1, n2, type=I_ETYPE_NAME)
    return match_diagram
