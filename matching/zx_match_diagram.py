from typing import Any, Iterator

import networkx as nx
import torch

from graph.pyzx_nx_conv import ETYPE
from matching.match_types import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch
from matching.utils import compute_matches
from matching.zx_diagram import ZXDiagram


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
        matches = list(compute_matches(diagram))
        for match in matches:
            add_match(self, diagram, match)
        assert self.number_of_nodes() == len(matches), "Number of nodes in match diagram != number of matches"
        add_composition_edges(self, diagram)


def add_match(match_diagram: ZXMatchDiagram, diagram: ZXDiagram, match: Match) -> None:
    if not match_diagram.has_node(match):
        match_diagram.add_node(match, type=match.index,
                               **collect([diagram.nodes[node] for node in match]))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not match_diagram.has_node(sub_match):
                add_match(match_diagram, diagram, sub_match)
            if not match_diagram.has_edge(sub_match, match):
                match_diagram.add_edge(sub_match, match, type=I_ETYPE_INDEX)
    return


def add_composition_edges(match_diagram: ZXMatchDiagram, diagram: ZXDiagram) -> None:
    for u in diagram.basis_nodes():
        for v in basis_neighbors(diagram, u):
            u_match = f_right_match_from_ndata(diagram, u)
            v_match = f_right_match_from_ndata(diagram, v)
            if not connected(match_diagram, u_match, v_match):
                match_diagram.add_edge(u_match, v_match, type=B_ETYPE_INDEX)


def basis_neighbors(diagram: ZXDiagram, n: int) -> set[int]:
    return {m for m in diagram.neighbors(n) if diagram.is_basis(m)}


def collect(dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {f'{k}s': torch.tensor([d[k] for d in dicts]) for k in dicts[0]}


def f_right_match_from_ndata(diagram: ZXDiagram, n: int) -> FRightMatch:
    if diagram.is_z_basis(n):
        return FRightZMatch(n)
    elif diagram.is_x_basis(n):
        return FRightXMatch(n)
    else:
        raise Exception(f'Unexpected node type {diagram.type(n)}')


def is_inclusion_edge(etype: str | int) -> bool:
    if isinstance(etype, str):
        return etype == I_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == I_ETYPE_INDEX
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


def is_bridge_edge(etype: str | int) -> bool:
    if isinstance(etype, str):
        return etype == B_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == B_ETYPE_INDEX
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

