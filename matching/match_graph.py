import abc
from collections.abc import Generator, Iterator
from typing import Generic, Tuple, TypeVar, Union, TypeVarTuple, NewType, Any

import networkx as nx
import torch_geometric as pyg

from matplotlib import pyplot as plt
from graph.nx_drawing import draw_nx_zx_diagram
from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conversion import node_types, edge_types
from matching.b_rule import match_b_left, match_b_right
from matching.base import Match, FLeftMatch, FRightMatch
from matching.f_rule import match_f_left_z, match_f_right_z, match_f_left_x, match_f_right_x
from matching.y_rule import match_y_left_z, match_y_right_z, match_y_left_x, match_y_right_x


class Rule(abc.ABC):

    @abc.abstractmethod
    def matcher(self, diagram: nx.MultiGraph) -> Iterator[Match]:
        pass

    @abc.abstractmethod
    def rewrite(self, diagram: nx.MultiGraph, match: Match) -> None:
        pass


def compute_base_matches(diagram: nx.MultiGraph) -> Iterator[FRightMatch]:
    yield from match_f_right_z(diagram)
    yield from match_f_right_x(diagram)


# TODO: This hints that Match class hierarchy should delineate base and compound matches.
def compute_compound_matches(diagram: nx.MultiGraph) -> Iterator[Match]:
    yield from match_f_left_z(diagram)
    yield from match_f_left_x(diagram)
    yield from match_b_left(diagram)
    yield from match_b_right(diagram)
    yield from match_y_left_z(diagram)
    yield from match_y_left_x(diagram)
    yield from match_y_right_z(diagram)
    yield from match_y_right_x(diagram)


def compute_matches(diagram: nx.MultiGraph) -> tuple[Iterator[FRightMatch], Iterator[Match]]:
    return compute_base_matches(diagram), compute_compound_matches(diagram)


I_ETYPE_INDEX = 3
I_ETYPE_NAME = 'inclusion'

MATCH_NAME_TO_NTYPE = {
    'FLeftMatch': 3,
    'FRightMatch': 4,
    'BLeftMatch': 5,
    'BRightMatch': 6,
    'YLeftMatch': 7,
    'YRightMatch': 8
}


def collect(dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {f'{k}s': [d[k] for d in dicts] for k in dicts[0]}


def add_match_node(diagram: nx.MultiGraph, match_diagram: nx.MultiGraph, match: Match) -> None:
    match_diagram.add_node(match, type=MATCH_NAME_TO_NTYPE[match.__class__.__name__],
                           **collect([diagram.nodes[node] for node in match]))


def add_inclusion_edge(match_diagram: nx.MultiGraph, node: int, match: Match) -> None:
    if not match_diagram.has_edge(FRightMatch(node), match):
        match_diagram.add_edge(FRightMatch(node), match, type=I_ETYPE_INDEX)


def compute_match_diagram(diagram: nx.MultiGraph) -> nx.MultiGraph:
    """
    Assumes all base matches are processed first. Assumes all nodes in compound matches are present in base diagram.
    """
    base_matches, compound_matches = compute_matches(diagram)
    match_diagram = nx.MultiGraph()
    for base_match in base_matches:
        add_match_node(diagram, match_diagram, base_match)
    for compound_match in compound_matches:
        add_match_node(diagram, match_diagram, compound_match)
        for node in compound_match:
            add_inclusion_edge(match_diagram, node, compound_match)
    return match_diagram


class ZXDiagram(nx.MultiGraph):

    def __init__(self, nx_graph: nx.MultiGraph, **attr):
        super().__init__(nx_graph, **attr)

    def compute_matches(self) -> Iterator[Match]:
        pass

    def apply_rewrite(self, match: Match) -> None:
        pass


class ZXMatchDiagram(nx.MultiGraph):

    def __init__(self, zx_diagram: ZXDiagram, **attr):
        self.zx_diagram = zx_diagram
        self.zx_match_diagram = compute_match_diagram(zx_diagram)
        self.node_attrs = ['type', 'types', 'phases']
        self.edge_attrs = ['type']
        super().__init__(self.zx_match_diagram, **attr)

    def apply_rewrite(self, match: Match) -> None:
        pass

    def to_pyg_heterograph(self) -> pyg.data.HeteroData:
        assert not self.zx_diagram.is_directed(), "Graph must be undirected"
        hdata = pyg.utils.from_networkx(
            self.zx_diagram,
            group_node_attrs=self.node_attrs,
            group_edge_attrs=self.edge_attrs).to_heterogeneous(node_type=node_types(self.zx_diagram),
                                                               edge_type=edge_types(self.zx_diagram),
                                                               node_type_names=list(MATCH_NAME_TO_NTYPE.keys()),
                                                               edge_type_names=[I_ETYPE_NAME])
        return hdata


d = ZXMatchDiagram(ZXDiagram(nx_clifford_graph(10, 10)))

print(d.nodes(data=True))
print(d.edges(data=True))
