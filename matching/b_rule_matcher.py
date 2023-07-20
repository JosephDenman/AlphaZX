from collections.abc import Iterator
from typing import Any

import networkx as nx
from graph.pyzx_nx_conv import is_basis, S_ETYPE_INDEX, NTYPE, ETYPE, Z_NTYPE_INDEX, X_NTYPE_INDEX
from matching.match_types import BLeftMatch, BRightMatch
from matching.utils import node_attributes_equal, filter_permutations
from matching.zx_diagram import ZXDiagram


def b_right_pattern() -> ZXDiagram:
    """
    z0 -- x1
    """
    diagram = ZXDiagram()
    diagram.add_x_node(0)
    diagram.add_z_node(0)
    diagram.add_edge(0, 1)
    return diagram


def b_nodes_match(v: dict[str, Any], w: dict[str, Any]) -> bool:
    """
    :param v: Node from the diagram being matched against.
    :param w: Node from the given pattern.
    """
    return is_basis(v[NTYPE]) and is_basis(w[NTYPE]) and node_attributes_equal(v, w)


def b_edges_match(e: dict[int, dict[str, Any]], f: dict[int, dict[str, Any]]) -> bool:
    return len(e) == 1 and list(e.values())[0][ETYPE] == f[0][ETYPE] == S_ETYPE_INDEX


def b_right_matches(nx_graph: nx.MultiGraph) -> Iterator[BRightMatch]:
    return (BRightMatch(match) for match in nx.isomorphism.MultiGraphMatcher(nx_graph, b_right_pattern(),
                                                                             node_match=b_nodes_match,
                                                                             edge_match=b_edges_match)
        .subgraph_isomorphisms_iter())


def b_left_pattern(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2,
                   top_right: int = 3) -> ZXDiagram:
    diagram = ZXDiagram()
    diagram.add_nodes_from([bottom_left, bottom_right], type=Z_NTYPE_INDEX, phase=0, degree=3)
    diagram.add_nodes_from([top_left, top_right], type=X_NTYPE_INDEX, phase=0, degree=3)
    diagram.add_edges_from(
        [(bottom_left, top_left), (bottom_left, top_right), (bottom_right, top_left), (bottom_right, top_right)],
        type=S_ETYPE_INDEX)
    return diagram


def b_left_pattern_loop_bottom(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2,
                               top_right: int = 3) -> nx.MultiGraph:
    nx_graph = b_left_pattern(bottom_left, bottom_right, top_left, top_right)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return nx_graph


def b_left_pattern_loop_top(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2,
                            top_right: int = 3) -> nx.MultiGraph:
    nx_graph = b_left_pattern(bottom_left, bottom_right, top_left, top_right)
    nx_graph.add_edge(2, 3, type=S_ETYPE_INDEX)
    return nx_graph


def b_left_matches(nx_graph: nx.MultiGraph) -> Iterator[BLeftMatch]:
    for pattern in [b_left_pattern(), b_left_pattern_loop_bottom(), b_left_pattern_loop_top()]:
        yield from (BLeftMatch(match) for match in
                    filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, pattern, node_match=b_nodes_match,
                                                                         edge_match=b_edges_match)
                                        .subgraph_isomorphisms_iter()))
