from typing import Dict, Any

import networkx as nx

from graph.pyzx_nx_conversion import is_basis, S_ETYPE_INDEX, NTYPE, ETYPE, Z_NTYPE_INDEX, X_NTYPE_INDEX
from matching.base import node_attributes_equal, filter_permutations, Matches, BLeftMatch, BRightMatch, dicts_to_tuples


def b_right_pattern() -> nx.MultiGraph:
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=Z_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_node(1, type=X_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return nx_graph


def b_nodes_match(v: Dict[str, Any], w: Dict[str, Any]) -> bool:
    return is_basis(v[NTYPE]) and node_attributes_equal(v, w)


def b_edges_match(e: Dict[int, Dict[str, Any]], f: Dict[int, Dict[str, Any]]) -> bool:
    return len(e) == 1 and list(e.values())[0][ETYPE] == f[0][ETYPE] == S_ETYPE_INDEX


def match_b_right(nx_graph: nx.MultiGraph) -> Matches[BRightMatch]:
    return (BRightMatch(*match) for match in
            dicts_to_tuples(nx.isomorphism.MultiGraphMatcher(nx_graph, b_right_pattern(),
                                                             node_match=b_nodes_match,
                                                             edge_match=b_edges_match).subgraph_isomorphisms_iter()))


def b_left_pattern(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2,
                   top_right: int = 3) -> nx.MultiGraph:
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([bottom_left, bottom_right], type=Z_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_nodes_from([top_left, top_right], type=X_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_edges_from(
        [(bottom_left, top_left), (bottom_left, top_right), (bottom_right, top_left), (bottom_right, top_right)],
        type=S_ETYPE_INDEX)
    return nx_graph


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


def match_b_left(nx_graph: nx.MultiGraph) -> Matches[BLeftMatch]:
    yield from (BLeftMatch(*match) for match in
                dicts_to_tuples(
                    filter_permutations(
                        nx.isomorphism.MultiGraphMatcher(nx_graph, b_left_pattern(), node_match=b_nodes_match,
                                                         edge_match=b_edges_match).subgraph_isomorphisms_iter())))
    yield from (BLeftMatch(*match) for match in
                dicts_to_tuples(
                    filter_permutations(
                        nx.isomorphism.MultiGraphMatcher(nx_graph, b_left_pattern_loop_bottom(),
                                                         node_match=b_nodes_match,
                                                         edge_match=b_edges_match).subgraph_isomorphisms_iter())))
    yield from (BLeftMatch(*match) for match in
                dicts_to_tuples(
                    filter_permutations(
                        nx.isomorphism.MultiGraphMatcher(nx_graph, b_left_pattern_loop_top(),
                                                         node_match=b_nodes_match,
                                                         edge_match=b_edges_match).subgraph_isomorphisms_iter())))
