from collections.abc import Iterator
from typing import Dict, Any

import networkx as nx

from graph.pyzx_nx_conversion import S_ETYPE_INDEX, is_basis, Z_NTYPE_NAME, X_NTYPE_NAME, NTYPE, ETYPE
from matching.match import RuleMode, YRightZMatch, YRightXMatch, YLeftZMatch, YLeftXMatch, YRightMatch
from matching.utils import rule_mode_to_ntype_indices, filter_permutations, node_attributes_equal


def y_left_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_node(1, type=node_types[1], phase=0.0, degree=3)
    nx_graph.add_node(2, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_node(3, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)], type=S_ETYPE_INDEX)
    return nx_graph


def y_left_z_pattern() -> nx.MultiGraph:
    return y_left_pattern(Z_NTYPE_NAME)


def y_left_x_pattern() -> nx.MultiGraph:
    return y_left_pattern(X_NTYPE_NAME)


def y_nodes_match(v: dict[str, Any], w: dict[str, Any]) -> bool:
    return is_basis(v[NTYPE]) and node_attributes_equal(v, w)


def y_edges_match(e: dict[int, dict[str, Any]], f: dict[int, Dict[str, Any]]) -> bool:
    return len(e) == 1 and e[0][ETYPE] == f[0][ETYPE] == S_ETYPE_INDEX


def y_left_matches(nx_graph: nx.MultiGraph) -> Iterator[YLeftZMatch | YLeftXMatch]:
    yield from y_left_z_matches(nx_graph)
    yield from y_left_x_matches(nx_graph)


def y_left_z_matches(nx_graph: nx.MultiGraph) -> Iterator[YLeftZMatch]:
    return (YLeftZMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_left_z_pattern(),
                                                                 node_match=y_nodes_match,
                                                                 edge_match=y_edges_match)
                                .subgraph_isomorphisms_iter()))


def y_left_x_matches(nx_graph: nx.MultiGraph) -> Iterator[YLeftXMatch]:
    return (YLeftXMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_left_x_pattern(),
                                                                 node_match=y_nodes_match,
                                                                 edge_match=y_edges_match)
                                .subgraph_isomorphisms_iter()))


def y_right_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_node(1, type=node_types[1], phase=-0.5, degree=3)
    nx_graph.add_node(2, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_node(3, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)], type=S_ETYPE_INDEX)
    return nx_graph


def y_right_z_pattern() -> nx.MultiGraph:
    return y_left_pattern(Z_NTYPE_NAME)


def y_right_x_pattern() -> nx.MultiGraph:
    return y_left_pattern(X_NTYPE_NAME)


def y_right_matches(nx_graph: nx.MultiGraph) -> Iterator[YRightMatch]:
    yield from y_right_z_matches(nx_graph)
    yield from y_right_x_matches(nx_graph)


def y_right_z_matches(nx_graph: nx.MultiGraph) -> Iterator[YRightZMatch]:
    return (YRightZMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_right_z_pattern(),
                                                                 node_match=y_nodes_match,
                                                                 edge_match=y_edges_match)
                                .subgraph_isomorphisms_iter()))


def y_right_x_matches(nx_graph: nx.MultiGraph) -> Iterator[YRightXMatch]:
    return (YRightXMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_right_x_pattern(),
                                                                 node_match=y_nodes_match,
                                                                 edge_match=y_edges_match)
                                .subgraph_isomorphisms_iter()))
