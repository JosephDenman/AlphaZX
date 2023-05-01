from typing import Dict, Any, Generator

import networkx as nx

from graph.pyzx_nx_dgl_conversion import S_ETYPE_INDEX, is_basis, Z_NTYPE_NAME, X_NTYPE_NAME
from matching.base import RuleMode, rule_mode_to_ntype_indices, all_node_attributes_equal, filter_permutations, \
    TYPE, Matches


def y_left_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_node(1, type=node_types[1], phase=0.0, degree=3)
    nx_graph.add_node(2, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_node(3, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 2, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 3, type=S_ETYPE_INDEX)
    return nx_graph


def y_left_pattern_z() -> nx.MultiGraph:
    return y_left_pattern(Z_NTYPE_NAME)


def y_left_pattern_x() -> nx.MultiGraph:
    return y_left_pattern(X_NTYPE_NAME)


def y_nodes_match(v: Dict[str, Any], w: Dict[str, Any]) -> bool:
    return is_basis(v[TYPE]) and all_node_attributes_equal(v, w)


def y_edges_match(e: Dict[int, Dict[str, Any]], f: Dict[int, Dict[str, Any]]) -> bool:
    return len(e) == 1 and e[0][TYPE] == f[0][TYPE] == S_ETYPE_INDEX


def match_y_left(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches:
    return filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_left_pattern(rule_mode),
                                                                node_match=y_nodes_match,
                                                                edge_match=y_edges_match)
                               .subgraph_isomorphisms_iter())


def match_y_left_z(nx_graph: nx.MultiGraph) -> Matches:
    return match_y_left(nx_graph, Z_NTYPE_NAME)


def match_y_left_x(nx_graph: nx.MultiGraph) -> Matches:
    return match_y_left(nx_graph, X_NTYPE_NAME)


def y_right_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0.5, degree=2)
    nx_graph.add_node(1, type=node_types[1], phase=-0.5, degree=3)
    nx_graph.add_node(2, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_node(3, type=node_types[0], phase=-0.5, degree=2)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 2, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 3, type=S_ETYPE_INDEX)
    return nx_graph


def y_right_pattern_z() -> nx.MultiGraph:
    return y_left_pattern(Z_NTYPE_NAME)


def y_right_pattern_x() -> nx.MultiGraph:
    return y_left_pattern(X_NTYPE_NAME)


def match_y_right(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches:
    return filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, y_right_pattern(rule_mode),
                                                                node_match=y_nodes_match,
                                                                edge_match=y_edges_match)
                               .subgraph_isomorphisms_iter())


def match_y_right_z(nx_graph: nx.MultiGraph) -> Matches:
    return match_y_right(nx_graph, Z_NTYPE_NAME)


def match_y_right_x(nx_graph: nx.MultiGraph) -> Matches:
    return match_y_right(nx_graph, X_NTYPE_NAME)
