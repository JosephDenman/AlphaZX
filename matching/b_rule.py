from typing import Generator, Dict, Any

import networkx as nx

from graph.pyzx_nx_conversion import is_basis, Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, NTYPE, ETYPE
from matching.base import RuleMode, rule_mode_to_ntype_indices, node_attributes_equal, filter_permutations, \
    Matches, BLeftMatch, BRightMatch, dicts_to_tuples


def b_right_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0, degree=3)
    nx_graph.add_node(1, type=node_types[1], phase=0, degree=3)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return nx_graph


def b_right_pattern_z() -> nx.MultiGraph:
    return b_right_pattern(Z_NTYPE_NAME)


def b_right_pattern_x() -> nx.MultiGraph:
    return b_right_pattern(X_NTYPE_NAME)


def b_nodes_match(v: Dict[str, Any], w: Dict[str, Any]) -> bool:
    return is_basis(v[NTYPE]) and node_attributes_equal(v, w)


def b_edges_match(e: Dict[int, Dict[str, Any]], f: Dict[int, Dict[str, Any]]) -> bool:
    return len(e) == 1 and list(e.values())[0][ETYPE] == f[0][ETYPE] == S_ETYPE_INDEX


def match_b_right(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches[BRightMatch]:
    return dicts_to_tuples(nx.isomorphism.MultiGraphMatcher(nx_graph, b_right_pattern(rule_mode),
                                                            node_match=b_nodes_match,
                                                            edge_match=b_edges_match).subgraph_isomorphisms_iter())


def match_b_right_z(nx_graph: nx.MultiGraph) -> Matches[BRightMatch]:
    return match_b_right(nx_graph, Z_NTYPE_NAME)


def match_b_right_x(nx_graph: nx.MultiGraph) -> Matches[BRightMatch]:
    return match_b_right(nx_graph, X_NTYPE_NAME)


def b_left_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0, degree=3)
    nx_graph.add_node(1, type=node_types[0], phase=0, degree=3)
    nx_graph.add_node(2, type=node_types[1], phase=0, degree=3)
    nx_graph.add_node(3, type=node_types[1], phase=0, degree=3)
    nx_graph.add_edge(0, 2, type=S_ETYPE_INDEX)
    nx_graph.add_edge(0, 3, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 2, type=S_ETYPE_INDEX)
    nx_graph.add_edge(1, 3, type=S_ETYPE_INDEX)
    return nx_graph


def b_left_pattern_z() -> nx.MultiGraph:
    return b_left_pattern(Z_NTYPE_NAME)


def b_left_pattern_x() -> nx.MultiGraph:
    return b_left_pattern(X_NTYPE_NAME)


def match_b_left(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches[BLeftMatch]:
    return dicts_to_tuples(filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, b_left_pattern(rule_mode),
                                                                                node_match=b_nodes_match,
                                                                                edge_match=b_edges_match).isomorphisms_iter()))


def match_b_left_z(nx_graph: nx.MultiGraph) -> Matches[BLeftMatch]:
    return match_b_left(nx_graph, Z_NTYPE_NAME)


def match_b_left_x(nx_graph: nx.MultiGraph) -> Matches[BLeftMatch]:
    return match_b_left(nx_graph, X_NTYPE_NAME)
