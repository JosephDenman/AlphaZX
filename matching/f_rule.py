from typing import Dict, Any

import networkx as nx

from graph.pyzx_nx_conversion import is_basis, Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, is_simple_edge, NTYPE, ETYPE
from matching.base import RuleMode, rule_mode_to_ntype_index, filter_permutations, Matches, FRightMatch, FLeftMatch, \
    dicts_to_tuples


def f_left_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([0, 1], type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return nx_graph


def f_left_pattern_z() -> nx.MultiGraph:
    return f_left_pattern(Z_NTYPE_NAME)


def f_left_pattern_x() -> nx.MultiGraph:
    return f_left_pattern(X_NTYPE_NAME)


def f_nodes_match(v: Dict[str, Any], w: Dict[str, Any]) -> bool:
    return is_basis(v[NTYPE]) and v[NTYPE] == w[NTYPE]


def non_hadamard_edge_exists(e: Dict[int, Dict[str, Any]]) -> bool:
    return any([is_simple_edge(attributes[ETYPE]) for attributes in e.values()])


def f_left_edges_match(e: Dict[int, Dict[str, Any]], _: Dict[int, Dict[str, Any]]) -> bool:
    return non_hadamard_edge_exists(e)


def match_f_left(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches[FLeftMatch]:
    return (FLeftMatch(*match) for match in
            dicts_to_tuples(filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, f_left_pattern(rule_mode),
                                                                                 node_match=f_nodes_match,
                                                                                 edge_match=f_left_edges_match)
                                                .subgraph_monomorphisms_iter())))


def match_f_left_z(nx_graph: nx.MultiGraph) -> Matches[FLeftMatch]:
    return match_f_left(nx_graph, Z_NTYPE_NAME)


def match_f_left_x(nx_graph: nx.MultiGraph) -> Matches[FLeftMatch]:
    return match_f_left(nx_graph, X_NTYPE_NAME)


def f_right_pattern(rule_mode: RuleMode) -> nx.MultiGraph:
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    return nx_graph


def f_right_pattern_z() -> nx.MultiGraph:
    return f_right_pattern(Z_NTYPE_NAME)


def f_right_pattern_x() -> nx.MultiGraph:
    return f_right_pattern(X_NTYPE_NAME)


def match_f_right(nx_graph: nx.MultiGraph, rule_mode: RuleMode) -> Matches[FRightMatch]:
    return (FRightMatch(*match) for match in
            dicts_to_tuples(nx.isomorphism.MultiGraphMatcher(nx_graph, f_right_pattern(rule_mode),
                                                             node_match=f_nodes_match).subgraph_isomorphisms_iter()))


def match_f_right_z(nx_graph: nx.MultiGraph) -> Matches[FRightMatch]:
    return match_f_right(nx_graph, Z_NTYPE_NAME)


def match_f_right_x(nx_graph: nx.MultiGraph) -> Matches[FRightMatch]:
    return match_f_right(nx_graph, X_NTYPE_NAME)
