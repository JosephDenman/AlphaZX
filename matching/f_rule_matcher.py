from collections.abc import Iterator
from typing import Any

import networkx as nx

from graph.pyzx_nx_conv import is_basis, Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, is_simple_edge, NTYPE, ETYPE
from matching.match_types import Basis, FRightZMatch, FLeftXMatch, FLeftZMatch, FRightXMatch, FLeftMatch, FRightMatch
from matching.utils import filter_permutations, basis_to_ntype_index


def f_left_pattern(basis: Basis) -> nx.MultiGraph:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([0, 1], type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    return nx_graph


def f_left_z_pattern() -> nx.MultiGraph:
    return f_left_pattern(Z_NTYPE_NAME)


def f_left_x_pattern() -> nx.MultiGraph:
    return f_left_pattern(X_NTYPE_NAME)


def f_nodes_match(v: dict[str, Any], w: dict[str, Any]) -> bool:
    return is_basis(v[NTYPE]) and v[NTYPE] == w[NTYPE]


def f_left_edges_match(e: dict[int, dict[str, Any]], _: dict[int, dict[str, Any]]) -> bool:
    # true if non-Hadamard edge exists
    return any([is_simple_edge(attributes[ETYPE]) for attributes in e.values()])


def f_left_matches(nx_graph: nx.MultiGraph) -> Iterator[FLeftMatch]:
    yield from f_left_z_matches(nx_graph)
    yield from f_left_x_matches(nx_graph)


def f_left_z_matches(nx_graph: nx.MultiGraph) -> Iterator[FLeftZMatch]:
    return (FLeftZMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, f_left_z_pattern(),
                                                                 node_match=f_nodes_match,
                                                                 edge_match=f_left_edges_match)
                                .subgraph_monomorphisms_iter()))


def f_left_x_matches(nx_graph: nx.MultiGraph) -> Iterator[FLeftXMatch]:
    return (FLeftXMatch(match) for match in
            filter_permutations(nx.isomorphism.MultiGraphMatcher(nx_graph, f_left_x_pattern(),
                                                                 node_match=f_nodes_match,
                                                                 edge_match=f_left_edges_match)
                                .subgraph_monomorphisms_iter()))


def f_right_pattern(basis: Basis) -> nx.MultiGraph:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    return nx_graph


def f_right_z_pattern() -> nx.MultiGraph:
    return f_right_pattern(Z_NTYPE_NAME)


def f_right_x_pattern() -> nx.MultiGraph:
    return f_right_pattern(X_NTYPE_NAME)


def f_right_matches(nx_graph: nx.MultiGraph) -> Iterator[FRightMatch]:
    yield from f_right_z_matches(nx_graph)
    yield from f_right_x_matches(nx_graph)


def f_right_z_matches(nx_graph: nx.MultiGraph) -> Iterator[FRightZMatch]:
    return (FRightZMatch(match) for match in
            nx.isomorphism.MultiGraphMatcher(nx_graph, f_right_z_pattern(),
                                             node_match=f_nodes_match)
            .subgraph_isomorphisms_iter())


def f_right_x_matches(nx_graph: nx.MultiGraph) -> Iterator[FRightXMatch]:
    return (FRightXMatch(match) for match in
            nx.isomorphism.MultiGraphMatcher(nx_graph, f_right_x_pattern(),
                                             node_match=f_nodes_match)
            .subgraph_isomorphisms_iter())
