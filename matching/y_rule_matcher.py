from collections.abc import Iterator

import networkx as nx

from diagram.pyzx_nx_conv import S_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME, Z_NTYPE_INDEX, X_NTYPE_INDEX
from diagram.zx_diagram import ZXDiagram
from matching.match import Basis, YRightZMatch, YRightXMatch, YLeftZMatch, YLeftXMatch, YRightMatch, YLeftMatch


def basis_to_ntype_indices(basis: Basis) -> tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if basis == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def y_left_pattern(basis: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=-0.5)
    nx_graph.add_node(1, type=node_types[1], phase=0.0)
    nx_graph.add_node(2, type=node_types[0], phase=0.5)
    nx_graph.add_node(3, type=node_types[0], phase=0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)], type=S_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def y_left_z_pattern() -> ZXDiagram:
    return y_left_pattern(Z_NTYPE_NAME)


def y_left_x_pattern() -> ZXDiagram:
    return y_left_pattern(X_NTYPE_NAME)


def y_left_matches(diagram: ZXDiagram) -> Iterator[YLeftMatch]:
    yield from y_left_z_matches(diagram)
    yield from y_left_x_matches(diagram)


def y_left_z_matches(diagram: ZXDiagram) -> Iterator[YLeftZMatch]:
    for n in diagram.x_nodes():
        if diagram.degree(n) == 3 and diagram.phase(n) == 0:
            if all([diagram.degree(m) == 2 and diagram.is_z_basis(m) for m in diagram.neighbors(n)]) and sum(
                    [diagram.phase(m) for m in diagram.neighbors(n)]) == 0.5:
                z0, z2, z3 = sorted(diagram.neighbors(n), key=lambda m: diagram.phase(m))
                yield YLeftZMatch(z0, n, z2, z3)


def y_left_x_matches(diagram: ZXDiagram) -> Iterator[YLeftXMatch]:
    for n in diagram.z_nodes():
        if diagram.degree(n) == 3 and diagram.phase(n) == 0:
            if all([diagram.degree(m) == 2 and diagram.is_x_basis(m) for m in diagram.neighbors(n)]) and sum(
                    [diagram.phase(m) for m in diagram.neighbors(n)]) == 0.5:
                x0, x2, x3 = sorted(diagram.neighbors(n), key=lambda m: diagram.phase(m))
                yield YLeftXMatch(x0, n, x2, x3)


def y_right_pattern(rule_mode: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0.5)
    nx_graph.add_node(1, type=node_types[1], phase=-0.5)
    nx_graph.add_node(2, type=node_types[0], phase=-0.5)
    nx_graph.add_node(3, type=node_types[0], phase=-0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)], type=S_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def y_right_z_pattern() -> ZXDiagram:
    return y_right_pattern(Z_NTYPE_NAME)


def y_right_x_pattern() -> ZXDiagram:
    return y_right_pattern(X_NTYPE_NAME)


def y_right_matches(diagram: ZXDiagram) -> Iterator[YRightMatch]:
    yield from y_right_z_matches(diagram)
    yield from y_right_x_matches(diagram)


def y_right_z_matches(diagram: ZXDiagram) -> Iterator[YRightZMatch]:
    for n in diagram.x_nodes():
        if diagram.degree(n) == 3 and diagram.phase(n) == -0.5:
            if all([diagram.degree(m) == 2 and diagram.is_z_basis(m) for m in diagram.neighbors(n)]) and sum(
                    [diagram.phase(m) for m in diagram.neighbors(n)]) == -0.5:
                x0, x2, x3 = sorted(diagram.neighbors(n), key=lambda m: diagram.phase(m), reverse=True)
                yield YRightZMatch(x0, n, x2, x3)


def y_right_x_matches(diagram: ZXDiagram) -> Iterator[YRightXMatch]:
    for n in diagram.z_nodes():
        if diagram.degree(n) == 3 and diagram.phase(n) == -0.5:
            if all([diagram.degree(m) == 2 and diagram.is_x_basis(m) for m in diagram.neighbors(n)]) and sum(
                    [diagram.phase(m) for m in diagram.neighbors(n)]) == -0.5:
                x0, x2, x3 = sorted(diagram.neighbors(n), key=lambda m: diagram.phase(m), reverse=True)
                yield YRightXMatch(x0, n, x2, x3)
