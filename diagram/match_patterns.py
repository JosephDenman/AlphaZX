import networkx as nx

from diagram.pyzx_nx_conv import Z_NTYPE_NAME, X_NTYPE_NAME
from diagram.zx_diagram import ZXDiagram
from diagram.match import Basis


def f_left_pattern(basis: Basis) -> ZXDiagram:
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([0, 1], type=basis)
    nx_graph.add_edge(0, 1)
    return ZXDiagram(nx_graph)


def f_left_z_pattern() -> ZXDiagram:
    return f_left_pattern(Z_NTYPE_NAME)


def f_left_x_pattern() -> ZXDiagram:
    return f_left_pattern(X_NTYPE_NAME)


def f_right_pattern(basis: Basis) -> ZXDiagram:
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=basis)
    return ZXDiagram(nx_graph)


def f_right_z_pattern() -> ZXDiagram:
    return f_right_pattern(Z_NTYPE_NAME)


def f_right_x_pattern() -> ZXDiagram:
    return f_right_pattern(X_NTYPE_NAME)


def b_right_pattern() -> ZXDiagram:
    """
    x0 -- z1
    """
    diagram = ZXDiagram()
    x, z = diagram.add_x_node(0), diagram.add_z_node(0)
    diagram.add_s_edge(x, z)
    return diagram


def b_left_pattern(bl: int = 0, br: int = 1, tl: int = 2, tr: int = 3) -> ZXDiagram:
    graph = nx.MultiGraph()
    graph.add_nodes_from([bl, br], type=Z_NTYPE_NAME, phase=0)
    graph.add_nodes_from([tl, tr], type=X_NTYPE_NAME, phase=0)
    graph.add_edges_from([(bl, tl), (bl, tr), (br, tl), (br, tr)])
    return ZXDiagram(graph)


def basis_to_ntype_indices(basis: Basis) -> tuple[int, int]:
    return (Z_NTYPE_NAME, X_NTYPE_NAME) if basis == Z_NTYPE_NAME else (X_NTYPE_NAME, Z_NTYPE_NAME)


def y_left_pattern(basis: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=-0.5)
    nx_graph.add_node(1, type=node_types[1], phase=0.0)
    nx_graph.add_node(2, type=node_types[0], phase=0.5)
    nx_graph.add_node(3, type=node_types[0], phase=0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return ZXDiagram(nx_graph)


def y_left_z_pattern() -> ZXDiagram:
    return y_left_pattern(Z_NTYPE_NAME)


def y_left_x_pattern() -> ZXDiagram:
    return y_left_pattern(X_NTYPE_NAME)


def y_right_pattern(rule_mode: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0.5)
    nx_graph.add_node(1, type=node_types[1], phase=-0.5)
    nx_graph.add_node(2, type=node_types[0], phase=-0.5)
    nx_graph.add_node(3, type=node_types[0], phase=-0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return ZXDiagram(nx_graph)


def y_right_z_pattern() -> ZXDiagram:
    return y_right_pattern(Z_NTYPE_NAME)


def y_right_x_pattern() -> ZXDiagram:
    return y_right_pattern(X_NTYPE_NAME)