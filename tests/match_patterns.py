import networkx as nx

from alphazx.diagram.match import Basis, FRightXMatch, FRightZMatch
from alphazx.diagram.zx_diagram import ZXDiagram


def f_left_pattern(d: int, basis: Basis) -> ZXDiagram:
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([0, 1], type=basis, phase=0)
    nx_graph.add_edge(0, 1)
    return ZXDiagram(d, nx_graph)


def f_left_z_pattern(d: int) -> ZXDiagram:
    return f_left_pattern(d, FRightZMatch.abbrev)


def f_left_x_pattern(d: int) -> ZXDiagram:
    return f_left_pattern(d, FRightXMatch.abbrev)


def f_right_pattern(d: int, basis: Basis) -> ZXDiagram:
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=basis, phase=0)
    return ZXDiagram(d, nx_graph)


def f_right_z_pattern(d: int) -> ZXDiagram:
    return f_right_pattern(d, FRightZMatch.abbrev)


def f_right_x_pattern(d: int) -> ZXDiagram:
    return f_right_pattern(d, FRightXMatch.abbrev)


def b_right_pattern(d: int) -> ZXDiagram:
    """
    x0 -- z1
    """
    diagram = ZXDiagram(d)
    z, x = diagram.add_z_node(0), diagram.add_x_node(0),
    diagram.add_s_edge(z, x)
    return diagram


def b_left_pattern(d: int, bl: int = 0, br: int = 1, tl: int = 2, tr: int = 3) -> ZXDiagram:
    # bl and br are Z nodes
    # tl and tr are X nodes
    graph = nx.MultiGraph()
    graph.add_nodes_from([bl, br], type=FRightZMatch.abbrev, phase=0)
    graph.add_nodes_from([tl, tr], type=FRightXMatch.abbrev, phase=0)
    graph.add_edges_from([(bl, tl), (bl, tr), (br, tl), (br, tr)])
    return ZXDiagram(d, graph)


def basis_to_ntype_indices(basis: Basis) -> tuple[int, int]:
    return (FRightZMatch.abbrev, FRightXMatch.abbrev) if basis == FRightZMatch.abbrev else (
    FRightXMatch.abbrev, FRightZMatch.abbrev)


def y_left_pattern(d: int, basis: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=-0.5)
    nx_graph.add_node(1, type=node_types[1], phase=0.0)
    nx_graph.add_node(2, type=node_types[0], phase=0.5)
    nx_graph.add_node(3, type=node_types[0], phase=0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return ZXDiagram(d, nx_graph)


def y_left_z_pattern(d: int) -> ZXDiagram:
    return y_left_pattern(d, FRightZMatch.abbrev)


def y_left_x_pattern(d: int) -> ZXDiagram:
    return y_left_pattern(d, FRightXMatch.abbrev)


def y_right_pattern(d: int, rule_mode: Basis) -> ZXDiagram:
    node_types = basis_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_types[0], phase=0.5)
    nx_graph.add_node(1, type=node_types[1], phase=-0.5)
    nx_graph.add_node(2, type=node_types[0], phase=-0.5)
    nx_graph.add_node(3, type=node_types[0], phase=-0.5)
    nx_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return ZXDiagram(d, nx_graph)


def y_right_z_pattern(d: int) -> ZXDiagram:
    return y_right_pattern(d, FRightZMatch.abbrev)


def y_right_x_pattern(d: int) -> ZXDiagram:
    return y_right_pattern(d, FRightXMatch.abbrev)
