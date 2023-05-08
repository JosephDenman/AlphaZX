import networkx as nx
from matplotlib import pyplot as plt

from graph.nx_drawing import draw_nx_zx_diagram
from graph.pyzx_graph_generator import nx_clifford_graph
from matching.b_rule import match_b_left, match_b_right
from matching.base import Matches, Match
from matching.f_rule import match_f_left_z, match_f_right_z, match_f_left_x, match_f_right_x
from matching.y_rule import match_y_left_z, match_y_right_z, match_y_left_x, match_y_right_x


def compute_matches(nx_graph: nx.MultiGraph) -> Matches[Match]:
    yield from match_f_left_z(nx_graph)
    yield from match_f_left_x(nx_graph)
    yield from match_f_right_z(nx_graph)
    yield from match_f_right_x(nx_graph)
    yield from match_b_left(nx_graph)
    yield from match_b_right(nx_graph)
    yield from match_y_left_z(nx_graph)
    yield from match_y_left_x(nx_graph)
    yield from match_y_right_z(nx_graph)
    yield from match_y_right_x(nx_graph)


def gen_diagram():
    my_num_qubits = 10
    my_depth = 10
    return nx_clifford_graph(my_num_qubits, my_depth)


def draw_matches(nx_graph: nx.MultiGraph) -> None:
    matches = compute_matches(nx_graph)
    for match in matches:
        plt.figure()
        draw_nx_zx_diagram(nx_graph, match)
    plt.show()


INC_ETYPE_INDEX = 3
INC_ETYPE_NAME = 'incl'

MATCH_NAME_TO_NTYPE = {
    'FLeftMatch': 3,
    'FRightMatch': 4,
    'BLeftMatch': 5,
    'BRightMatch': 6,
    'YLeftMatch': 7,
    'YRightMatch': 8
}


def compute_match_graph(nx_graph: nx.MultiGraph) -> None:
    matches = compute_matches(nx_graph)
    for match in list(matches):
        nx_graph.add_node(match, type=MATCH_NAME_TO_NTYPE[match.__class__.__name__])
        for node in match:
            assert nx_graph.has_node(node), f'Graph does not contain node {node} in match {match}'
            nx_graph.add_edge(node, match, type=INC_ETYPE_INDEX)


diagram = gen_diagram()
compute_match_graph(diagram)
print(diagram.nodes(data=True))
