import unittest

import networkx as nx
from matplotlib import pyplot as plt

from graph.nx_drawing import draw_nx_zx_diagram
from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conversion import H_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME
from matching.base import RuleMode
from matching.y_rule import match_y_left_z, y_left_pattern_z, y_left_pattern_x, match_y_left_x, match_y_left, \
    y_left_pattern


def parallel_edge_no_match_test_graph(rule_mode: RuleMode, first: bool = False, second: bool = False,
                                      third=False) -> nx.MultiGraph:
    nx_graph = y_left_pattern(rule_mode)
    if first:
        nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    if second:
        nx_graph.add_edge(1, 2, type=H_ETYPE_INDEX)
    if third:
        nx_graph.add_edge(1, 3, type=H_ETYPE_INDEX)
    return nx_graph


def disconnected_no_match_test_graph(rule_mode: RuleMode, first: bool = False, second: bool = False,
                                     third=False) -> nx.MultiGraph:
    nx_graph = y_left_pattern(rule_mode)
    if first:
        nx_graph.remove_edge(0, 1)
    if second:
        nx_graph.remove_edge(1, 2)
    if third:
        nx_graph.remove_edge(1, 3)
    return nx_graph


class MyTestCase(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_y_left_z(y_left_pattern_z())), [{0: 0, 1: 1, 2: 2, 3: 3}])

    def test_self_match_x(self):
        self.assertListEqual(list(match_y_left_x(y_left_pattern_x())), [{0: 0, 1: 1, 2: 2, 3: 3}])

    def test_parallel_edge_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_z(parallel_edge_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [{0: 0, 1: 1, 2: 2, 3: 3}])
                    else:
                        self.assertListEqual(matches, [])

    def test_parallel_edge_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_x(parallel_edge_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [{0: 0, 1: 1, 2: 2, 3: 3}])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_z(disconnected_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [{0: 0, 1: 1, 2: 2, 3: 3}])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_x(disconnected_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [{0: 0, 1: 1, 2: 2, 3: 3}])
                    else:
                        self.assertListEqual(matches, [])

    """def test_draw(self):
        self.assertTrue(True)
        my_num_qubits = 20000
        my_depth = 20000
        nx_graph = nx_clifford_graph(my_num_qubits, my_depth, no_hadamard=True, t_gates=True)
        # plt.figure()
        # draw_nx_zx_diagram(nx_graph)
        # plt.show()
        matches = match_y_left_z(nx_graph)
        for i, match in enumerate(list(matches)):
            plt.figure()
            draw_nx_zx_diagram(nx_graph, match)
        plt.show()"""
