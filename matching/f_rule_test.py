import unittest

import networkx as nx
from graph.pyzx_nx_dgl_conversion import Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, \
    H_ETYPE_INDEX
from matching.base import RuleMode, rule_mode_to_ntype_index
from matching.f_rule import match_f_left_z, match_f_left_x, f_left_pattern_z, f_left_pattern_x, match_f_right_z, \
    f_right_pattern_z, match_f_right_x, f_right_pattern_x


def consecutive_parallel_edge_test_graph(rule_mode: RuleMode):
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_type)
    nx_graph.add_node(45, type=node_type)
    nx_graph.add_node(57, type=node_type)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    return nx_graph


def hadamard_edge_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return nx_graph


def simple_hadamard_edge_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return nx_graph


def disconnected_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_type = rule_mode_to_ntype_index(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    return nx_graph


class FMatchLeft(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_f_left_z(f_left_pattern_z())), [{0: 0, 1: 1}])

    def test_self_match_x(self):
        self.assertListEqual(list(match_f_left_x(f_left_pattern_x())), [{0: 0, 1: 1}])

    def test_parallel_edge_match_z(self):
        self.assertListEqual(list(match_f_left_z(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME))),
                             [{23: 0, 45: 1}, {45: 0, 57: 1}])

    def test_parallel_edge_match_x(self):
        self.assertListEqual(list(match_f_left_x(consecutive_parallel_edge_test_graph(X_NTYPE_NAME))),
                             [{23: 0, 45: 1}, {45: 0, 57: 1}])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(match_f_left_z(hadamard_edge_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(match_f_left_x(hadamard_edge_test_graph(X_NTYPE_NAME))), [])

    def test_hadamard_edge_match_z(self):
        self.assertListEqual(list(match_f_left_z(simple_hadamard_edge_test_graph(Z_NTYPE_NAME))), [{0: 0, 1: 1}])

    def test_hadamard_edge_match_x(self):
        self.assertListEqual(list(match_f_left_x(simple_hadamard_edge_test_graph(X_NTYPE_NAME))), [{0: 0, 1: 1}])

    def test_disconnected_no_match_z(self):
        self.assertListEqual(list(match_f_left_z(disconnected_test_graph(Z_NTYPE_NAME))), [])

    def test_disconnected_no_match_x(self):
        self.assertListEqual(list(match_f_left_x(disconnected_test_graph(X_NTYPE_NAME))), [])

    def test_draw(self):
        self.assertTrue(True)
        """my_num_qubits = 10
        my_depth = 10
        nx_graph = nx_clifford_graph(my_num_qubits, my_depth, no_hadamard=True, t_gates=False)
        x_matches = match_f_right_x(nx_graph)
        for i, match in enumerate(list(x_matches)):
            plt.figure()
            draw_nx_zx_diagram(nx_graph, match)
        z_matches = match_f_right_z(nx_graph)
        for i, match in enumerate(list(z_matches)):
            plt.figure()
            draw_nx_zx_diagram(nx_graph, match)
        plt.show()"""


class FMatchRight(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_f_right_z(f_right_pattern_z())), [{0: 0}])

    def test_self_match_x(self):
        self.assertListEqual(list(match_f_right_x(f_right_pattern_x())), [{0: 0}])

    def test_match_f_left_pattern_z(self):
        self.assertListEqual(list(match_f_right_z(f_left_pattern_z())), [{0: 0}, {1: 0}])

    def test_match_f_left_pattern_x(self):
        self.assertListEqual(list(match_f_right_x(f_left_pattern_x())), [{0: 0}, {1: 0}])

    def test_different_basis_no_match_z(self):
        self.assertListEqual(list(match_f_right_z(f_right_pattern_x())), [])

    def test_different_basis_no_match_x(self):
        self.assertListEqual(list(match_f_right_x(f_right_pattern_z())), [])

    def test_disconnected_match_z(self):
        self.assertListEqual(list(match_f_right_z(disconnected_test_graph(Z_NTYPE_NAME))), [{0: 0}, {1: 0}])

    def test_disconnected_match_x(self):
        self.assertListEqual(list(match_f_right_x(disconnected_test_graph(X_NTYPE_NAME))), [{0: 0}, {1: 0}])

    def test_consecutive_parallel_edge_match_z(self):
        self.assertListEqual(list(match_f_right_z(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME))),
                             [{23: 0}, {45: 0}, {57: 0}])

    def test_consecutive_parallel_edge_match_x(self):
        self.assertListEqual(list(match_f_right_x(consecutive_parallel_edge_test_graph(X_NTYPE_NAME))),
                             [{23: 0}, {45: 0}, {57: 0}])

    """
    def test_parallel_edge_match_x(self):
        self.assertListEqual(list(match_f_left_x(parallel_edge_match_test_graph(X_NTYPE_NAME))),
                             [{23: 0, 45: 1}, {45: 0, 57: 1}])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(match_f_left_z(hadamard_edge_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(match_f_left_x(hadamard_edge_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_hadamard_edge_match_z(self):
        self.assertListEqual(list(match_f_left_z(hadamard_edge_match_test_graph(Z_NTYPE_NAME))), [{0: 0, 1: 1}])

    def test_hadamard_edge_match_x(self):
        self.assertListEqual(list(match_f_left_x(hadamard_edge_match_test_graph(X_NTYPE_NAME))), [{0: 0, 1: 1}])

    def test_disconnected_no_match_z(self):
        self.assertListEqual(list(match_f_left_z(disconnected_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_disconnected_no_match_x(self):
        self.assertListEqual(list(match_f_left_x(disconnected_no_match_test_graph(X_NTYPE_NAME))), [])"""
