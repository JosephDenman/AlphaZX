import unittest

import networkx as nx

from graph.pyzx_nx_conv import Z_NTYPE_NAME, X_NTYPE_NAME, S_ETYPE_INDEX, H_ETYPE_INDEX
from matching.match_types import Basis, FLeftZMatch, FLeftXMatch, FRightZMatch, FRightXMatch
from matching.f_rule_matcher import f_left_z_matches, f_left_x_matches, f_left_z_pattern, f_left_x_pattern, \
    f_right_z_matches, \
    f_right_z_pattern, f_right_x_matches, f_right_x_pattern, basis_to_ntype_index
from matching.zx_diagram import ZXDiagram


def consecutive_parallel_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_type)
    nx_graph.add_node(45, type=node_type)
    nx_graph.add_node(57, type=node_type)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def hadamard_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def simple_hadamard_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def disconnected_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    return ZXDiagram(nx_graph)


class FLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(f_left_z_matches(f_left_z_pattern())), [FLeftZMatch(0, 1)])

    def test_self_match_x(self):
        self.assertListEqual(list(f_left_x_matches(f_left_x_pattern())), [FLeftXMatch(0, 1)])

    def test_parallel_edge_match_z(self):
        self.assertListEqual(list(f_left_z_matches(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME))),
                             [FLeftZMatch(23, 45), FLeftZMatch(45, 57)])

    def test_parallel_edge_match_x(self):
        self.assertListEqual(list(f_left_x_matches(consecutive_parallel_edge_test_graph(X_NTYPE_NAME))),
                             [FLeftXMatch(23, 45), FLeftXMatch(45, 57)])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(f_left_z_matches(hadamard_edge_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(f_left_x_matches(hadamard_edge_test_graph(X_NTYPE_NAME))), [])

    def test_hadamard_edge_match_z(self):
        self.assertListEqual(list(f_left_z_matches(simple_hadamard_edge_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_match_x(self):
        self.assertListEqual(list(f_left_x_matches(simple_hadamard_edge_test_graph(X_NTYPE_NAME))), [])

    def test_disconnected_no_match_z(self):
        self.assertListEqual(list(f_left_z_matches(disconnected_test_graph(Z_NTYPE_NAME))), [])

    def test_disconnected_no_match_x(self):
        self.assertListEqual(list(f_left_x_matches(disconnected_test_graph(X_NTYPE_NAME))), [])


class FRightMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(f_right_z_matches(f_right_z_pattern())), [FRightZMatch(0)])

    def test_self_match_x(self):
        self.assertListEqual(list(f_right_x_matches(f_right_x_pattern())), [FRightXMatch(0)])

    def test_match_f_left_pattern_z(self):
        self.assertListEqual(list(f_right_z_matches(f_left_z_pattern())), [FRightZMatch(0), FRightZMatch(1)])

    def test_match_f_left_pattern_x(self):
        self.assertListEqual(list(f_right_x_matches(f_left_x_pattern())), [FRightXMatch(0), FRightXMatch(1)])

    def test_different_basis_no_match_z(self):
        self.assertListEqual(list(f_right_z_matches(f_right_x_pattern())), [])

    def test_different_basis_no_match_x(self):
        self.assertListEqual(list(f_right_x_matches(f_right_z_pattern())), [])

    def test_disconnected_match_z(self):
        self.assertListEqual(list(f_right_z_matches(disconnected_test_graph(Z_NTYPE_NAME))),
                             [FRightZMatch(0), FRightZMatch(1)])

    def test_disconnected_match_x(self):
        self.assertListEqual(list(f_right_x_matches(disconnected_test_graph(X_NTYPE_NAME))),
                             [FRightXMatch(0), FRightXMatch(1)])

    def test_consecutive_parallel_edge_match_z(self):
        self.assertListEqual(list(f_right_z_matches(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME))),
                             [FRightZMatch(57), FRightZMatch(45), FRightZMatch(23)])

    def test_consecutive_parallel_edge_match_x(self):
        self.assertListEqual(list(f_right_x_matches(consecutive_parallel_edge_test_graph(X_NTYPE_NAME))),
                             [FRightXMatch(57), FRightXMatch(45), FRightXMatch(23)])
