import unittest
from typing import List, Dict, Tuple

import networkx as nx
from matplotlib import pyplot as plt

from graph.nx_drawing import draw_nx_zx_diagram
from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conversion import S_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME, H_ETYPE_INDEX, COLUMN, ROW
from matching.b_rule import match_b_right_z, match_b_right_x, b_right_pattern_z, \
    b_right_pattern_x, match_b_left_z, b_left_pattern_z, b_left_pattern_x, match_b_left_x
from matching.base import RuleMode, rule_mode_to_ntype_indices


class BRuleLeft(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_b_left_z(b_left_pattern_z())), [{0: 0, 1: 1, 2: 2, 3: 3}])

    def test_self_match_x(self):
        self.assertListEqual(list(match_b_left_x(b_left_pattern_x())), [{0: 0, 1: 1, 2: 2, 3: 3}])


def wrong_degree_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(1, type=node_types[0], phase=0, degree=2)
    nx_graph.add_node(2, type=node_types[1], phase=0, degree=2)
    nx_graph.add_edge(1, 2, type=S_ETYPE_INDEX)
    return nx_graph


def parallel_edge_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_types[0], phase=0, degree=3)
    nx_graph.add_node(17, type=node_types[1], phase=0, degree=3)
    nx_graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 17, type=H_ETYPE_INDEX)
    return nx_graph


def nonzero_phase_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_types[0], phase=4, degree=3)
    nx_graph.add_node(17, type=node_types[1], phase=2, degree=3)
    nx_graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    return nx_graph


def hadamard_edge_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_types[0], phase=0, degree=3)
    nx_graph.add_node(17, type=node_types[1], phase=0, degree=3)
    nx_graph.add_edge(23, 17, type=H_ETYPE_INDEX)
    return nx_graph


def spring_layout_data(nx_graph: nx.MultiGraph) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    xs = [x for _, x in nx_graph.nodes(data=COLUMN)]
    max_x = max(xs)
    min_x = min(xs)
    pos = {n: (ndata[COLUMN], ndata[ROW]) for n, ndata in nx_graph.nodes(data=True) if
           ndata[COLUMN] == max_x or ndata[COLUMN] == min_x}
    fixed = list(pos.keys())
    return fixed, pos


def add_layer_data(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        ndata['layer'] = ndata[COLUMN] / 100


class BRuleRight(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_b_right_z(b_right_pattern_z())), [{0: 0, 1: 1}])

    def test_self_match_x(self):
        self.assertListEqual(list(match_b_right_x(b_right_pattern_x())), [{0: 0, 1: 1}])

    def test_line_graph_no_match_z(self):
        self.assertListEqual(list(match_b_right_z(wrong_degree_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_line_graph_no_match_x(self):
        self.assertListEqual(list(match_b_right_x(wrong_degree_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_parallel_edge_no_match_z(self):
        self.assertListEqual(list(match_b_right_z(parallel_edge_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_parallel_edge_no_match_x(self):
        self.assertListEqual(list(match_b_right_x(parallel_edge_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_nonzero_phase_no_match_z(self):
        self.assertListEqual(list(match_b_right_z(nonzero_phase_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_nonzero_phase_no_match_x(self):
        self.assertListEqual(list(match_b_right_x(nonzero_phase_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(match_b_right_z(hadamard_edge_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(match_b_right_x(hadamard_edge_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_draw(self):
        self.assertTrue(True)
        my_num_qubits = 20
        my_depth = 20
        nx_graph = nx_clifford_graph(my_num_qubits, my_depth)
        z_matches = match_b_right_z(nx_graph)
        #fixed_nodes, node_pos = spring_layout_data(nx_graph)
        #add_layer_data(nx_graph)
        #pos = nx.spring_layout(nx_graph, pos=node_pos, fixed=fixed_nodes)
        #pos = nx.rescale_layout_dict(nx.multipartite_layout(nx_graph, subset_key='layer'))
        #draw_nx_zx_diagram(nx_graph, pos=pos)
        #plt.show()
        for i, match in enumerate(list(z_matches)):
            plt.figure()
            draw_nx_zx_diagram(nx_graph, match)
        plt.show()
