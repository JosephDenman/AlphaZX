import unittest

from graph.pyzx_nx_conv import Z_NTYPE_NAME, X_NTYPE_NAME
from matching.match import Basis, YLeftZMatch, YLeftXMatch, YRightZMatch, YRightXMatch
from matching.y_rule_matcher import y_left_z_matches, y_left_z_pattern, y_left_x_pattern, y_left_x_matches, \
    y_left_pattern, y_right_z_pattern, y_right_z_matches, y_right_x_pattern, y_right_x_matches
from matching.zx_diagram import ZXDiagram


def parallel_edge_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                      third=False) -> ZXDiagram:
    diagram = y_left_pattern(basis)
    b4, b5, b6 = diagram.add_b_nodes(3)
    diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
    if first:
        diagram.add_s_edge(0, 1)
    if second:
        diagram.add_s_edge(1, 2)
    if third:
        diagram.add_s_edge(1, 3)
    return diagram


def disconnected_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                     third=False) -> ZXDiagram:
    diagram = y_left_pattern(basis)
    b4, b5, b6 = diagram.add_b_nodes(3)
    diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
    if first:
        diagram.remove_edge(0, 1)
    if second:
        diagram.remove_edge(1, 2)
    if third:
        diagram.remove_edge(1, 3)
    return diagram


class YLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        diagram = y_left_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(y_left_z_matches(diagram)), [YLeftZMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        diagram = y_left_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(y_left_x_matches(diagram)), [YLeftXMatch(0, 1, 2, 3)])

    def test_parallel_edge_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(y_left_z_matches(parallel_edge_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftZMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_parallel_edge_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(y_left_x_matches(parallel_edge_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftXMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(y_left_z_matches(disconnected_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftZMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(y_left_x_matches(disconnected_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftXMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])


class YRightMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        diagram = y_right_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(y_right_z_matches(diagram)), [YRightZMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        diagram = y_right_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(y_right_x_matches(diagram)), [YRightXMatch(0, 1, 2, 3)])