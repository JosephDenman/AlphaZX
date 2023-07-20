import unittest

import networkx as nx

from graph.pyzx_nx_conv import H_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME
from matching.match_types import Basis, YLeftZMatch, YLeftXMatch
from matching.y_rule_matcher import y_left_z_matches, y_left_z_pattern, y_left_x_pattern, y_left_x_matches, y_left_pattern


def parallel_edge_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                      third=False) -> nx.MultiGraph:
    nx_graph = y_left_pattern(basis)
    if first:
        nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    if second:
        nx_graph.add_edge(1, 2, type=H_ETYPE_INDEX)
    if third:
        nx_graph.add_edge(1, 3, type=H_ETYPE_INDEX)
    return nx_graph


def disconnected_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                     third=False) -> nx.MultiGraph:
    nx_graph = y_left_pattern(basis)
    if first:
        nx_graph.remove_edge(0, 1)
    if second:
        nx_graph.remove_edge(1, 2)
    if third:
        nx_graph.remove_edge(1, 3)
    return nx_graph


class YLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(y_left_z_matches(y_left_z_pattern())), [YLeftZMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        self.assertListEqual(list(y_left_x_matches(y_left_x_pattern())), [YLeftXMatch(0, 1, 2, 3)])

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

    def test_symmetric(self):
        # TODO: Test symmetric matches are filtered out.
        self.assertTrue(True)
