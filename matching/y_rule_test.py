import unittest

import networkx as nx

from graph.pyzx_nx_conversion import H_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME
from matching.base import RuleMode, YLeftMatch
from matching.y_rule import match_y_left_z, y_left_pattern_z, y_left_pattern_x, match_y_left_x, y_left_pattern


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


class YLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(match_y_left_z(y_left_pattern_z())), [YLeftMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        self.assertListEqual(list(match_y_left_x(y_left_pattern_x())), [YLeftMatch(0, 1, 2, 3)])

    def test_parallel_edge_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_z(parallel_edge_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_parallel_edge_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_x(parallel_edge_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_z(disconnected_no_match_test_graph(Z_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(match_y_left_x(disconnected_no_match_test_graph(X_NTYPE_NAME, a, b, c)))
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])
