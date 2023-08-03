import unittest

from matching.match import YRightXMatch, YRightZMatch, YLeftXMatch, YLeftZMatch
from matching.y_rule_matcher import y_left_z_pattern, y_left_z_matches, y_left_matches, y_right_matches, \
    y_left_x_pattern, y_right_z_pattern, y_right_x_pattern
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite


class YLeftRewriteTest(unittest.TestCase):

    def test_self_z_rewrite(self):
        diagram = y_left_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_left_rewrite(list(y_left_matches(diagram))[0], diagram)
        self.assertListEqual(list(y_right_matches(diagram)), [YRightXMatch(0, 1, 2, 3)])

    def test_self_x_rewrite(self):
        diagram = y_left_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_left_rewrite(list(y_left_matches(diagram))[0], diagram)
        self.assertListEqual(list(y_right_matches(diagram)), [YRightZMatch(0, 1, 2, 3)])


class YRightRewriteTest(unittest.TestCase):

    def test_self_z_rewrite(self):
        diagram = y_right_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_right_rewrite(list(y_right_matches(diagram))[0], diagram)
        self.assertListEqual(list(y_left_matches(diagram)), [YLeftXMatch(0, 1, 2, 3)])

    def test_self_x_rewrite(self):
        diagram = y_right_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_right_rewrite(list(y_right_matches(diagram))[0], diagram)
        self.assertListEqual(list(y_left_matches(diagram)), [YLeftZMatch(0, 1, 2, 3)])