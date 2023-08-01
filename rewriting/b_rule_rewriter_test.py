import unittest

from graph.pyzx_nx_conv import B_NTYPE_INDEX, X_NTYPE_INDEX, Z_NTYPE_INDEX, is_z_basis, is_x_basis, is_boundary
from matching.b_rule_matcher import b_right_pattern, b_right_matches, b_left_matches, b_left_pattern
from matching.match_types import BLeftMatch, BRightMatch
from matching.zx_diagram import ZXDiagram
from rewriting.b_rule_rewriter import b_right_rewrite, b_left_rewrite

from hypothesis import given
from hypothesis import strategies as st

from rewriting.test_utils import st_b_right_nodes


def add_node(data: tuple[B_NTYPE_INDEX | X_NTYPE_INDEX | Z_NTYPE_INDEX, float], diagram: ZXDiagram) -> int:
    ntype, phase = data
    if is_z_basis(ntype):
        return diagram.add_z_node(phase)
    elif is_x_basis(ntype):
        return diagram.add_x_node(phase)
    elif is_boundary(ntype):
        return diagram.add_b_node()
    else:
        raise Exception(f'Unexpected node type {ntype}')


class BRightRewriteTest(unittest.TestCase):

    @given(st_b_right_nodes())
    def test_simple_b_right_rewrite(self, data_list):
        bl_data, br_data, tl_data, tr_data = data_list
        diagram = ZXDiagram(b_right_pattern())
        diagram.add_s_edge(add_node(bl_data, diagram), 0)
        diagram.add_s_edge(add_node(br_data, diagram), 0)
        diagram.add_s_edge(1, add_node(tr_data, diagram))
        diagram.add_s_edge(1, add_node(tl_data, diagram))
        b_right_rewrite(list(b_right_matches(diagram))[0], diagram)
        self.assertEqual(list(b_left_matches(diagram))[0], BLeftMatch(7, 9, 6, 8))


class BLeftRewriteTest(unittest.TestCase):

    def test_simple_b_left_rewrite(self):
        diagram = b_left_pattern()
        b4, b5, b6, b7 = diagram.add_b_node(), diagram.add_b_node(), diagram.add_b_node(), diagram.add_b_node()
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, b6), (3, b7)])
        b_left_rewrite(list(b_left_matches(diagram))[0], diagram)
        self.assertEqual(list(b_right_matches(diagram))[0], BRightMatch(8, 9))

