import unittest

from hypothesis import strategies as st, given
from hypothesis.strategies import composite

from graph.pyzx_nx_conv import Z_NTYPE_INDEX, X_NTYPE_INDEX, B_NTYPE_INDEX
from matching.f_rule_matcher import f_right_matches
from matching.zx_diagram import ZXDiagram
from rewriting.f_rule_rewriter import f_right_rewrite
from rewriting.test_utils import st_node, st_phase


@composite
def f_right_st(draw):
    z_basis = draw(st.booleans())
    center = (1, draw(st_phase(64))) if z_basis else (2, draw(st_phase(64)))
    neighbor_number = draw(st.integers(min_value=1, max_value=20))
    neighbors = draw(st.lists(st_node(64), min_size=neighbor_number, max_size=neighbor_number))
    edge_count = draw(st.integers(min_value=1, max_value=len(neighbors)))
    edges = draw(
        st.sets(st.tuples(st.sampled_from([0]), st.sampled_from(list(range(1, neighbor_number + 1)))),
                min_size=edge_count,
                max_size=edge_count))
    transfer_edge_count = draw(st.integers(min_value=0, max_value=len(edges)))
    transfer_edges = draw(
        st.sets(st.sampled_from(list(edges)), min_size=transfer_edge_count, max_size=transfer_edge_count))
    new_edges = draw(st.integers(min_value=1, max_value=20))
    phase = draw(st_phase(64))
    return center, neighbors, edges, transfer_edges, new_edges, phase


def add_node(ntype: int, phase: float, diagram: ZXDiagram) -> None:
    if ntype == Z_NTYPE_INDEX:
        diagram.add_z_node(phase=phase)
    elif ntype == X_NTYPE_INDEX:
        diagram.add_x_node(phase=phase)
    elif ntype == B_NTYPE_INDEX:
        diagram.add_b_node()
    else:
        raise Exception(f'Unexpected node type {ntype}')


class FRightRewriteTest(unittest.TestCase):

    @given(f_right_st())
    def test_simple_f_right_rewrite(self, data):
        center, neighbors, edges, transfer_edges, new_edges, phase = data
        ntype, phase = center
        diagram = ZXDiagram()
        add_node(ntype, phase, diagram)
        for ntype, phase in neighbors:
            add_node(ntype, phase, diagram)
        for s, t in edges:
            diagram.add_s_edge(s, t)
        match = list(f_right_matches(diagram))[0]
        f_right_rewrite(match, phase, new_edges, transfer_edges, diagram)
        self.assertEqual(len(neighbors) + 2, diagram.number_of_nodes())
        self.assertEqual(new_edges + len(edges), diagram.number_of_edges())
