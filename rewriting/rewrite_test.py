from typing import Optional

from hypothesis import strategies as st, given
from hypothesis.strategies import composite, SearchStrategy

from diagram.match import BLeftMatch, YLeftZMatch, YLeftXMatch, YRightZMatch, YRightXMatch, BRightMatch
from diagram.match_patterns import b_right_pattern, b_left_pattern, y_left_z_pattern, y_left_x_pattern, \
    y_right_x_pattern, y_right_z_pattern
from diagram.pyzx_nx_conv import is_z_basis, is_x_basis, is_boundary, Z_NTYPE_INDEX, X_NTYPE_INDEX, B_NTYPE_INDEX
from diagram.zx_diagram import ZXDiagram
from rewriting.b_rule_rewriter import b_right_rewrite, b_left_rewrite
from rewriting.f_rule_rewriter import f_right_rewrite
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite

# Phase denominator for all tests
PD = 4


@composite
def f_right_st(draw):
    z_basis = draw(st.booleans())
    center = (1, draw(st_phase(PD))) if z_basis else (2, draw(st_phase(PD)))
    neighbor_number = draw(st.integers(min_value=1, max_value=20))
    neighbors = draw(st.lists(st_node(PD), min_size=neighbor_number, max_size=neighbor_number))
    edge_count = draw(st.integers(min_value=1, max_value=len(neighbors)))
    edges = draw(
        st.sets(st.tuples(st.sampled_from([0]), st.sampled_from(list(range(1, neighbor_number + 1)))),
                min_size=edge_count,
                max_size=edge_count))
    transfer_edge_count = draw(st.integers(min_value=0, max_value=len(edges)))
    transfer_edges = draw(
        st.sets(st.sampled_from(list(edges)), min_size=transfer_edge_count, max_size=transfer_edge_count))
    new_edges = draw(st.integers(min_value=1, max_value=20))
    phase = draw(st_phase(PD))
    return center, neighbors, edges, transfer_edges, new_edges, phase


def st_phase(d: int) -> SearchStrategy[float]:
    return st.sampled_from([n / d for n in range(2 * d)])


def st_node(d: Optional[int]) -> SearchStrategy[tuple[int, float]]:
    if d is None:
        return st.tuples(st.integers(0, 2), st.sampled_from([0]))
    else:
        return st.tuples(st.integers(0, 2), st_phase(d))


def st_b_right_nodes() -> SearchStrategy[list[tuple[int, float]]]:
    return st.lists(st_node(None), min_size=4, max_size=4)


def add_basis_node(ntype: int, phase: float, diagram: ZXDiagram) -> None:
    if ntype == Z_NTYPE_INDEX:
        diagram.add_z_node(phase=phase)
    elif ntype == X_NTYPE_INDEX:
        diagram.add_x_node(phase=phase)
    elif ntype == B_NTYPE_INDEX:
        diagram.add_b_node()
    else:
        raise Exception(f'Unexpected node type {ntype}')


class TestFRightRewrite:

    @given(f_right_st())
    def test_simple_f_right_rewrite(self, data):
        center, neighbors, edges, transfer_edges, new_edges, phase = data
        ntype, phase = center
        diagram = ZXDiagram(PD)
        add_basis_node(ntype, phase, diagram)
        for ntype, phase in neighbors:
            add_basis_node(ntype, phase, diagram)
        for s, t in edges:
            diagram.add_s_edge(s, t)
        match = list(diagram.f_right_matches())[0]
        f_right_rewrite(match, phase, new_edges, transfer_edges, diagram)
        assert len(neighbors) + 2 == diagram.number_of_nodes()
        assert new_edges + len(edges) == diagram.number_of_edges()


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


class TestBRightRewrite:

    @given(st_b_right_nodes())
    def test_simple_b_right_rewrite(self, data_list):
        bl_data, br_data, tl_data, tr_data = data_list
        diagram = ZXDiagram(PD, b_right_pattern(PD))
        diagram.add_s_edge(add_node(bl_data, diagram), 0)
        diagram.add_s_edge(add_node(br_data, diagram), 0)
        diagram.add_s_edge(1, add_node(tr_data, diagram))
        diagram.add_s_edge(1, add_node(tl_data, diagram))
        b_right_rewrite(list(diagram.b_right_matches())[0], diagram)
        assert list(diagram.b_left_matches())[0] == BLeftMatch(7, 9, 6, 8)


class TestBLeftRewrite:

    def test_simple_b_left_rewrite(self):
        diagram = b_left_pattern(PD)
        b4, b5, b6, b7 = diagram.add_b_node(), diagram.add_b_node(), diagram.add_b_node(), diagram.add_b_node()
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, b6), (3, b7)])
        b_left_rewrite(list(diagram.b_left_matches())[0], diagram)
        assert list(diagram.b_right_matches())[0] == BRightMatch(8, 9)


class TestYLeftRewrite:

    def test_self_z_rewrite(self):
        diagram = y_left_z_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_left_rewrite(list(diagram.y_left_matches())[0], diagram)
        assert list(diagram.y_right_matches())[0] == YRightXMatch(0, 1, 2, 3)

    def test_self_x_rewrite(self):
        diagram = y_left_x_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_left_rewrite(list(diagram.y_left_matches())[0], diagram)
        assert list(diagram.y_right_matches())[0] == YRightZMatch(0, 1, 2, 3)


class TestYRightRewrite:

    def test_self_z_rewrite(self):
        diagram = y_right_z_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_right_rewrite(list(diagram.y_right_matches())[0], diagram)
        assert list(diagram.y_left_matches())[0] == YLeftXMatch(0, 1, 2, 3)

    def test_self_x_rewrite(self):
        diagram = y_right_x_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        y_right_rewrite(list(diagram.y_right_matches())[0], diagram)
        assert list(diagram.y_left_matches())[0] == YLeftZMatch(0, 1, 2, 3)
