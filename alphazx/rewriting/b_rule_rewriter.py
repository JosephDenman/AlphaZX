from alphazx.diagram.match import BLeftMatch, BRightMatch, Basis, Match
from alphazx.diagram.zx_diagram import ZXDiagram


def validate_node(basis: Basis, n: int, diagram: ZXDiagram) -> None:
    if basis == 'z':
        assert diagram.is_z_basis(n), f'Expected node {n} to be Z basis'
    else:
        assert diagram.is_x_basis(n), f'Expected node {n} to be X basis'
    d = diagram.degree(n)
    assert d == 3, f'Expected node {n} to have degree three, has degree {d}'


def assert_neighbors(n: int, m: int, diagram: ZXDiagram) -> None:
    assert n in diagram.neighbors(m), f'Expected {n} and {m} to be neighbors'


def assert_degree(n: int, expected_degree: int, diagram: ZXDiagram) -> None:
    actual_degree = diagram.degree(n)
    assert actual_degree == expected_degree, f'Expected degree {expected_degree} for node {n} but found {actual_degree}'


def non_match_neighbors(a: int, match: Match, diagram: ZXDiagram) -> list[int]:
    return [b for b in diagram.neighbors(a) if b not in match.nodes]


def are_neighbors(a: int, b: int, diagram: ZXDiagram) -> bool:
    return a in diagram.neighbors(b)


def b_left_rewrite(b_left_match: BLeftMatch, diagram: ZXDiagram) -> None:
    z, x, m, n = b_left_match

    validate_node('z', z, diagram)
    validate_node('x', x, diagram)
    validate_node('z', m, diagram)
    validate_node('x', n, diagram)

    assert_degree(z, 3, diagram)
    assert_degree(x, 3, diagram)
    assert_degree(m, 3, diagram)
    assert_degree(n, 3, diagram)

    assert_neighbors(z, x, diagram)
    assert_neighbors(x, m, diagram)
    assert_neighbors(m, n, diagram)
    assert_neighbors(n, z, diagram)

    bottom, top = diagram.add_x_node(0), diagram.add_z_node(0)
    diagram.add_s_edge(bottom, top)

    if are_neighbors(z, m, diagram):
        diagram.add_s_edge(bottom, bottom)
    else:
        [z_neighbor] = non_match_neighbors(z, b_left_match, diagram)
        [m_neighbor] = non_match_neighbors(m, b_left_match, diagram)
        diagram.add_s_edges_from([(z_neighbor, bottom), (m_neighbor, bottom)])
    diagram.remove_z_node(z)
    diagram.remove_z_node(m)

    if are_neighbors(x, n, diagram):
        diagram.add_s_edge(top, top)
    else:
        [x_neighbor] = non_match_neighbors(x, b_left_match, diagram)
        [n_neighbor] = non_match_neighbors(n, b_left_match, diagram)
        diagram.add_s_edges_from([(top, x_neighbor), (top, n_neighbor)])
    diagram.remove_x_node(x)
    diagram.remove_x_node(n)

    assert_degree(bottom, 3, diagram)
    assert_degree(top, 3, diagram)


def b_right_rewrite(b_right_match: BRightMatch, diagram: ZXDiagram) -> None:
    z, x = b_right_match

    validate_node('x', x, diagram)
    validate_node('z', z, diagram)

    assert_degree(z, 3, diagram)
    assert_degree(x, 3, diagram)

    assert_neighbors(x, z, diagram)

    b_nmn = non_match_neighbors(x, b_right_match, diagram)
    [bl, br] = b_nmn if len(b_nmn) == 2 else [b_nmn[0], b_nmn[0]]

    t_nmn = non_match_neighbors(z, b_right_match, diagram)
    [tl, tr] = t_nmn if len(t_nmn) == 2 else [t_nmn[0], t_nmn[0]]

    z0, z1, x2, x3 = diagram.add_z_node(0), diagram.add_z_node(0), diagram.add_x_node(0), diagram.add_x_node(0)

    diagram.add_s_edges_from([(bl, z0), (br, z1)])
    diagram.add_s_edges_from([(n, m) for n in [z0, z1] for m in [x2, x3]])
    diagram.add_s_edges_from([(x2, tl), (x3, tr)])

    diagram.remove_x_node(x)
    diagram.remove_z_node(z)

    validate_node('z', z0, diagram)
    validate_node('z', z1, diagram)
    validate_node('x', x2, diagram)
    validate_node('x', x3, diagram)

    assert_degree(z0, 3, diagram)
    assert_degree(z1, 3, diagram)
    assert_degree(x2, 3, diagram)
    assert_degree(x3, 3, diagram)

    assert_neighbors(bl, z0, diagram)
    assert_neighbors(br, z1, diagram)
    assert_neighbors(x2, tl, diagram)
    assert_neighbors(x3, tr, diagram)
