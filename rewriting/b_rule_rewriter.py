from matching.match_types import BLeftMatch, BRightMatch, Basis, Match
from matching.zx_diagram import ZXDiagram


def validate_node(basis: Basis, n: int, diagram: ZXDiagram) -> None:
    if basis == 'z':
        assert diagram.is_z_basis(n), f'Expected node {n} to be Z basis'
    else:
        assert diagram.is_x_basis(n), f'Expected node {n} to be X basis'
    d = diagram.degree(n)
    assert d == 3, f'Expected node {n} to have degree three, has degree {d}'


def assert_neighbors(n: int, m: int, diagram: ZXDiagram) -> None:
    assert n in diagram.neighbors(m), f'Expected {n} and {m} to be neighbors'


def b_left_rewrite(b_left_match: BLeftMatch, diagram: ZXDiagram) -> None:
    """
    Replace these manual checks at the top with a subgraph verification match.
    """
    bottom_left, bottom_right, top_left, top_right = b_left_match

    validate_node('z', bottom_left, diagram)
    validate_node('z', bottom_right, diagram)
    validate_node('x', top_left, diagram)
    validate_node('x', top_right, diagram)

    assert_neighbors(top_left, bottom_left, diagram)
    assert_neighbors(top_left, bottom_right, diagram)
    assert_neighbors(top_right, bottom_left, diagram)
    assert_neighbors(top_right, bottom_right, diagram)

    bottom, top = diagram.add_x_node(0), diagram.add_z_node(0)

    bottom_left_neighbors = match_neighbors(bottom_left, b_left_match, diagram)
    assert len(bottom_left_neighbors) == 1, f'Expected {bottom_left_neighbors} to have one element'

    bottom_right_neighbors = match_neighbors(bottom_right, b_left_match, diagram)
    assert len(bottom_right_neighbors) == 1, f'Expected {bottom_right_neighbors} to have one element'

    top_left_neighbors = match_neighbors(top_left, b_left_match, diagram)
    assert len(top_left_neighbors) == 1, f'Expected {top_left_neighbors} to have one element'

    top_right_neighbors = match_neighbors(top_right, b_left_match, diagram)
    assert len(top_right_neighbors) == 1, f'Expected {top_right_neighbors} to have one element'

    diagram.add_s_edges_from([(bottom, bottom_left_neighbors[0]), (bottom, bottom_right_neighbors[0])])
    diagram.add_s_edge(bottom, top)
    diagram.add_s_edges_from([(top, top_left_neighbors[0]), (top, top_right_neighbors[0])])

    diagram.remove_incident_edges(top_left)
    diagram.remove_incident_edges(top_right)
    diagram.remove_incident_edges(bottom_left)
    diagram.remove_incident_edges(bottom_right)

    diagram.remove_x_node(top_left)
    diagram.remove_x_node(top_right)
    diagram.remove_z_node(bottom_left)
    diagram.remove_z_node(bottom_right)


def match_neighbors(n: int, match: Match, diagram: ZXDiagram) -> list[int]:
    return [neighbor for neighbor in diagram.neighbors(n) if neighbor not in match]


def b_right_rewrite(b_right_match: BRightMatch, diagram: ZXDiagram) -> None:
    """
    NOTE: This rewrite only works for ideal match case, with four distinct boundary nodes.
    """
    bottom, top = b_right_match
    validate_node('x', bottom, diagram)
    validate_node('z', top, diagram)

    assert_neighbors(bottom, top, diagram)

    bottom_neighbors = match_neighbors(bottom, b_right_match, diagram)
    assert len(bottom_neighbors) == 2, f'Expected {bottom_neighbors} to have two elements'

    top_neighbors = match_neighbors(top, b_right_match, diagram)
    assert len(top_neighbors) == 2, f'Expected {bottom_neighbors} to have two elements'

    top_left, top_right = diagram.add_x_node(0), diagram.add_x_node(0)
    bottom_left, bottom_right = diagram.add_z_node(0), diagram.add_z_node(0)

    diagram.add_s_edges_from([(top_left, top_neighbors[0]), (top_right, top_neighbors[1])])
    diagram.add_s_edges_from([(n, m) for n in [bottom_left, bottom_right] for m in [top_left, top_right]])
    diagram.add_s_edges_from([(bottom_left, bottom_neighbors[0]), (bottom_right, bottom_neighbors[1])])

    diagram.remove_incident_edges(top)
    diagram.remove_incident_edges(bottom)

    diagram.remove_x_node(bottom)
    diagram.remove_z_node(top)
