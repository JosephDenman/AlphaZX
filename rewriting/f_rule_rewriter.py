from diagram.zx_diagram import ZXDiagram
from diagram.match import FLeftMatch, FRightMatch


def add_node(f_match: FRightMatch | FLeftMatch, phase: float, diagram: ZXDiagram) -> int:
    if f_match.is_z_basis:
        return diagram.add_z_node(phase)
    else:
        return diagram.add_x_node(phase)


def f_left_rewrite(f_left_match: FLeftMatch, diagram: ZXDiagram) -> None:
    left, right = f_left_match
    assert diagram.is_basis(left), f'Node {left} is not a basis node'
    assert diagram.is_basis(right), f'Node {right} is not a basis node'
    node = add_node(f_left_match, (diagram.phase(left) + diagram.phase(right)) % 2, diagram)
    for neighbor in diagram.neighbors_from(f_left_match):
        diagram.add_s_edge(node, neighbor)
    diagram.remove_incident_edges(left)
    diagram.remove_incident_edges(right)
    diagram.remove_nodes_from(f_left_match)


def f_right_rewrite(f_right_match: FRightMatch, phase: float, new_edges: int,
                    transfer_edges: set[tuple[int, int]],
                    diagram: ZXDiagram) -> None:
    """
    :param f_right_match:
    :param phase:
    :param new_edges:
    :param transfer_edges: Set of tuples where the first component is a neighbor of the center node in the match
                           and the second number is the key of an edge between the neighbor and the center node in
                           the match.
    :param diagram:
    """
    assert -2 < phase < 2, f'Expected {phase} to be in [0, 2)'
    center = f_right_match._nodes[0]
    assert diagram.is_basis(center), f'Node {center} is not a basis node'
    center_phase = diagram.phase(center)
    assert -2 < center_phase < 2, f'Expected {center_phase} to be in [0, 2)'
    left = add_node(f_right_match, phase, diagram)
    right = add_node(f_right_match, (center_phase - phase) % 2, diagram)
    diagram.add_s_edges_from([(left, right)] * new_edges)
    for _, neighbor, k in diagram.incident_edges(center):
        diagram.add_s_edge(right if (neighbor, k) in transfer_edges else left, neighbor)
    diagram.remove_incident_edges(center)
    diagram.remove_nodes_from(f_right_match._nodes)
