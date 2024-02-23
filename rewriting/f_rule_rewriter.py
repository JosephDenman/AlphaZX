from diagram.zx_diagram import ZXDiagram
from diagram.match import FLeftMatch, FRightMatch


def add_node(f_match: FRightMatch | FLeftMatch, phase: float, diagram: ZXDiagram) -> int:
    if f_match.is_z_basis:
        return diagram.add_z_node(phase)
    else:
        return diagram.add_x_node(phase)


def assert_is_basis_node(zx_diagram: ZXDiagram, n: int) -> None:
    assert zx_diagram.is_basis(n), f'Node {n} is not a basis node'


def f_left_rewrite(f_left_match: FLeftMatch, zx_diagram: ZXDiagram) -> None:
    left, right = f_left_match
    assert_is_basis_node(zx_diagram, left)
    assert_is_basis_node(zx_diagram, right)
    node = add_node(f_left_match, zx_diagram.phase(left) + zx_diagram.phase(right), zx_diagram)
    for neighbor in zx_diagram.neighbors_from(f_left_match):
        zx_diagram.add_s_edge(node, neighbor)
    zx_diagram.remove_incident_edges(left)
    zx_diagram.remove_incident_edges(right)
    zx_diagram.remove_nodes_from(f_left_match.nodes)


def assert_is_valid_phase(zx_diagram: ZXDiagram, phase: float) -> None:
    assert zx_diagram.is_valid_phase(
        phase), f'Phase {phase} is invalid for diagram with phase denominator {zx_diagram.phase_denominator}'


def f_right_rewrite(f_right_match: FRightMatch, phase: float, new_edges: int,
                    transfer_edges: set[tuple[int, int]],
                    zx_diagram: ZXDiagram) -> None:
    assert_is_valid_phase(zx_diagram, phase)
    center = f_right_match.nodes[0]
    left = add_node(f_right_match, phase, zx_diagram)
    right = add_node(f_right_match, zx_diagram.phase(center) - phase, zx_diagram)
    zx_diagram.add_s_edges_from([(left, right)] * new_edges)
    for _, neighbor, k in zx_diagram.incident_edges(center):
        zx_diagram.add_s_edge(right if (neighbor, k) in transfer_edges else left, neighbor)
    zx_diagram.remove_incident_edges(center)
    zx_diagram.remove_nodes_from(f_right_match.nodes)
