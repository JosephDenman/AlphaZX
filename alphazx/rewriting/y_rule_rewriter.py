from alphazx.diagram.match import YRightMatch, YLeftMatch, FRightZMatch, FRightXMatch
from alphazx.diagram.zx_diagram import ZXDiagram
from .update_set import UpdateSet


def add_node_with_flipped_basis(basis: str, phase: float, zx_diagram: ZXDiagram) -> int:
    if basis == FRightZMatch.abbrev:
        return zx_diagram.add_x_node(phase)
    elif basis == FRightXMatch.abbrev:
        return zx_diagram.add_z_node(phase)
    else:
        raise Exception(f'Unexpected basis {basis}')


def y_right_rewrite(y_right_match: YRightMatch, diagram: ZXDiagram) -> UpdateSet:
    bottom, center, top_left, top_right = y_right_match

    [top_left_neighbor] = [n for n in diagram.neighbors(top_left) if n != center]
    [top_right_neighbor] = [n for n in diagram.neighbors(top_right) if n != center]
    [bottom_neighbor] = [n for n in diagram.neighbors(bottom) if n != center]

    new_bottom = add_node_with_flipped_basis(diagram.basis(bottom), -0.5, diagram)
    new_center = add_node_with_flipped_basis(diagram.basis(center), 0., diagram)
    new_top_left = add_node_with_flipped_basis(diagram.basis(top_left), 0.5, diagram)
    new_top_right = add_node_with_flipped_basis(diagram.basis(top_right), 0.5, diagram)

    diagram.add_s_edges_from([(bottom_neighbor, new_bottom), (new_bottom, new_center), (new_center, new_top_left),
                              (new_top_left, top_left_neighbor), (new_center, new_top_right),
                              (new_top_right, top_right_neighbor)])

    return UpdateSet({bottom, center, top_left, top_right}, {new_bottom, new_center, new_top_left, new_top_right},
                     y_right_match)


def y_left_rewrite(y_left_match: YLeftMatch, diagram: ZXDiagram) -> UpdateSet:
    bottom, center, top_left, top_right = y_left_match

    [top_left_neighbor] = [n for n in diagram.neighbors(top_left) if n != center]
    [top_right_neighbor] = [n for n in diagram.neighbors(top_right) if n != center]
    [bottom_neighbor] = [n for n in diagram.neighbors(bottom) if n != center]

    new_bottom = add_node_with_flipped_basis(diagram.basis(bottom), 0.5, diagram)
    new_center = add_node_with_flipped_basis(diagram.basis(center), -0.5, diagram)
    new_top_left = add_node_with_flipped_basis(diagram.basis(top_left), -0.5, diagram)
    new_top_right = add_node_with_flipped_basis(diagram.basis(top_right), -0.5, diagram)

    diagram.add_s_edges_from([(bottom_neighbor, new_bottom), (new_bottom, new_center), (new_center, new_top_left),
                              (new_top_left, top_left_neighbor), (new_center, new_top_right),
                              (new_top_right, top_right_neighbor)])

    return UpdateSet({bottom, center, top_left, top_right}, {new_bottom, new_center, new_top_left, new_top_right},
                     y_left_match)
