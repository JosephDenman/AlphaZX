from diagram.match import YRightMatch, YLeftMatch
from diagram.zx_diagram import ZXDiagram


def y_right_rewrite(y_right_match: YRightMatch, diagram: ZXDiagram) -> None:
    """
    As with all rewrites defined in this package, the behavior of this rewrite is unreliable if the diagram has
    changed since the given match was computed.
    """
    bottom, center, top_left, top_right = y_right_match
    diagram.flip_basis(bottom)
    diagram.set_phase(bottom, -0.5)
    diagram.flip_basis(center)
    diagram.set_phase(center, 0)
    diagram.flip_basis(top_left)
    diagram.set_phase(top_left, 0.5)
    diagram.flip_basis(top_right)
    diagram.set_phase(top_right, 0.5)


def y_left_rewrite(y_left_match: YLeftMatch, diagram: ZXDiagram) -> None:
    """
    As with all rewrites defined in this package, the behavior of this rewrite is unreliable if the diagram has
    changed since the given match was computed.
    """
    bottom, center, top_left, top_right = y_left_match
    diagram.flip_basis(bottom)
    diagram.set_phase(bottom, 0.5)
    diagram.flip_basis(center)
    diagram.set_phase(center, -0.5)
    diagram.flip_basis(top_left)
    diagram.set_phase(top_left, -0.5)
    diagram.flip_basis(top_right)
    diagram.set_phase(top_right, -0.5)
