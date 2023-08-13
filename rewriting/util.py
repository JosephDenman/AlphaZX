from diagram.zx_diagram import ZXDiagram
from matching.match import Match, FLeftMatch, FRightMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from rewriting.b_rule_rewriter import b_left_rewrite, b_right_rewrite
from rewriting.f_rule_rewriter import f_left_rewrite, f_right_rewrite
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite

FRightParameters = tuple[float, int, set[tuple[int, int]]]


def rewrite(match: Match, diagram: ZXDiagram, f_right_parameters: FRightParameters | None = None) -> None:
    if isinstance(match, FLeftMatch):
        f_left_rewrite(match, diagram)
    elif isinstance(match, FRightMatch):
        if f_right_parameters is None:
            raise Exception(f'Expected parameters for {match}')
        phase, new_edges, transfer_edges = f_right_parameters
        f_right_rewrite(match, phase, new_edges, transfer_edges, diagram)
    elif isinstance(match, BLeftMatch):
        b_left_rewrite(match, diagram)
    elif isinstance(match, BRightMatch):
        b_right_rewrite(match, diagram)
    elif isinstance(match, YLeftMatch):
        y_left_rewrite(match, diagram)
    elif isinstance(match, YRightMatch):
        y_right_rewrite(match, diagram)
    else:
        raise Exception(f'Bug found: unexpected match type {match}')
