from src.diagram.match import Match, FLeftMatch, FRightMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from src.diagram.zx_diagram import ZXDiagram
from src.rewriting.b_rule_rewriter import b_left_rewrite, b_right_rewrite
from src.rewriting.f_rule_rewriter import f_left_rewrite, f_right_rewrite
from src.rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite

FRightParameters = tuple[float, int, set[tuple[int, int]]]


def rewrite(diagram: ZXDiagram, match: Match, f_right_params: FRightParameters | None = None) -> None:
    if isinstance(match, FLeftMatch):
        f_left_rewrite(match, diagram)
    elif isinstance(match, FRightMatch):
        if f_right_params is None:
            raise Exception(f'Expected parameters for {match}')
        phase, new_edges, transfer_edges = f_right_params
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
