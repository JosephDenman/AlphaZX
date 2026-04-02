from alphazx.diagram import ZXMatchDiagram
from alphazx.diagram.match import MatchNode, FLeftMatch, FRightMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from alphazx.diagram.zx_diagram import ZXDiagram
from .update_set import UpdateSet
from .b_rule_rewriter import b_left_rewrite, b_right_rewrite
from .f_rule_rewriter import f_left_rewrite, f_right_rewrite
from .y_rule_rewriter import y_left_rewrite, y_right_rewrite


def rewrite(diagram: ZXDiagram, match: MatchNode, f_right_params: tuple[float, int, set[int]] | None = None) -> UpdateSet:
    if isinstance(match, FLeftMatch):
        return f_left_rewrite(match, diagram)
    elif isinstance(match, FRightMatch):
        if f_right_params is None:
            raise Exception(f'Expected parameters for {match}')
        phase, new_edges, transfer_edges = f_right_params
        return f_right_rewrite(match, phase, new_edges, transfer_edges, diagram)
    elif isinstance(match, BLeftMatch):
        return b_left_rewrite(match, diagram)
    elif isinstance(match, BRightMatch):
        return b_right_rewrite(match, diagram)
    elif isinstance(match, YLeftMatch):
        return y_left_rewrite(match, diagram)
    elif isinstance(match, YRightMatch):
        return y_right_rewrite(match, diagram)
    else:
        raise Exception(f'Bug found: unexpected match type {match}')


