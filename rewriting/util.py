from diagram.diagram_generators import clifford_zx_diagram
from diagram.zx_diagram import ZXDiagram
from diagram.match import Match, FLeftMatch, FRightMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from diagram.zx_match_diagram import to_zx_match_diagram
from rewriting.b_rule_rewriter import b_left_rewrite, b_right_rewrite
from rewriting.f_rule_rewriter import f_left_rewrite, f_right_rewrite
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite

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


nodes = []
for i in range(1000):
    print('i = ', i)
    d = clifford_zx_diagram(100, 100, True)
    md = to_zx_match_diagram(d, False)
    nodes.append(d.number_of_nodes() + md.number_of_nodes())

print(sum(nodes)/len(nodes))
