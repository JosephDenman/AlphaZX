import random

import numpy as np

from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conv import nx_to_pyg_hetero
from matching.match import Match, FRightMatch, FLeftMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from matching.utils import compute_matches
from matching.zx_diagram import ZXDiagram
from matching.zx_match_diagram import ZXMatchDiagram
from rewriting.b_rule_rewriter import b_left_rewrite, b_right_rewrite
from rewriting.f_rule_rewriter import f_right_rewrite, f_left_rewrite
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite


def gen_f_right_parameters(f_right_match: FRightMatch, diagram: ZXDiagram, d: int) -> tuple[
    float, int, set[tuple[int, int]]]:
    phase = random.choice([n / d for n in list(range(d * 2))])
    new_edges = random.randint(1, 5)
    current_edges = diagram.incident_edges(f_right_match[0])
    k = random.randint(0, len(current_edges))
    transfer_edges = set(random.sample(sorted(current_edges), k))
    return phase, new_edges, transfer_edges


def rewriter(match: Match, diagram: ZXDiagram, d: int, c: dict[str, int]) -> None:
    if isinstance(match, FLeftMatch):
        f_left_rewrite(match, diagram)
        c['f_lefts'] = c['f_lefts'] + 1
    elif isinstance(match, FRightMatch):
        phase, new_edges, transfer_edges = gen_f_right_parameters(match, diagram, d)
        f_right_rewrite(match, phase, new_edges, transfer_edges, diagram)
        c['f_rights'] = c['f_rights'] + 1
    elif isinstance(match, BLeftMatch):
        b_left_rewrite(match, diagram)
        c['b_lefts'] = c['b_lefts'] + 1
    elif isinstance(match, BRightMatch):
        b_right_rewrite(match, diagram)
        c['b_rights'] = c['b_rights'] + 1
    elif isinstance(match, YLeftMatch):
        y_left_rewrite(match, diagram)
        c['y_lefts'] = c['y_lefts'] + 1
    elif isinstance(match, YRightMatch):
        y_right_rewrite(match, diagram)
        c['y_rights'] = c['y_rights'] + 1
    else:
        raise Exception(f'Bug found: unexpected match type {match}')


denominator = 2
g = ZXDiagram(nx_clifford_graph(10, 10))
i = 0
counts = {
    'f_lefts': 0,
    'f_rights': 0,
    'b_lefts': 0,
    'b_rights': 0,
    'y_lefts': 0,
    'y_rights': 0
}

"""
Next steps:

1. Use categorical representation of phase and type.
2. Add degree information before converting to a match graph.
3. Implement environment (MuZero)
4. Implement model. 

"""

while True:
    m = np.random.choice(list(compute_matches(g)))
    print(
        '-------------------------------------------------------------------------------------------------------------')
    print('i = ', i)
    print('m = ', m)
    print('g = ', g)
    print('counts = ', counts)
    rewriter(m, g, denominator, counts)
    m_diagram = ZXMatchDiagram(g)
    print('match_diagram = ', m_diagram)
    print('match_diagram.nodes = ', m_diagram.nodes(data=True))
    print('match_diagram.edges = ', m_diagram.edges(data=True))
    print('m_diagram_hetero = ', nx_to_pyg_hetero(g, 'type'))
    i = i + 1
