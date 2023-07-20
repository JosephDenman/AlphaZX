import random

from graph.pyzx_graph_generator import nx_clifford_graph
from matching.diagram_match_diagram_conv import compute_matches
from matching.match_types import Match, FRightMatch, FLeftMatch, BLeftMatch, BRightMatch, YLeftMatch, YRightMatch
from matching.zx_diagram import ZXDiagram
import numpy as np

from rewriting.b_rule_rewriter import b_left_rewrite, b_right_rewrite
from rewriting.f_rule_rewriter import f_right_rewrite, f_left_rewrite
from rewriting.y_rule_rewriter import y_left_rewrite, y_right_rewrite


def gen_f_right_parameters(f_right_match: FRightMatch, diagram: ZXDiagram) -> tuple[float, int, set[tuple[int, int]]]:
    phase = random.choice([n / d for n in list(range(d * 2))])
    new_edges = random.randint(1, 100)
    current_edges = diagram.incident_edges(f_right_match[0])
    k = random.randint(0, len(current_edges))
    transfer_edges = set(random.sample(current_edges, k))
    return phase, new_edges, transfer_edges


def rewriter(match: Match, diagram: ZXDiagram) -> None:
    if isinstance(match, FLeftMatch):
        f_left_rewrite(match, diagram)
    elif isinstance(match, FRightMatch):
        phase, new_edges, transfer_edges = gen_f_right_parameters(match, diagram)
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


d = ZXDiagram(nx_clifford_graph(10, 10))
for i in range(10):
    m = np.random.choice(list(compute_matches(d)))
    print('i = ', i)
    print('m = ', m)
    rewriter(m, d)
