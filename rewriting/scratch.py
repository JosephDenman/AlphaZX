import random

import numpy as np

from diagram.pyzx_graph_generator import nx_clifford_graph
from diagram.pyzx_nx_conv import nx_to_pyg_hetero
from diagram.zx_diagram import ZXDiagram
from diagram.zx_match_diagram import ZXMatchDiagram
from diagram.match import FRightMatch
from rewriting.util import rewrite


def gen_f_right_parameters(f_right_match: FRightMatch, diagram: ZXDiagram, d: int) -> tuple[
        float, int, set[tuple[int, int]]]:
    phase = random.choice([n / d for n in list(range(d * 2))])
    new_edges = random.randint(1, 5)
    current_edges = diagram.incident_edges(f_right_match[0])
    k = random.randint(0, len(current_edges))
    transfer_edges = set(random.sample(sorted(current_edges), k))
    return phase, new_edges, transfer_edges


denominator = 2
g = ZXDiagram(nx_clifford_graph(10, 10))
i = 0

"""
Next steps:

1. Use categorical representation of phase and type.
2. Add degree information before converting to a match diagram.
3. Implement environment (MuZero)
4. Implement models. 

"""

while True:
    m = np.random.choice(list(g.compute_matches()))
    print(
        '-------------------------------------------------------------------------------------------------------------')
    print('i = ', i)
    print('m = ', m)
    print('g = ', g)
    params = gen_f_right_parameters(m, g, denominator) if isinstance(m, FRightMatch) else None
    print('params = ', params)
    rewrite(m, g, params)
    m_diagram = ZXMatchDiagram(g)
    print('match_diagram = ', m_diagram)
    print('match_diagram.nodes = ', m_diagram.nodes(data=True))
    print('match_diagram.edges = ', m_diagram.edges(data=True))
    print('m_diagram_hetero = ', nx_to_pyg_hetero(g, 'type'))
    i = i + 1
