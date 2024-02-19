import random

import networkx as nx
from diagram.pyzx_graph_generator import clifford_nx_graph
from diagram.zx_diagram import ZXDiagram
from diagram.zx_match_diagram import to_zx_match_diagram
from diagram.match import FRightMatch, Match


def gen_f_right_parameters(f_right_match: FRightMatch, diagram: ZXDiagram, d: int) -> tuple[
        float, int, set[tuple[int, int]]]:
    phase = random.choice([n / d for n in list(range(d * 2))])
    new_edges = random.randint(1, 5)
    current_edges = diagram.incident_edges(f_right_match[0])
    k = random.randint(0, len(current_edges))
    transfer_edges = set(random.sample(sorted(current_edges), k))
    return phase, new_edges, transfer_edges


g = clifford_nx_graph(30, 30)
d = ZXDiagram(g)
print('d_nodes = ', d.nodes(data=True))
print('d_edges = ', d.edges(data=False))
print('md_nodes = ', d.nodes(data=False))
print('md_edges = ', d.edges(data=False))
smd = to_zx_match_diagram(d)


def stringify(v: Match | list | str) -> str:
    if isinstance(v, Match):
        return repr(v)
    elif isinstance(v, list):
        return str(v)
    elif isinstance(v, str):
        return v
    else:
        raise Exception(f'Unsupported value {type(v)}')


"""
Next steps:

1. Use categorical representation of phase and type.
2. Add degree information before converting to a match diagram.
3. Implement environment (MuZero)
4. Implement models. 

"""