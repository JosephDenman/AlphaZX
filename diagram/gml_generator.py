import networkx as nx
import torch

from diagram.diagram_generators import clifford_zx_diagram
from diagram.match import Match
from diagram.zx_match_diagram import to_zx_match_diagram, ZXMatchDiagram

d = clifford_zx_diagram(10, 10, True)
md = to_zx_match_diagram(d, False)


def stringify(v: Match | list | str) -> str:
    if isinstance(v, Match):
        return repr(v)
    elif isinstance(v, list):
        return str(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, torch.Tensor):
        return str(v.tolist())
    else:
        raise Exception(f'Unsupported value {type(v)}')


def print_nodes(zx_match_diagram: ZXMatchDiagram) -> None:
    for n in zx_match_diagram.nodes(data=True):
        print('n = ', *n)


def print_edges(zx_match_diagram: ZXMatchDiagram) -> None:
    for s, t, k, edata in zx_match_diagram.edges(data=True, keys=True):
        print('e = ', {'source': s, 'target': t, 'key': k, 'edata': edata})


print_nodes(md)
print_edges(md)

nx.write_gml(d, './diagram.gml', stringify)
nx.write_gml(md, './match_diagram.gml', stringify)
