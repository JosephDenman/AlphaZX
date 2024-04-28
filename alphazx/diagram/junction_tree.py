import networkx as nx

import itertools

from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.zx_diagram import ZXDiagram

d = clifford_zx_diagram(100, 100)


def compute_junction_tree(zx_diagram: ZXDiagram) -> nx.Graph:
    g = nx.Graph()
    for n, ndata in zx_diagram.nodes(data=True):
        g.add_node(n, **ndata)
    for m, n in zx_diagram.edges():
        g.add_edge(m, n, num_edges=zx_diagram.number_of_edges(n, m))
    return nx.junction_tree(g)


# print('d.nodes = ', d.nodes(data=True))
# print('d.edges = ', d.edges(data=True))
print()
jd = compute_junction_tree(d)
# print('jd.nodes = ', jd.nodes(data=True))
# print('jd.edges = ', jd.edges(data=True))

matches = list(d.compute_matches())
# for match in matches:
#     assert any([jd.has_node(tuple(nodes)) for nodes in itertools.permutations(match.nodes)]), f'Match {match} not in junction tree'

nodes_to_remove = []
for n, ndata in jd.nodes(data=True):
    if len(n) == 3 or len(n) > 4:
        print('n = ', n)
        print('ndata = ', ndata)
        nodes_to_remove.append(n)

jd.remove_nodes_from(nodes_to_remove)

print(len(matches))
print(jd.number_of_nodes())
