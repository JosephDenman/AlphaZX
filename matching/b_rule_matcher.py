from collections.abc import Iterator

import networkx as nx
from graph.pyzx_nx_conv import S_ETYPE_INDEX, Z_NTYPE_INDEX, X_NTYPE_INDEX
from matching.match import BLeftMatch, BRightMatch
from matching.zx_diagram import ZXDiagram


def b_right_pattern() -> ZXDiagram:
    """
    x0 -- z1
    """
    diagram = ZXDiagram()
    x, z = diagram.add_x_node(0), diagram.add_z_node(0)
    diagram.add_s_edge(x, z)
    return diagram


def b_right_matches(diagram: ZXDiagram) -> Iterator[BRightMatch]:
    for s, t in set(diagram.edges(data=False)):
        if diagram.degree(s) == 3 and diagram.degree(t) == 3:
            if diagram.is_basis(s) and diagram.is_basis(t) and diagram.type(s) != diagram.type(t):
                if diagram.phase(s) == 0 and diagram.phase(t) == 0:
                    x, z = (s, t) if diagram.is_x_basis(s) and diagram.is_z_basis(t) else (t, s)
                    yield BRightMatch(x, z)


def b_left_pattern(bl: int = 0, br: int = 1, tl: int = 2, tr: int = 3) -> ZXDiagram:
    graph = nx.MultiGraph()
    graph.add_nodes_from([bl, br], type=Z_NTYPE_INDEX, phase=0)
    graph.add_nodes_from([tl, tr], type=X_NTYPE_INDEX, phase=0)
    graph.add_edges_from([(bl, tl), (bl, tr), (br, tl), (br, tr)], type=S_ETYPE_INDEX)
    return ZXDiagram(graph)


def b_left_matches(diagram: ZXDiagram) -> Iterator[BLeftMatch]:
    candidates = {(s, t) if diagram.is_z_basis(s) and diagram.is_x_basis(t) else (t, s) for s, t in
                  set(diagram.edges(data=False))
                  if diagram.degree(s) == 3 and diagram.degree(t) == 3 and
                  diagram.is_basis(s) and diagram.is_basis(t) and diagram.type(s) != diagram.type(t) and
                  diagram.phase(s) == 0 and diagram.phase(t) == 0}
    while len(candidates) > 0:
        z, x = candidates.pop()
        for n in diagram.neighbors(z):
            if not n == x and diagram.is_x_basis(n) and diagram.phase(n) == 0 and diagram.degree(n) == 3:
                for m in diagram.neighbors(n):
                    if not m == z and diagram.is_z_basis(m) and diagram.phase(m) == 0 and diagram.degree(m) == 3:
                        for o in diagram.neighbors(m):
                            if o == x:
                                candidates.discard((z, x))
                                candidates.discard((z, n))
                                candidates.discard((m, x))
                                candidates.discard((m, n))
                                yield BLeftMatch(z, x, m, n)
