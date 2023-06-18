from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import networkx as nx
import torch
import torch_geometric as pyg

from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conversion import NTYPE, is_basis, is_z_basis, is_x_basis, ETYPE
from matching.b_rule import b_left_matches, b_right_matches
from matching.match import Match, CompoundMatch, FRightZMatch, FRightXMatch, FRightMatch
from matching.f_rule import f_left_z_matches, f_right_z_matches, f_left_x_matches, f_right_x_matches
from matching.y_rule import y_left_z_matches, y_right_z_matches, y_left_x_matches, y_right_x_matches
from matching.zx_diagram import ZXDiagram


def compute_matches(diagram: nx.MultiGraph) -> Iterator[Match]:
    yield from f_right_z_matches(diagram)
    yield from f_right_x_matches(diagram)
    yield from f_left_z_matches(diagram)
    yield from f_left_x_matches(diagram)
    yield from b_left_matches(diagram)
    yield from b_right_matches(diagram)
    yield from y_left_z_matches(diagram)
    yield from y_left_x_matches(diagram)
    yield from y_right_z_matches(diagram)
    yield from y_right_x_matches(diagram)


I_ETYPE_INDEX = 3
I_ETYPE_NAME = 'inclusion'
B_ETYPE_INDEX = 4
B_ETYPE_NAME = 'bridge'
NTYPES = 'types'


def is_inclusion_edge(etype: str | int) -> bool:
    if isinstance(etype, str):
        return etype == I_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == I_ETYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def collect(dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {f'{k}s': torch.tensor([d[k] for d in dicts]) for k in dicts[0]}


def add_match(match_diagram: ZXMatchDiagram, diagram: nx.MultiGraph, match: Match) -> None:
    if not match_diagram.has_node(match):
        match_diagram.add_node(match, type=match.index,
                               **collect([diagram.nodes[node] for node in match]))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not match_diagram.has_node(sub_match):
                add_match(match_diagram, diagram, sub_match)
            if not match_diagram.has_edge(sub_match, match):
                match_diagram.add_edge(sub_match, match, type=I_ETYPE_INDEX)
    return


def f_right_match_from_ndata(n: int, ndata: dict[str, Any]) -> FRightMatch:
    if is_z_basis(ndata[NTYPE]):
        return FRightZMatch(n)
    elif is_x_basis(ndata[NTYPE]):
        return FRightXMatch(n)
    else:
        raise Exception(f'Unexpected node type {ndata[NTYPE]}')


def basis_nodes(diagram: nx.Graph) -> Iterator[int]:
    yield from (n for n, ndata in diagram.nodes(data=True) if is_basis(ndata[NTYPE]))


def basis_neighbors(diagram: nx.Graph, n: int) -> Iterator[int]:
    yield from (m for m in diagram.neighbors(n) if is_basis(diagram.nodes[m][NTYPE]))


def has_i_edge(match_diagram: nx.Graph, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_inclusion_edge(edata[ETYPE])
    return False


def inclusion_neighbors(match_diagram: nx.Graph, u_match: Match) -> Iterator[Match]:
    for u_neighbor in match_diagram.neighbors(u_match):
        if has_i_edge(match_diagram, u_match, u_neighbor):
            yield u_neighbor


def is_bridge_edge(etype: str | int) -> bool:
    if isinstance(etype, str):
        return etype == B_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == B_ETYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def has_b_edge(match_diagram: nx.Graph, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_bridge_edge(edata[ETYPE])
    return False


def is_match_neighbor(match_diagram: nx.Graph, u_match: Match, v_match: Match) -> bool:
    for u_neighbor in inclusion_neighbors(match_diagram, u_match):
        if v_match in inclusion_neighbors(match_diagram, u_neighbor):
            return True
    return False


def connected(match_diagram: nx.Graph, u_match: Match, v_match: Match) -> bool:
    return is_match_neighbor(match_diagram, u_match, v_match) or has_b_edge(match_diagram, u_match, v_match)


def add_composition_edges(match_diagram: nx.Graph, diagram: nx.MultiGraph) -> None:
    for u in basis_nodes(diagram):
        for v in basis_neighbors(diagram, u):
            u_match = f_right_match_from_ndata(u, diagram.nodes[u])
            v_match = f_right_match_from_ndata(v, diagram.nodes[v])
            if not connected(match_diagram, u_match, v_match):
                match_diagram.add_edge(u_match, v_match, type=B_ETYPE_INDEX)


def compute_match_diagram(diagram: ZXDiagram) -> ZXMatchDiagram:
    match_diagram = ZXMatchDiagram(diagram)
    matches = list(compute_matches(diagram))
    for match in matches:
        add_match(match_diagram, diagram, match)
    assert match_diagram.number_of_nodes() == len(matches), "Number of nodes in match diagram != number of matches"
    add_composition_edges(match_diagram, diagram)
    return match_diagram


def to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
        return torch.stack(value, dim=0).reshape((-1,))
    elif isinstance(value, torch.Tensor):
        return value.reshape((-1,))
    else:
        try:
            return torch.tensor(value, dtype=torch.long).reshape((-1,))
        except (ValueError, TypeError):
            pass


def ndata_to_feature(ndata: dict[str, Any]) -> torch.Tensor:
    feature = []
    for key, value in ndata.items():
        if key == NTYPE:
            feature.append(to_tensor(torch.nn.functional.one_hot(torch.tensor(value), 10)))
        elif key == NTYPES:
            feature.append(to_tensor(torch.nn.functional.one_hot(value, 10)))
        else:
            feature.append(to_tensor(value))
    return torch.cat(feature)


class ZXMatchDiagram(nx.Graph):
    NTYPE = 'type'
    NTYPES = 'types'
    PHASES = 'phases'
    DEGREES = 'degrees'

    def __init__(self, diagram: ZXDiagram, **attr):
        self.zx_diagram = diagram
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        # self.match_diagram = compute_match_diagram(self.zx_diagram)
        super().__init__(self.match_diagram, **attr)

    def __node_type_name(self, data: dict[str, Any] | Match) -> str:
        if isinstance(data, Match):
            return self.ntype_idx_to_name[self.match_diagram.nodes[data][NTYPE]]
        else:
            return self.ntype_idx_to_name[data[NTYPE]]

    def __edge_type_name(self, s: Match, t: Match, edata: dict[str, Any]) -> tuple[str, str, str]:
        s_ntype = self.match_diagram.nodes[s][NTYPE]
        t_ntype = self.match_diagram.nodes[t][NTYPE]
        e_type = I_ETYPE_NAME if is_inclusion_edge(edata[ETYPE]) else B_ETYPE_NAME
        return self.ntype_idx_to_name[s_ntype], e_type, self.ntype_idx_to_name[t_ntype]

    def __add_edge_to_e_hdata(self, s: Match, t: Match, n_order_dict: dict[Any, dict], edata: dict[str, Any], e_hdata):
        sources, targets = e_hdata[self.__edge_type_name(s, t, edata)]['edge_index']
        sources.append(n_order_dict[self.__node_type_name(s)][s])
        targets.append(n_order_dict[self.__node_type_name(t)][t])

    def to_pyg_heterograph(self):
        assert not self.zx_diagram.is_directed(), "Graph must be undirected"
        n_hdata = defaultdict(lambda: defaultdict(list))
        n_order_list = defaultdict(list)
        for n, ndata in self.match_diagram.nodes(data=True):
            ntype = self.__node_type_name(ndata)
            n_hdata[ntype]['x'].append(ndata_to_feature(ndata))
            n_order_list[ntype].append(n)
        n_order_dict = {}
        for ntype in n_hdata:
            n_hdata[ntype]['x'] = torch.stack(n_hdata[ntype]['x'])
            n_order_dict[ntype] = {node: i for i, node in enumerate(n_order_list[ntype])}
        added_edges = set()
        e_hdata = defaultdict(lambda: defaultdict(lambda: [[], []]))
        for source, target, edata in self.match_diagram.edges(data=True):
            if not (source, target) in added_edges and not (target, source) in added_edges:
                self.__add_edge_to_e_hdata(source, target, n_order_dict, edata, e_hdata)
                self.__add_edge_to_e_hdata(target, source, n_order_dict, edata, e_hdata)
                added_edges.add((source, target))
                added_edges.add((target, source))
        for etype in e_hdata:
            e_hdata[etype]['edge_index'] = torch.tensor(e_hdata[etype]['edge_index'], dtype=torch.long)
        n_hdata.update(e_hdata)
        return pyg.data.HeteroData(n_hdata)

    @property
    def ntype_idx_to_name(self) -> dict[int, str]:
        return {0: 'f_right_z',
                1: 'f_right_x',
                2: 'f_left_z',
                3: 'f_left_x',
                4: 'b_left',
                5: 'b_right',
                6: 'y_left_z',
                7: 'y_left_x',
                8: 'y_right_z',
                9: 'y_right_x'}

    @property
    def ntype_name_to_idx(self) -> dict[str, int]:
        return {v: k for k, v in self.ntype_idx_to_name}

    @property
    def etype_idx_to_name(self) -> dict[int, (str, str, str)]:
        return {0: ('f_left_z', 'inclusion', 'f_right_z'),
                1: ('f_right_z', 'inclusion', 'f_left_z'),
                2: ('f_left_x', 'inclusion', 'f_right_x'),
                3: ('f_right_x', 'inclusion', 'f_left_x'),
                4: ('b_left', 'inclusion', 'b_right'),
                5: ('b_right', 'inclusion', 'b_left'),
                6: ('b_left', 'inclusion', 'f_right_z'),
                7: ('f_right_z', 'inclusion', 'b_left'),
                8: ('b_left', 'inclusion', 'f_right_x'),
                9: ('f_right_x', 'inclusion', 'b_left'),
                10: ('b_right', 'inclusion', 'f_right_z'),
                11: ('f_right_z', 'inclusion', 'b_right'),
                12: ('b_right', 'inclusion', 'f_right_x'),
                13: ('f_right_x', 'inclusion', 'b_right'),
                14: ('f_right_z', 'inclusion', 'y_left_z'),
                15: ('y_left_z', 'inclusion', 'f_right_z'),
                16: ('f_right_x', 'inclusion', 'y_left_x'),
                17: ('y_left_x', 'inclusion', 'f_right_x'),
                18: ('y_right_z', 'inclusion', 'f_right_z'),
                19: ('f_right_z', 'inclusion', 'y_right_z'),
                20: ('y_right_z', 'inclusion', 'f_right_x'),
                21: ('f_right_x', 'inclusion', 'y_right_z'),
                22: ('y_right_x', 'inclusion', 'f_right_x'),
                23: ('f_right_x', 'inclusion', 'y_right_x'),
                24: ('y_right_x', 'inclusion', 'f_right_z'),
                25: ('f_right_z', 'inclusion', 'y_right_x'),
                26: ('y_left_z', 'inclusion', 'f_right_z'),
                27: ('f_right_z', 'inclusion', 'y_left_z'),
                28: ('y_left_z', 'inclusion', 'f_right_x'),
                29: ('f_right_x', 'inclusion', 'y_left_z'),
                30: ('y_left_x', 'inclusion', 'f_right_x'),
                31: ('f_right_x', 'inclusion', 'y_left_x'),
                32: ('y_left_x', 'inclusion', 'f_right_z'),
                33: ('f_right_z', 'inclusion', 'y_left_x'),
                34: ('f_right_z', 'bridge', 'f_right_x'),
                35: ('f_right_x', 'bridge', 'f_right_z')}

    @property
    def etype_name_to_idx(self) -> dict[(str, str, str), int]:
        return {v: k for k, v in self.etype_idx_to_name}


zx_diagram = ZXDiagram(nx_clifford_graph(10, 10))

zx_hdata = ZXMatchDiagram(zx_diagram).to_pyg_heterograph()

print('zx_hdata = ', zx_hdata.metadata())
