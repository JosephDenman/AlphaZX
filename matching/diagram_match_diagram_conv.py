from collections import defaultdict
from typing import Any, Iterator
import torch_geometric as pyg

import torch

from graph.pyzx_nx_conv import ETYPE, NTYPE, is_basis, is_z_basis, is_x_basis
from matching.b_rule_matcher import b_left_matches, b_right_matches
from matching.f_rule_matcher import f_right_z_matches, f_right_x_matches, f_left_z_matches, f_left_x_matches
from matching.match_types import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch
from matching.y_rule_matcher import y_left_z_matches, y_left_x_matches, y_right_z_matches, y_right_x_matches
from matching.zx_diagram import ZXDiagram
from matching.zx_match_diagram import ZXMatchDiagram


# TODO: Eventually pass list of matchers
def compute_matches(diagram: ZXDiagram) -> Iterator[Match]:
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


def add_match(match_diagram: ZXMatchDiagram, diagram: ZXDiagram, match: Match) -> None:
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


def basis_nodes(diagram: ZXDiagram) -> Iterator[int]:
    yield from (n for n, ndata in diagram.nodes(data=True) if is_basis(ndata[NTYPE]))


def basis_neighbors(diagram: ZXDiagram, n: int) -> Iterator[int]:
    yield from (m for m in diagram.neighbors(n) if is_basis(diagram.nodes[m][NTYPE]))


def has_i_edge(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_inclusion_edge(edata[ETYPE])
    return False


def inclusion_neighbors(match_diagram: ZXMatchDiagram, u_match: Match) -> Iterator[Match]:
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


def has_b_edge(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_bridge_edge(edata[ETYPE])
    return False


def is_match_neighbor(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    for u_neighbor in inclusion_neighbors(match_diagram, u_match):
        if v_match in inclusion_neighbors(match_diagram, u_neighbor):
            return True
    return False


def connected(match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    return is_match_neighbor(match_diagram, u_match, v_match) or has_b_edge(match_diagram, u_match, v_match)


def add_composition_edges(match_diagram: ZXMatchDiagram, diagram: ZXDiagram) -> None:
    for u in basis_nodes(diagram):
        for v in basis_neighbors(diagram, u):
            u_match = f_right_match_from_ndata(u, diagram.nodes[u])
            v_match = f_right_match_from_ndata(v, diagram.nodes[v])
            if not connected(match_diagram, u_match, v_match):
                match_diagram.add_edge(u_match, v_match, type=B_ETYPE_INDEX)


# TODO: Eventually pass list of matchers.
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


def node_type_name(match_diagram: ZXMatchDiagram, data: dict[str, Any] | Match) -> str:
    if isinstance(data, Match):
        return match_diagram.ntype_idx_to_name[match_diagram.nodes[data][NTYPE]]
    else:
        return match_diagram.ntype_idx_to_name[data[NTYPE]]


def edge_type_name(match_diagram: ZXMatchDiagram, s: Match, t: Match, edata: dict[str, Any]) -> tuple[str, str, str]:
    e_type = I_ETYPE_NAME if is_inclusion_edge(edata[ETYPE]) else B_ETYPE_NAME
    return match_diagram.ntype_idx_to_name[match_diagram.nodes[s][NTYPE]], e_type, match_diagram.ntype_idx_to_name[
        match_diagram.nodes[t][NTYPE]]


def add_edge_to_e_hdata(match_diagram: ZXMatchDiagram, s: Match, t: Match, n_order_dict: dict[Any, dict],
                        edata: dict[str, Any], e_hdata):
    sources, targets = e_hdata[edge_type_name(match_diagram, s, t, edata)]['edge_index']
    sources.append(n_order_dict[node_type_name(match_diagram, s)][s])
    targets.append(n_order_dict[node_type_name(match_diagram, t)][t])


def to_pyg_heterograph(match_diagram: ZXMatchDiagram):
    assert not match_diagram.zx_diagram.is_directed(), "Graph must be undirected"
    n_hdata = defaultdict(lambda: defaultdict(list))
    n_order_list = defaultdict(list)
    for n, ndata in match_diagram.nodes(data=True):
        ntype = node_type_name(match_diagram, ndata)
        n_hdata[ntype]['x'].append(ndata_to_feature(ndata))
        n_order_list[ntype].append(n)
    n_order_dict = {}
    for ntype in n_hdata:
        n_hdata[ntype]['x'] = torch.stack(n_hdata[ntype]['x'])
        n_order_dict[ntype] = {node: i for i, node in enumerate(n_order_list[ntype])}
    added_edges = set()
    e_hdata = defaultdict(lambda: defaultdict(lambda: [[], []]))
    for source, target, edata in match_diagram.edges(data=True):
        if not (source, target) in added_edges and not (target, source) in added_edges:
            add_edge_to_e_hdata(match_diagram, source, target, n_order_dict, edata, e_hdata)
            add_edge_to_e_hdata(match_diagram, target, source, n_order_dict, edata, e_hdata)
            added_edges.add((source, target))
            added_edges.add((target, source))
    for etype in e_hdata:
        e_hdata[etype]['edge_index'] = torch.tensor(e_hdata[etype]['edge_index'], dtype=torch.long)
    n_hdata.update(e_hdata)
    return pyg.data.HeteroData(n_hdata)