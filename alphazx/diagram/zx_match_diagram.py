import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.typing import NodeType

from alphazx.diagram.match import MatchNode, CompoundMatchNode, FRightMatch, from_index_and_node_set, \
    METADATA, ZXMatchDiagramNode, SuperNode, compute_meta_edge_type, SimpleMatchNode
from alphazx.diagram.zx_diagram import ZXDiagram, base_match_from_node
from alphazx.models import compute_basis_neighbors


def add_attr_dicts(hdata: pyg.data.HeteroData) -> None:
    for ntype in METADATA.node_type_abbrevs:
        if hdata[ntype].x.shape[0] == 0:
            del hdata[ntype]
        else:
            hdata[ntype].node_phase = hdata[ntype].x[:, 0].to(torch.float32)
    for etype in METADATA.edge_types:
        if hdata[etype].edge_index.shape[1] == 0:
            del hdata[etype]
        else:
            hdata[etype].edge_size = hdata[etype].edge_attr[:, 0].to(torch.float32)


def compute_edge_type_attr(m: ZXMatchDiagramNode, n: ZXMatchDiagramNode) -> tuple[str, str, str]:
    return m.abbrev, compute_meta_edge_type(m.__class__, n.__class__), n.abbrev


def compute_edge_size_attr(zx_diagram: ZXDiagram, a: ZXMatchDiagramNode, b: ZXMatchDiagramNode) -> torch.Tensor:
    return torch.tensor(
        float(zx_diagram.number_of_edges(a.node, b.node)) if isinstance(a, SimpleMatchNode) and isinstance(b,
                                                                                                           SimpleMatchNode) else 1.)


def compute_node_type_attr(match: ZXMatchDiagramNode) -> str:
    return match.abbrev


class ZXMatchDiagram(nx.DiGraph):
    """
    A directed graph representing a ZX-diagram as a collection of matches. The nodes of the graph are matches, and the
    edges are either edges from the ZX-diagram or edges between matches indicating a sub-match relationship. All boundary
    nodes are represented as a single node. All edges between boundary nodes and Z/X nodes in the input ZX-diagram
    (e.g. b0 -- z0 and b1 -- z1) are replaced with edges between the single boundary node and the corresponding match node
    (e.g. b -- z0 and b -- z1).
    """

    def __init__(self, zx_diagram: ZXDiagram):
        self.zx_diagram = zx_diagram
        self.phase_denominator = self.zx_diagram.phase_denominator
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        self.super_nodes = set()
        super().__init__(nx.DiGraph())
        for match_node_type_abbrev in METADATA.match_node_type_abbrevs:
            setattr(self, f'{match_node_type_abbrev}_nodes', set())
        for n in self.zx_diagram.nodes():
            assert n >= 0, f'Node {n} is not non-negative'
        for n in zx_diagram.nodes:
            self.add_node(base_match_from_node(zx_diagram, n))
        for m, n in set(zx_diagram.edges()):
            match_m = base_match_from_node(zx_diagram, n)
            match_n = base_match_from_node(zx_diagram, m)
            self.add_edge(match_m, match_n)

    def add_node(self, match: MatchNode, **attr: dict):
        super().add_node(match,
                         node_set=compute_node_set_attr(match),
                         node_type=compute_node_type_attr(match),
                         node_phase=compute_node_phase_attr(self.zx_diagram, match), **attr)
        getattr(self, f'{match.abbrev}_nodes').add(match)
        super_node = match.super_node()()
        super().add_node(super_node,
                         node_set=compute_node_set_attr(super_node),
                         node_type=compute_node_type_attr(super_node),
                         node_phase=compute_node_phase_attr(self.zx_diagram, super_node))
        self.super_nodes.add(super_node)
        for existing_super_node in self.super_nodes:
            if existing_super_node.abbrev != super_node.abbrev:
                self.add_edge(super_node, existing_super_node)
        self.add_edge(match, super_node)

    def add_edge(self, a: ZXMatchDiagramNode, b: ZXMatchDiagramNode, **attr: dict):
        super().add_edge(a, b, edge_type=compute_edge_type_attr(a, b),
                         edge_size=compute_edge_size_attr(self.zx_diagram, a, b), **attr)
        super().add_edge(b, a, edge_type=compute_edge_type_attr(b, a),
                         edge_size=compute_edge_size_attr(self.zx_diagram, b, a), **attr)

    def to_pyg_hdata(self, with_reverse_mapping: bool = False, sort_by_row: bool = False) -> pyg.data.HeteroData | \
                                                                                             tuple[
                                                                                                 pyg.data.HeteroData, 'HeteroDataIndexToMatch']:
        n_types = torch.tensor(
            [METADATA.node_type_abbrev_index_dict[ndata['node_type']] for _, ndata in self.nodes(data=True)],
            dtype=torch.long)
        e_types = torch.tensor(
            [METADATA.edge_type_to_index_dict[edata['edge_type']] for _, _, edata in self.edges(data=True)],
            dtype=torch.long)
        hdata = pyg.utils.from_networkx(self,
                                        group_node_attrs=['node_phase'],
                                        group_edge_attrs=['edge_size']).to_heterogeneous(n_types,
                                                                                         e_types,
                                                                                         METADATA.node_type_abbrevs,
                                                                                         METADATA.edge_types).sort(
            sort_by_row)
        add_attr_dicts(hdata)
        hdata.validate()
        if with_reverse_mapping:
            return hdata, HeteroDataIndexToMatch(self)
        return hdata

    def to_pyg_data(self, with_reverse_mapping: bool = False, sort_by_row: bool = False) -> pyg.data.Data | tuple[
        pyg.data.Data, 'DataIndexToMatch']:
        # Temporarily convert string attributes to tensors in-place, then restore.
        # This avoids copy.deepcopy of the entire NetworkX DiGraph, which was the
        # single most expensive operation in the game-play hot path.
        saved_node_types = {}
        for n, ndata in self.nodes(data=True):
            saved_node_types[id(ndata)] = ndata['node_type']
            ndata['node_type'] = torch.tensor(METADATA.node_type_abbrev_index_dict[ndata['node_type']])

        saved_edge_types = {}
        for u, v, edata in self.edges(data=True):
            saved_edge_types[id(edata)] = edata['edge_type']
            edata['edge_type'] = torch.tensor(METADATA.edge_type_to_index_dict[edata['edge_type']])

        try:
            data = pyg.utils.from_networkx(self,
                                           group_node_attrs=['node_type', 'node_phase', 'node_set'],
                                           group_edge_attrs=['edge_type', 'edge_size'])
        finally:
            # Restore original string attributes so the match diagram stays clean
            for n, ndata in self.nodes(data=True):
                ndata['node_type'] = saved_node_types[id(ndata)]
            for u, v, edata in self.edges(data=True):
                edata['edge_type'] = saved_edge_types[id(edata)]

        data.id = torch.tensor(self.zx_diagram.id, dtype=torch.float32)
        data.node_type = data.x[:, 0].to(dtype=torch.long)
        data.node_phase = data.x[:, 1].to(dtype=torch.float32)
        data.node_set = data.x[:, 2:].to(dtype=torch.long)
        # TODO: For some reason 'data.sort' does not work...
        data.edge_index, data.edge_attr = pyg.utils.sort_edge_index(data.edge_index, data.edge_attr,
                                                                    sort_by_row=sort_by_row)
        data.edge_attr = data.edge_attr.to(dtype=torch.long)
        data.edge_type = data.edge_attr[:, 0]
        data.edge_size = data.edge_attr[:, 1]
        data.validate()
        if with_reverse_mapping:
            return data, DataIndexToMatch(data)
        return data


def check_super_nodes_exist(zx_match_diagram: ZXMatchDiagram) -> None:
    for n, ndata in zx_match_diagram.nodes(data=True):
        if isinstance(n, MatchNode):
            super_node = n.super_node()()
            assert super_node in zx_match_diagram, f'Node {n} does not have a corresponding super node {super_node.abbrev}'
            assert zx_match_diagram.has_edge(n, super_node), f'Expected edge between {n.abbrev} and {super_node.abbrev}'
            assert zx_match_diagram.has_edge(super_node, n), f'Expected edge between {super_node.abbrev} and {n.abbrev}'


def check_super_node_counts(zx_match_diagram: ZXMatchDiagram) -> None:
    super_nodes = {}
    for n, ndata in zx_match_diagram.nodes(data=True):
        if isinstance(n, SuperNode):
            super_nodes[n.abbrev] = super_nodes.get(n.abbrev, 0) + 1
    for k, v in super_nodes.items():
        assert v == 1, f'Match diagram contains {v} super nodes {k}'


def check_super_node_edges(zx_match_diagram: ZXMatchDiagram) -> None:
    for super_node_a in zx_match_diagram.super_nodes:
        for super_node_b in zx_match_diagram.super_nodes:
            if super_node_a.abbrev != super_node_b.abbrev:
                assert zx_match_diagram.has_edge(super_node_a,
                                                 super_node_b), f'Expected edge between {super_node_a.abbrev} and {super_node_b.abbrev}'
                assert zx_match_diagram.has_edge(super_node_b,
                                                 super_node_a), f'Expected edge between {super_node_b.abbrev} and {super_node_a.abbrev}'


def check_opposite_edges(zx_match_diagram: ZXMatchDiagram) -> None:
    for a, b in zx_match_diagram.edges:
        assert zx_match_diagram.has_edge(b, a), f'Edge from {a} to {b} has no opposite edge'


def check_basis_node_counts(zx_diagram: ZXDiagram, zx_match_diagram: ZXMatchDiagram) -> None:
    actual_num_b_nodes = len(zx_match_diagram.b_nodes)
    expected_num_b_nodes = zx_diagram.num_b_nodes()
    actual_num_z_nodes = len(zx_match_diagram.frz_nodes)
    expected_num_z_nodes = zx_diagram.num_z_nodes()
    actual_num_x_nodes = len(zx_match_diagram.frx_nodes)
    expected_num_x_nodes = zx_diagram.num_x_nodes()
    assert actual_num_b_nodes == expected_num_b_nodes, f'Actual number of boundary nodes {actual_num_b_nodes} != expected number of boundary nodes {expected_num_b_nodes}'
    assert actual_num_z_nodes == expected_num_z_nodes, f'Actual number of Z nodes {actual_num_z_nodes} != expected number of Z nodes {expected_num_z_nodes}'
    assert actual_num_x_nodes == expected_num_x_nodes, f'Actual number of Z nodes {actual_num_x_nodes} != expected number of Z nodes {expected_num_x_nodes}'


def to_zx_match_diagram(zx_diagram: ZXDiagram) -> ZXMatchDiagram:
    zx_match_diagram = ZXMatchDiagram(zx_diagram)
    matches = set(zx_diagram.compute_matches())
    # for match in matches:
    #     for sub_match in match.sub_matches:
    #         assert sub_match in matches, f'Submatch {sub_match} of match {match} not in input diagram'
    for match in matches:
        # assert not isinstance(match, BoundaryMatch), 'Boundary match in input diagram'
        add_match(zx_match_diagram, zx_diagram, match)
    # for match in matches:
    #     assert match in zx_match_diagram, f'Node {match} not in match diagram'
    #     check_super_nodes_exist(zx_match_diagram)
    #     check_super_node_counts(zx_match_diagram)
    #     check_super_node_edges(zx_match_diagram)
    # check_basis_node_counts(zx_diagram, zx_match_diagram)
    # check_opposite_edges(zx_match_diagram)
    return zx_match_diagram


def add_match(zx_match_diagram: ZXMatchDiagram, zx_diagram: ZXDiagram, match: MatchNode) -> None:
    zx_match_diagram.add_node(match)
    if isinstance(match, CompoundMatchNode):
        for sub_match in match.sub_matches:
            add_match(zx_match_diagram, zx_diagram, sub_match)
            zx_match_diagram.add_edge(match, sub_match)


def compute_node_set_attr(node: ZXMatchDiagramNode) -> torch.Tensor:
    nodes = torch.tensor([len(node.nodes)] + node.nodes, dtype=torch.long) if isinstance(node,
                                                                                         MatchNode) else torch.empty(0,
                                                                                                                     dtype=torch.long)
    pad = (0, METADATA.max_match_size - len(node.nodes) if isinstance(node, MatchNode) else METADATA.max_match_size + 1)
    return torch.nn.functional.pad(nodes, pad, mode='constant', value=0)


def compute_node_phase_attr(zx_diagram: ZXDiagram, match: ZXMatchDiagramNode) -> torch.Tensor:
    if isinstance(match, FRightMatch):
        # Although the phase outputs of the GNN are categorical, we represent the input features as floats.
        return torch.tensor(zx_diagram.phase(match.nodes[0]), dtype=torch.float)
    else:
        return torch.tensor(0.)


class DataIndexToMatch:
    def __init__(self, data: pyg.data.Data):
        self.indices = dict()
        for i, node_set in enumerate(data.node_set):
            node_set = node_set.tolist()
            node_set = node_set[1:node_set[0] + 1]
            self.indices[i] = from_index_and_node_set(data.node_type[i].item(), node_set)

    def __getitem__(self, item: int):
        return self.indices[item]


class HeteroDataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        self.node_metadata = set(METADATA.node_type_abbrevs)
        for node_type in self.node_metadata:
            self.indices[node_type] = []
        for match, ndata in zx_match_diagram.nodes(data=True):
            self.indices[ndata['node_type']].append(match)

    def __getitem__(self, item: tuple[NodeType, int]):
        return self.indices[item[0]][item[1]]


def compute_new_phase(cat_phase: torch.Tensor, phase_denominator: int) -> float:
    """
    Converts a tensor phase representation back into a float value based on the unit circle position,
    ensuring compatibility with the wrap-around behavior of the U(1) group.

    :param cat_phase: A scalar tensor representing the discrete position on the unit circle.
    :param phase_denominator: The number of discrete positions (categories) on the unit circle.
    :return: The float value representing the position on the unit circle.
    """
    if phase_denominator <= 0:
        raise ValueError(f"The phase denominator {phase_denominator} is not positive.")
    # Ensure position wraps around using modulus to handle negative and overflow positions
    normalized_position = int(cat_phase) % phase_denominator
    return normalized_position / phase_denominator


def compute_num_new_edges(cat_new_edges: int) -> int:
    # The number of new edges is the category value plus one, since the new edge distribution is a categorical distribution
    # with categories 0, 1, 2, ..., n, where n + 1 is the max number of possible new edges.
    return cat_new_edges + 1


def compute_transfer_edges(node: int,
                           bernoulli_transfer_edges: tuple,
                           data: pyg.data.Data,
                           data_index: DataIndexToMatch) -> set[int]:
    basis_neighbors = compute_basis_neighbors(data.edge_index, node, data.node_type)
    basis_neighbors = basis_neighbors[torch.tensor(bernoulli_transfer_edges[:len(basis_neighbors)], dtype=torch.bool)]
    transfer_edges = []
    for neighbor in basis_neighbors.tolist():
        neighbor_match = data_index[neighbor]
        transfer_edges.append(neighbor_match.node)
    return set(transfer_edges)


def compute_f_right_params(action: tuple, data: pyg.data.Data, data_index: DataIndexToMatch,
                           zx_match_diagram: ZXMatchDiagram) -> tuple[float, int, set[int]]:
    phase = compute_new_phase(action[3], zx_match_diagram.phase_denominator)
    new_edges = compute_num_new_edges(action[4])
    transfer_edges = compute_transfer_edges(action[2], action[5:], data, data_index)
    return phase, new_edges, transfer_edges
