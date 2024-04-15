import networkx as nx
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from torch_geometric.typing import NodeType

from alphazx.diagram.constants import B_ETYPE_NAME, I_ETYPE_NAME
from alphazx.diagram.match import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch, BoundaryMatch, \
    BaseMatch, NODE_TYPE_TO_INDEX_METADATA, EDGE_TYPE_TO_INDEX_METADATA, NODE_METADATA, EDGE_METADATA
from alphazx.diagram.zx_diagram import ZXDiagram


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
        super().__init__(nx.DiGraph())
        for n in self.zx_diagram.nodes():
            assert n >= 0, f'Node {n} is not non-negative'
        boundary_match = BoundaryMatch(-1)
        self.add_node(boundary_match,
                      type=compute_node_type_attr(boundary_match),
                      phase=compute_node_phase_attr(zx_diagram, boundary_match))
        for n in zx_diagram.nodes:
            if not zx_diagram.is_boundary(n):
                base_match = base_match_from_node(zx_diagram, n)
                self.add_node(base_match,
                              type=compute_node_type_attr(base_match),
                              phase=compute_node_phase_attr(zx_diagram, base_match))
        for m, n in set(zx_diagram.edges()):
            match_m = base_match_from_node(zx_diagram, n)
            match_n = base_match_from_node(zx_diagram, m)
            self.add_edge(match_m,
                          match_n,
                          type=compute_edge_type_attr(match_m, match_n),
                          size=compute_edge_size_attr(zx_diagram, match_m, match_n))
            self.add_edge(match_n,
                          match_m,
                          type=compute_edge_type_attr(match_n, match_m),
                          size=compute_edge_size_attr(zx_diagram, match_n, match_m))

    def to_pyg_hdata(self, with_reverse_mapping: bool = False) -> pyg_data.HeteroData | tuple[
        pyg_data.HeteroData, 'HeteroDataIndexToMatch']:
        n_types = torch.tensor([NODE_TYPE_TO_INDEX_METADATA[ndata['type']] for _, ndata in self.nodes(data=True)],
                               dtype=torch.long)
        e_types = torch.tensor([EDGE_TYPE_TO_INDEX_METADATA[edata['type']] for _, _, edata in self.edges(data=True)],
                               dtype=torch.long)
        hdata = pyg_utils.from_networkx(self, group_node_attrs=['phase'], group_edge_attrs=['size']).to_heterogeneous(
            n_types, e_types, node_type_names=NODE_METADATA, edge_type_names=EDGE_METADATA)
        if with_reverse_mapping:
            return hdata, HeteroDataIndexToMatch(self)
        return hdata

    def to_pyg_data(self, with_reverse_mapping: bool = False) -> pyg_data.Data | tuple[
        pyg_data.Data, 'DataIndexToMatch']:
        for _, ndata in self.nodes(data=True):
            ndata['type'] = NODE_TYPE_TO_INDEX_METADATA[ndata['type']]
        for _, _, edata in self.edges(data=True):
            edata['type'] = EDGE_TYPE_TO_INDEX_METADATA[edata['type']]
        data = pyg_utils.from_networkx(self, group_node_attrs=['type', 'phase'], group_edge_attrs=['type', 'size'])
        if with_reverse_mapping:
            return data, DataIndexToMatch(self)
        return data


def to_zx_match_diagram(zx_diagram: ZXDiagram) -> ZXMatchDiagram:
    zx_match_diagram = ZXMatchDiagram(zx_diagram)
    matches = set(zx_diagram.compute_matches())
    for match in matches:
        for sub_match in match.sub_matches:
            assert sub_match in matches, f'Submatch {sub_match} of match {match} not in input diagram'
    for match in matches:
        assert not isinstance(match, BoundaryMatch), 'Boundary match in input diagram'
    for match in matches:
        add_match(zx_match_diagram, zx_diagram, match)
    for match in matches:
        assert zx_match_diagram.has_node(match), f'Node {match} not in match diagram'
    actual_num_nodes = zx_match_diagram.number_of_nodes()
    expected_num_nodes = len(matches) + 1
    assert actual_num_nodes == expected_num_nodes, f'Actual number of nodes {actual_num_nodes} != {expected_num_nodes}'
    return zx_match_diagram


def compute_edge_type_attr(m: Match, n: Match) -> tuple[str, str, str]:
    return m.abbrev, B_ETYPE_NAME if isinstance(m, BaseMatch) and isinstance(n, BaseMatch) else I_ETYPE_NAME, n.abbrev


def compute_edge_size_attr(zx_diagram: ZXDiagram, m: Match, n: Match) -> int:
    return zx_diagram.number_of_edges(m.node, n.node) if isinstance(m, FRightMatch) and isinstance(n,
                                                                                                   FRightMatch) else 1


def compute_node_type_attr(match: Match) -> str:
    return match.abbrev


def compute_node_phase_attr(zx_diagram: ZXDiagram, match: Match) -> torch.Tensor:
    if isinstance(match, FRightMatch):
        # Although the phase outputs of the DNN are categorical, we represent the input features as floats.
        return torch.tensor(zx_diagram.phase(match.nodes[0]), dtype=torch.float)
    else:
        # Because phases of f-right matches are always mod 2, giving non-f-right matches a phase of -1 differentiates
        # them from f-right matches.
        # TODO: This ^ is not correct.
        return torch.tensor(-1.)


def add_match(zx_match_diagram: ZXMatchDiagram, zx_diagram: ZXDiagram, match: Match) -> None:
    zx_match_diagram.add_node(match,
                              type=compute_node_type_attr(match),
                              phase=compute_node_phase_attr(zx_diagram, match))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            add_match(zx_match_diagram, zx_diagram, sub_match)
            zx_match_diagram.add_edge(match,
                                      sub_match,
                                      type=compute_edge_type_attr(match, sub_match),
                                      size=compute_edge_size_attr(zx_diagram, match, sub_match))
            zx_match_diagram.add_edge(sub_match,
                                      match,
                                      type=compute_edge_type_attr(sub_match, match),
                                      size=compute_edge_size_attr(zx_diagram, sub_match, match))


def base_match_from_node(diagram: ZXDiagram, node: int) -> BaseMatch:
    if diagram.is_z_basis(node):
        return FRightZMatch(node)
    elif diagram.is_x_basis(node):
        return FRightXMatch(node)
    elif diagram.is_boundary(node):
        return BoundaryMatch(-1)
    else:
        raise Exception(f'Unexpected node type {diagram.type(node)}')


class DataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        for i, match in enumerate(zx_match_diagram.nodes()):
            self.indices[i] = match

    def __getitem__(self, item: tuple[NodeType, int]):
        return self.indices[item[1]]


class HeteroDataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        self.node_metadata = set(NODE_METADATA)
        for node_type in self.node_metadata:
            self.indices[node_type] = []
        for match, ndata in zx_match_diagram.nodes(data=True):
            self.indices[ndata['type']].append(match)

    def __getitem__(self, item: tuple[NodeType, int]):
        return self.indices[item[0]][item[1]]
