import networkx as nx
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from torch_geometric.typing import NodeType

from alphazx.diagram.constants import B_ETYPE_NAME, I_ETYPE_NAME
from alphazx.diagram.match import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch, BoundaryMatch, BaseMatch, \
    NODE_TYPE_TO_INDEX_METADATA, EDGE_TYPE_TO_INDEX_METADATA, NODE_METADATA, EDGE_METADATA
from alphazx.diagram.zx_diagram import ZXDiagram


# TODO: Decide on how to handle boundary vertices. Should they be explicitly represented, represented as two aggregate nodes
#       (input and output) to allow information to flow more effectively between boundary nodes, or should nodes simply have
#       an 'is_connected_to_boundary' feature?
class ZXMatchDiagram(nx.MultiDiGraph):

    def __init__(self, zx_diagram: ZXDiagram):
        self.zx_diagram = zx_diagram
        self.phase_denominator = self.zx_diagram.phase_denominator
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        super().__init__(nx.MultiDiGraph())
        for n in zx_diagram.nodes:
            base_match = base_match_from_node(zx_diagram, n)
            self.add_node(base_match, type=compute_node_type_attr(base_match),
                          phase=compute_node_phase_attr(zx_diagram, base_match))
        for m, n, k in zx_diagram.edges(data=False, keys=True):
            match_m = base_match_from_node(zx_diagram, n)
            match_n = base_match_from_node(zx_diagram, m)
            self.add_edge(match_m, match_n, k, type=compute_edge_type_attr(match_m, match_n))
            self.add_edge(match_n, match_m, k, type=compute_edge_type_attr(match_n, match_m))

    def to_pyg_hdata(self, with_reverse_mapping: bool = False) -> pyg_data.HeteroData | tuple[
            pyg_data.HeteroData, 'HeteroDataIndexToMatch']:
        n_types = torch.tensor([NODE_TYPE_TO_INDEX_METADATA[ndata['type']] for _, ndata in self.nodes(data=True)],
                               dtype=torch.long)
        e_types = torch.tensor([EDGE_TYPE_TO_INDEX_METADATA[edata['type']] for _, _, edata in self.edges(data=True)],
                               dtype=torch.long)
        hdata = pyg_utils.from_networkx(self, group_node_attrs=['phase']).to_heterogeneous(n_types,
                                                                                           e_types,
                                                                                           node_type_names=NODE_METADATA,
                                                                                           edge_type_names=EDGE_METADATA)
        if with_reverse_mapping:
            return hdata, HeteroDataIndexToMatch(self)
        return hdata

    def to_pyg_data(self, with_reverse_mapping: bool = False) -> pyg_data.Data | tuple[
            pyg_data.Data, 'DataIndexToMatch']:
        for _, ndata in self.nodes(data=True):
            ndata['type'] = NODE_TYPE_TO_INDEX_METADATA[ndata['type']]
        for _, _, edata in self.edges(data=True):
            edata['type'] = EDGE_TYPE_TO_INDEX_METADATA[edata['type']]
        data = pyg_utils.from_networkx(self, group_node_attrs=['type', 'phase'], group_edge_attrs=['type'])
        if with_reverse_mapping:
            return data, DataIndexToMatch(self)
        return data


def to_zx_match_diagram(zx_diagram: ZXDiagram) -> ZXMatchDiagram:
    zx_match_diagram = ZXMatchDiagram(zx_diagram)
    matches = list(zx_diagram.compute_matches())
    for match in matches:
        add_match(zx_match_diagram, zx_diagram, match)
    num_nodes = zx_match_diagram.number_of_nodes()
    num_matches = len(matches) + zx_diagram.num_b_nodes()
    assert num_nodes == num_matches, f'Number of nodes {num_nodes} in match diagram != number of matches {num_matches}'
    return zx_match_diagram


def compute_edge_type_attr(m: Match, n: Match) -> tuple[str, str, str]:
    return m.abbrev, B_ETYPE_NAME if isinstance(m, BaseMatch) and isinstance(n, BaseMatch) else I_ETYPE_NAME, n.abbrev


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
    if not zx_match_diagram.has_node(match):
        zx_match_diagram.add_node(match,
                                  type=compute_node_type_attr(match),
                                  phase=compute_node_phase_attr(zx_diagram, match))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not zx_match_diagram.has_node(sub_match):
                add_match(zx_match_diagram, zx_diagram, sub_match)
            if not zx_match_diagram.has_edge(sub_match, match) and not zx_match_diagram.has_edge(match, sub_match):
                zx_match_diagram.add_edge(match, sub_match, type=compute_edge_type_attr(match, sub_match))
                zx_match_diagram.add_edge(sub_match, match, type=compute_edge_type_attr(sub_match, match))


def base_match_from_node(diagram: ZXDiagram, node: int) -> BaseMatch:
    if diagram.is_z_basis(node):
        return FRightZMatch(node)
    elif diagram.is_x_basis(node):
        return FRightXMatch(node)
    elif diagram.is_boundary(node):
        return BoundaryMatch(node)
    else:
        raise Exception(f'Unexpected node type {diagram.type(node)}')


class DataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        for i, match in enumerate(zx_match_diagram.nodes(data=False)):
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
