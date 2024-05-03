import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.typing import NodeType

from alphazx.diagram.match import Match, CompoundMatch, BoundaryMatch, NODE_TYPE_TO_INDEX_METADATA, \
    EDGE_TYPE_TO_INDEX_METADATA, NODE_METADATA, EDGE_METADATA, FRightMatch
from alphazx.diagram.pyg_conv import compute_node_type_attr, compute_edge_size_attr, compute_edge_type_attr
from alphazx.diagram.zx_diagram import ZXDiagram, base_match_from_node


def add_type_masks(data: pyg.data.Data) -> None:
    data['node_mask_dict'] = {}
    for ntype in NODE_METADATA:
        data['node_mask_dict'][ntype] = data.x[:, 0].int().eq(NODE_TYPE_TO_INDEX_METADATA[ntype])
    data['edge_mask_dict'] = {}
    for etype in EDGE_METADATA:
        data['edge_mask_dict'][etype] = data.edge_attr[:, 0].int().eq(EDGE_TYPE_TO_INDEX_METADATA[etype])


def add_attr_dicts(hdata: pyg.data.HeteroData) -> None:
    for ntype in NODE_METADATA:
        if hdata[ntype].x.shape[0] == 0:
            del hdata[ntype]
        else:
            hdata[ntype].node_phase = hdata[ntype].x[:, 0]
    for etype in EDGE_METADATA:
        if hdata[etype].edge_index.shape[1] == 0:
            del hdata[etype]
        else:
            hdata[etype].edge_size = hdata[etype].edge_attr[:, 0]


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
                      node_type=compute_node_type_attr(boundary_match),
                      node_phase=compute_node_phase_attr(zx_diagram, boundary_match))
        for n in zx_diagram.nodes:
            if not zx_diagram.is_boundary(n):
                base_match = base_match_from_node(zx_diagram, n)
                self.add_node(base_match,
                              node_type=compute_node_type_attr(base_match),
                              node_phase=compute_node_phase_attr(zx_diagram, base_match))
        self.num_simple_nodes = self.number_of_nodes()
        for m, n in set(zx_diagram.edges()):
            match_m = base_match_from_node(zx_diagram, n)
            match_n = base_match_from_node(zx_diagram, m)
            self.add_edge(match_m,
                          match_n,
                          edge_type=compute_edge_type_attr(match_m, match_n),
                          edge_size=compute_edge_size_attr(zx_diagram.number_of_edges(match_m.node, match_n.node),
                                                           match_m,
                                                           match_n))
            self.add_edge(match_n,
                          match_m,
                          edge_type=compute_edge_type_attr(match_n, match_m),
                          edge_size=compute_edge_size_attr(zx_diagram.number_of_edges(match_n.node, match_m.node),
                                                           match_n,
                                                           match_m))

    def to_pyg_hdata(self, with_reverse_mapping: bool = False, sort_by_row: bool = False) -> pyg.data.HeteroData | \
                                                                                             tuple[
                                                                                                 pyg.data.HeteroData, 'HeteroDataIndexToMatch']:
        n_types = torch.tensor([NODE_TYPE_TO_INDEX_METADATA[ndata['node_type']] for _, ndata in self.nodes(data=True)],
                               dtype=torch.long)
        e_types = torch.tensor(
            [EDGE_TYPE_TO_INDEX_METADATA[edata['edge_type']] for _, _, edata in self.edges(data=True)],
            dtype=torch.long)
        hdata = pyg.utils.from_networkx(self,
                                        group_node_attrs=['node_phase'],
                                        group_edge_attrs=['edge_size']).to_heterogeneous(n_types,
                                                                                         e_types,
                                                                                         NODE_METADATA,
                                                                                         EDGE_METADATA).sort(
            sort_by_row)
        add_attr_dicts(hdata)
        hdata.validate()
        if with_reverse_mapping:
            return hdata, HeteroDataIndexToMatch(self)
        return hdata

    def to_pyg_data(self, with_reverse_mapping: bool = False, sort_by_row: bool = False) -> pyg.data.Data | tuple[
        pyg.data.Data, 'DataIndexToMatch']:
        data = self.to_pyg_hdata(with_reverse_mapping=False, sort_by_row=sort_by_row).to_homogeneous(
            node_attrs=['node_phase'], edge_attrs=['edge_size'], add_node_type=True, add_edge_type=True, dummy_values=False).sort(
            sort_by_row)
        data.x = torch.stack([data.node_type, data.node_phase], dim=-1)
        data.edge_attr = torch.stack([data.edge_type, data.edge_size], dim=-1)
        data.sort(sort_by_row)
        data.validate()
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


def add_match(zx_match_diagram: ZXMatchDiagram, zx_diagram: ZXDiagram, match: Match) -> None:
    zx_match_diagram.add_node(match,
                              node_type=compute_node_type_attr(match),
                              node_phase=compute_node_phase_attr(zx_diagram, match))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            add_match(zx_match_diagram, zx_diagram, sub_match)
            zx_match_diagram.add_edge(match,
                                      sub_match,
                                      edge_type=compute_edge_type_attr(match, sub_match),
                                      edge_size=compute_edge_size_attr(zx_diagram.number_of_edges(match, sub_match),
                                                                       match,
                                                                       sub_match))
            zx_match_diagram.add_edge(sub_match,
                                      match,
                                      edge_type=compute_edge_type_attr(sub_match, match),
                                      edge_size=compute_edge_size_attr(zx_diagram.number_of_edges(sub_match, match),
                                                                       sub_match, match))


def compute_node_phase_attr(zx_diagram: ZXDiagram, match: Match) -> torch.Tensor:
    if isinstance(match, FRightMatch):
        # Although the phase outputs of the GNN are categorical, we represent the input features as floats.
        return torch.tensor(zx_diagram.phase(match.nodes[0]), dtype=torch.float)
    else:
        return torch.tensor(0.)


class DataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        for i, match in enumerate(zx_match_diagram.nodes()):
            self.indices[i] = match

    def __getitem__(self, item: int):
        return self.indices[item]


class HeteroDataIndexToMatch:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.indices = dict()
        self.node_metadata = set(NODE_METADATA)
        for node_type in self.node_metadata:
            self.indices[node_type] = []
        for match, ndata in zx_match_diagram.nodes(data=True):
            self.indices[ndata['node_type']].append(match)

    def __getitem__(self, item: tuple[NodeType, int]):
        return self.indices[item[0]][item[1]]
