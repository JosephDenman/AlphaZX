import networkx as nx
import torch
import torch.nn.functional as torch_func
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

from diagram.constants import B_ETYPE_INDEX, I_ETYPE_INDEX, I_ETYPE_ONE_HOT, B_ETYPE_ONE_HOT
from diagram.match import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch, MATCH_TYPE_COUNT
from diagram.zx_diagram import ZXDiagram


# TODO: In the future, post-processed versions of ZXMatchDiagram, e.g., adding certain features should be done by
#       defining methods that return new 'ZXMatchDiagram' instances with the desired properties. This way, post-processing
#       steps can be easily chained.
# TODO: Decide on how to handle boundary vertices. Should they be explicitly represented, represented as two aggregate nodes
#       (input and output) to allow information to flow more effectively between boundary nodes, or should nodes simply have
#       an 'is_connected_to_boundary' feature?
class ZXMatchDiagram(nx.MultiGraph):
    NTYPE = 'type'
    ETYPE = 'type'

    def __init__(self, zx_diagram: ZXDiagram, one_hot_types: bool):
        self.zx_diagram = zx_diagram
        self.one_hot_types = one_hot_types
        self.phase_denominator = self.zx_diagram.phase_denominator
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        self.max_degree = max(self.zx_diagram.degree, key=lambda x: x[1])[1]
        super().__init__(nx.MultiGraph())
        for n, ndata in zx_diagram.nodes(data=True):
            if zx_diagram.is_basis(n):
                f_right_match_node = f_right_match_from_node(zx_diagram, n)
                self.add_node(f_right_match_node, type=compute_node_type_attr(f_right_match_node, one_hot_types),
                              phase=compute_node_phase_attr(zx_diagram, f_right_match_node))
        for n, m, edata in zx_diagram.edges(data=True):
            if zx_diagram.is_basis(n) and zx_diagram.is_basis(m):
                self.add_edge(f_right_match_from_node(zx_diagram, n), f_right_match_from_node(zx_diagram, m),
                              type=B_ETYPE_ONE_HOT if one_hot_types else B_ETYPE_INDEX)

    def to_pyg_hetero_data(self) -> pyg_data.HeteroData:
        pass

    @staticmethod
    def __flatten_and_concatenate_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
        # TODO: Ensure incident edges attribute is properly concatenated
        # Flatten each tensor in the list to ensure it's 1D
        flattened_tensors = [torch.flatten(tensor) for tensor in tensors]
        # Concatenate all the flattened tensors into a single 1D tensor
        concatenated_tensor = torch.cat(flattened_tensors)
        return concatenated_tensor

    def to_pyg_data(self) -> pyg_data.Data:
        # Maps nodes to their positions in the node features tensor.
        node_position = dict()
        # Maps positions to their nodes in the original graph.
        position_node = dict()
        # Node features
        node_features_list = []
        for i, (node, node_data) in enumerate(self.nodes(data=True)):
            # Flatten, concatenate, and append node features
            node_features_list.append(
                self.__flatten_and_concatenate_tensors([node_data[attr] for attr in self.node_attrs]))
            node_position[node] = i
        node_features_tensor = torch.stack(node_features_list)
        # Edge indices and edge attributes
        edge_sources = []
        edge_targets = []
        indexed_edge_types = dict()
        for source, target, edge_data in self.edges(data=True):
            source_index = node_position[source]
            target_index = node_position[target]
            edge_sources.append(source_index)
            edge_targets.append(target_index)
            edge_type = torch.flatten(
                edge_data[self.ETYPE] if self.one_hot_types else torch.tensor([edge_data[self.ETYPE]],
                                                                              dtype=torch.int64))
            indexed_edge_types[(source_index, target_index)] = edge_type
        edge_types = []
        for source_index, target_index in zip(edge_sources, edge_targets):
            edge_types.append(indexed_edge_types[(source_index, target_index)])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.stack(edge_types)
        edge_index, edge_attr = pyg_utils.to_undirected(edge_index, edge_attr, num_nodes=node_features_tensor.size(0))
        data = pyg_data.Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr)
        assert data.is_undirected(), 'PyG graph is not undirected'
        data.sort()
        data.coalesce()
        data.validate()
        return data


def to_zx_match_diagram(zx_diagram: ZXDiagram, one_hot_types: bool) -> ZXMatchDiagram:
    zx_match_diagram = ZXMatchDiagram(zx_diagram, one_hot_types)
    matches = list(zx_diagram.compute_matches())
    for match in matches:
        add_match(zx_match_diagram, zx_diagram, match, one_hot_types)
    num_nodes = zx_match_diagram.number_of_nodes()
    num_matches = len(matches)
    assert num_nodes == num_matches, f'Number of nodes {num_nodes} in match diagram != number of matches ' \
                                     f'{num_matches}'
    assert zx_diagram.num_basis_nodes() == len([m for m in zx_match_diagram.nodes() if isinstance(m, FRightMatch)]), \
        'Each basis node in the base diagram should have a corresponding node in the match diagram'
    return zx_match_diagram


def compute_node_type_attr(match: Match, one_hot_types: bool) -> torch.Tensor:
    return torch_func.one_hot(torch.tensor(match.index), MATCH_TYPE_COUNT) if one_hot_types else match.index


def compute_node_phase_attr(zx_diagram: ZXDiagram, match: Match) -> torch.Tensor:
    if isinstance(match, FRightMatch):
        # Although the phase outputs of the DNN are categorical, we represent the input features as floats.
        return torch.tensor([zx_diagram.phase(match.nodes[0])], dtype=torch.float)
    else:
        # Because phases of f-right matches are always mod 2, giving non-f-right matches a phase of -1 differentiates
        # them from f-right matches.
        return torch.tensor([-1])


def add_match(zx_match_diagram: ZXMatchDiagram, zx_diagram: ZXDiagram, match: Match, one_hot_types: bool) -> None:
    if not zx_match_diagram.has_node(match):
        zx_match_diagram.add_node(match,
                                  type=compute_node_type_attr(match, one_hot_types),
                                  phase=compute_node_phase_attr(zx_diagram, match))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not zx_match_diagram.has_node(sub_match):
                add_match(zx_match_diagram, zx_diagram, sub_match, one_hot_types)
            if not zx_match_diagram.has_edge(sub_match, match):
                zx_match_diagram.add_edge(match, sub_match, type=I_ETYPE_ONE_HOT if one_hot_types else I_ETYPE_INDEX)


def f_right_match_from_node(diagram: ZXDiagram, node: int) -> FRightMatch:
    if diagram.is_z_basis(node):
        return FRightZMatch(node)
    elif diagram.is_x_basis(node):
        return FRightXMatch(node)
    else:
        raise Exception(f'Unexpected node type {diagram.type(node)}')
