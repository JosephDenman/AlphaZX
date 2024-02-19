from typing import Iterator

import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData

from diagram.match import Match, CompoundMatch, FRightMatch, FRightZMatch, FRightXMatch, MATCH_TYPE_COUNT
from diagram.pyzx_nx_conv import ETYPE
from diagram.zx_diagram import ZXDiagram

ETYPE_COUNT = 2
I_ETYPE_INDEX = 0
I_ETYPE_NAME = 'inclusion'
I_ETYPE_ONE_HOT = torch.nn.functional.one_hot(torch.tensor([I_ETYPE_INDEX]), ETYPE_COUNT)
B_ETYPE_INDEX = 1
B_ETYPE_NAME = 'bridge'
B_ETYPE_ONE_HOT = torch.nn.functional.one_hot(torch.tensor([B_ETYPE_INDEX]), ETYPE_COUNT)


class ZXMatchDiagram(nx.Graph):
    NTYPE = 'type'
    ETYPE = 'type'

    def __init__(self, diagram: ZXDiagram, one_hot_types: bool):
        self.zx_diagram = diagram
        self.one_hot_types = one_hot_types
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        super().__init__(nx.Graph())

    def to_pyg_hetero_data(self) -> HeteroData:
        pass

    @staticmethod
    def _flatten_and_concatenate_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
        # Flatten each tensor in the list to ensure it's 1D
        flattened_tensors = [torch.flatten(tensor) for tensor in tensors]
        # Concatenate all the flattened tensors into a single 1D tensor
        concatenated_tensor = torch.cat(flattened_tensors)
        return concatenated_tensor

    def to_pyg_data(self) -> Data:
        # Node indices
        node_index = dict()
        # Node features
        node_features_list = []
        for i, (node, node_data) in enumerate(self.nodes(data=True)):
            # Flatten, concatenate, and append node features
            node_features_list.append(
                self._flatten_and_concatenate_tensors([node_data[attr] for attr in self.node_attrs]))
            node_index[node] = i
        node_features_tensor = torch.stack(node_features_list)

        # Edge indices and edge attributes
        edge_sources = []
        edge_targets = []
        indexed_edge_types = dict()
        for source, target, edge_data in self.edges(data=True):
            source_index = node_index[source]
            target_index = node_index[target]
            edge_sources.append(source_index)
            edge_targets.append(target_index)
            edge_type = torch.flatten(
                edge_data[self.ETYPE] if self.one_hot_types else torch.tensor([edge_data[self.ETYPE]],
                                                                              dtype=torch.int64))
            indexed_edge_types[(source_index, target_index)] = edge_type

        # TODO: Verify that edges can be supplied to 'Data' unordered.
        edge_types = []
        for source_index, target_index in zip(edge_sources, edge_targets):
            edge_types.append(indexed_edge_types[(source_index, target_index)])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.stack(edge_types)

        # Create the PyTorch Geometric Data object including edge attributes
        data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr)
        return data


def to_zx_match_diagram(zx_diagram: ZXDiagram, one_hot_types: bool, phase_denominator: int) -> ZXMatchDiagram:
    zx_match_diagram = ZXMatchDiagram(zx_diagram, one_hot_types)
    matches = list(zx_diagram.compute_matches())
    for match in matches:
        add_match(zx_match_diagram, zx_diagram, match, one_hot_types, phase_denominator)
    add_b_edges(zx_match_diagram, zx_diagram, one_hot_types)
    num_nodes = zx_match_diagram.number_of_nodes()
    num_matches = len(matches)
    assert num_nodes == num_matches, f'Number of nodes {num_nodes} in match diagram != number of matches ' \
                                     f'{num_matches}'
    return zx_match_diagram


def compute_node_type_attr(match: Match, one_hot_types: bool) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor([match.index]), MATCH_TYPE_COUNT) if one_hot_types else match.index


def int_to_phase(phase_bucket: int, phase_denominator: int) -> float:
    """
    Converts an integer representation back into a float value based on the unit circle position,
    ensuring compatibility with the wrap-around behavior of the U(1) group.

    :param phase_bucket: The integer representing the discrete position on the unit circle.
    :param phase_denominator: The number of discrete positions (buckets) on the unit circle.
    :return: The float value representing the position on the unit circle.
    """
    if phase_denominator <= 0:
        raise ValueError(f"The phase denominator {phase_denominator} is not positive.")
    # Ensure position wraps around using modulus to handle negative and overflow positions
    normalized_position = phase_bucket % phase_denominator
    return normalized_position / phase_denominator


def phase_to_int(phase: float, phase_denominator: int) -> int:
    """
    Converts a float in a fixed subset of the unit circle into an integer representation,
    accounting for the wrap-around behavior characteristic of the U(1) group.

    :param phase: The float representing the position on the unit circle.
    :param phase_denominator: The number of discrete positions (buckets) on the unit circle.
    :return: The integer representation of the position on the unit circle.
    :raises ValueError: If the input value isn't exactly a multiple of 1/denominator.
    """
    if phase_denominator <= 0:
        raise ValueError(f"The phase denominator {phase_denominator} is not positive.")
    # Normalize the position to ensure positive and wrap-around behavior
    normalized_position = (phase * phase_denominator) % phase_denominator
    if not normalized_position.is_integer():
        raise ValueError(f"The input value {phase} is not a multiple of 1/{phase_denominator}.")
    return int(normalized_position)


def compute_node_phase_attr(zx_diagram: ZXDiagram, match: Match, phase_denominator: int) -> torch.Tensor:
    if isinstance(match, FRightMatch):
        return torch.tensor([phase_to_int(zx_diagram.phase(match.nodes[0]), phase_denominator)])
    else:
        # TODO: How to handle rewrites without phases?
        return torch.tensor([0])


def compute_edge_type_attr(etype_index: int, one_hot_types: bool) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor([etype_index]), ETYPE_COUNT) if one_hot_types else etype_index


def add_match(zx_match_diagram: ZXMatchDiagram, zx_diagram: ZXDiagram, match: Match, one_hot_types: bool,
              phase_denominator: int) -> None:
    if not zx_match_diagram.has_node(match):
        zx_match_diagram.add_node(match, type=compute_node_type_attr(match, one_hot_types),
                                  phase=compute_node_phase_attr(zx_diagram, match, phase_denominator))
    if isinstance(match, CompoundMatch):
        for sub_match in match.sub_matches:
            if not zx_match_diagram.has_node(sub_match):
                add_match(zx_match_diagram, zx_diagram, sub_match, one_hot_types, phase_denominator)
            if not zx_match_diagram.has_edge(sub_match, match):
                zx_match_diagram.add_edge(match, sub_match, type=compute_edge_type_attr(I_ETYPE_INDEX, one_hot_types))
    return


def add_b_edges(zx_match_diagram: ZXMatchDiagram, diagram: ZXDiagram, one_hot_types: bool) -> None:
    for u in diagram.basis_nodes():
        for v in basis_neighbors(diagram, u):
            u_match = f_right_match_from_ndata(diagram, u)
            v_match = f_right_match_from_ndata(diagram, v)
            if not connected(zx_match_diagram, u_match, v_match):
                zx_match_diagram.add_edge(u_match, v_match, type=compute_edge_type_attr(B_ETYPE_INDEX, one_hot_types))


def basis_neighbors(diagram: ZXDiagram, n: int) -> set[int]:
    return {m for m in diagram.neighbors(n) if diagram.is_basis(m)}


def f_right_match_from_ndata(diagram: ZXDiagram, node: int) -> FRightMatch:
    if diagram.is_z_basis(node):
        return FRightZMatch(node)
    elif diagram.is_x_basis(node):
        return FRightXMatch(node)
    else:
        raise Exception(f'Unexpected node type {diagram.type(node)}')


def is_i_edge(etype: str | int | torch.Tensor) -> bool:
    if isinstance(etype, str):
        return etype == I_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == I_ETYPE_INDEX
    elif isinstance(etype, torch.Tensor):
        return torch.equal(etype, I_ETYPE_ONE_HOT)
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def has_i_edge(zx_match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = zx_match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_i_edge(edata[ETYPE])
    return False


def i_neighbors(zx_match_diagram: ZXMatchDiagram, u_match: Match) -> Iterator[Match]:
    for u_neighbor in zx_match_diagram.neighbors(u_match):
        if has_i_edge(zx_match_diagram, u_match, u_neighbor):
            yield u_neighbor


def is_b_edge(etype: str | int | torch.Tensor) -> bool:
    if isinstance(etype, str):
        return etype == B_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == B_ETYPE_INDEX
    elif isinstance(etype, torch.Tensor):
        return torch.equal(etype, B_ETYPE_ONE_HOT)
    else:
        raise Exception('Unexpected node type representation ' + str(etype))


def has_b_edge(zx_match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    edata = zx_match_diagram.get_edge_data(u_match, v_match)
    if edata is not None:
        return is_b_edge(edata[ETYPE])
    return False


def is_match_neighbor(zx_match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    for u_neighbor in i_neighbors(zx_match_diagram, u_match):
        if v_match in i_neighbors(zx_match_diagram, u_neighbor):
            return True
    return False


def connected(zx_match_diagram: ZXMatchDiagram, u_match: Match, v_match: Match) -> bool:
    return is_match_neighbor(zx_match_diagram, u_match, v_match) or has_b_edge(zx_match_diagram, u_match, v_match)
