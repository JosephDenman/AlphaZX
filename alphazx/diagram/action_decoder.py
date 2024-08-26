import torch
import torch_geometric as pyg

from alphazx.diagram.zx_match_diagram import DataIndexToMatch, ZXMatchDiagram
from alphazx.models.utils import compute_basis_neighbors


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
