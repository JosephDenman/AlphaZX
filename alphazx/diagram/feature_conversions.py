import torch
import torch_geometric as pyg

from alphazx import compute_basis_neighbors
from alphazx.diagram.match import FRightMatch
from alphazx.diagram.zx_match_diagram import DataIndexToMatch


def is_integer_tensor(tensor: torch.Tensor) -> bool:
    """
    Checks if a 0D PyTorch tensor represents an integer.

    :param tensor: A 0D PyTorch tensor to be checked.
    :return: A boolean indicating if the tensor represents an integer. True if the tensor is of an integer type or is
             a floating-point number that represents an integer value (e.g., 2.0), and False otherwise.
    :raises ValueError: If the input is not a 0D PyTorch tensor representing an integer.
    """
    # Check if tensor is already of integer type
    if tensor.dim() == 0:
        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return True
        # Check if it's a floating point but represents an integer
        return float(tensor) == float(tensor.floor())
    else:
        raise ValueError("Input must be a 0D PyTorch tensor")


def cat_phase_to_float(cat_phase: torch.Tensor, phase_denominator: int) -> float:
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


def float_phase_to_cat(float_phase: float, phase_denominator: int) -> torch.Tensor:
    """
    Converts a float in a fixed subset of the unit circle into an integer representation,
    accounting for the wrap-around behavior characteristic of the U(1) group.

    :param float_phase: The float representing the position on the unit circle.
    :param phase_denominator: The number of discrete positions (buckets) on the unit circle.
    :return: The integer representation of the position on the unit circle.
    :raises ValueError: If the input value isn't exactly a multiple of 1/denominator.
    """
    if phase_denominator <= 0:
        raise ValueError(f"The phase denominator {phase_denominator} is not positive.")
    # Normalize the position to ensure positive and wrap-around behavior
    normalized_position = (float_phase * phase_denominator) % phase_denominator
    if not normalized_position.is_integer():
        raise ValueError(f"The input value {float_phase} is not a multiple of 1/{phase_denominator}.")
    return torch.tensor(int(normalized_position))


def cat_new_edges_to_int(cat_new_edges: int) -> int:
    # The number of new edges is the category value plus one, since the new edge distribution is a categorical distribution
    # with categories 0, 1, 2, ..., n, where n + 1 is the max number of possible new edges.
    return cat_new_edges + 1


def bernoulli_transfer_edges_to_set(node: int,
                                    bernoulli_transfer_edges: tuple,
                                    data: pyg.data.Data,
                                    data_index: DataIndexToMatch) -> set[int]:
    basis_neighbors = compute_basis_neighbors(data.edge_index, node, data.node_type)
    basis_neighbors = basis_neighbors[torch.tensor(bernoulli_transfer_edges[:len(basis_neighbors)], dtype=torch.bool)]
    transfer_edges = []
    for neighbor in basis_neighbors.tolist():
        neighbor_match = data_index[neighbor]
        assert isinstance(neighbor_match, FRightMatch), f'Expected {neighbor_match} to be an FRightMatch'
        transfer_edges.append(neighbor_match.node)
    return set(transfer_edges)


def eliminate_columns_with_value(matrix: torch.Tensor, value: int) -> torch.Tensor:
    """
    Eliminates columns from the input matrix that contain the specified value.

    Parameters:
        matrix (torch.Tensor): The input tensor from which columns are to be removed.
        value (int): The value based on which columns will be removed.

    Returns:
        torch.Tensor: A tensor with the specified columns removed.
    """
    # Create a mask that is True where the element is not equal to the value
    mask = matrix != value

    # Use `all()` along the rows (dim=0) to find columns where all elements are True
    # (i.e., columns that do not contain the value)
    valid_columns = mask.all(dim=0)

    # Use the mask to select valid columns
    result = matrix[:, valid_columns]
    return result
