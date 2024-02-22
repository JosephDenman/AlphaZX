import torch


def cat_phase_to_float(cat_phase: torch.Tensor, phase_denominator: int) -> float:
    """
    Converts a tensor phase representation back into a float value based on the unit circle position,
    ensuring compatibility with the wrap-around behavior of the U(1) group.

    :param cat_phase: A scalar tensor representing the discrete position on the unit circle.
    :param phase_denominator: The number of discrete positions (categories) on the unit circle.
    :return: The float value representing the position on the unit circle.
    """
    if len(cat_phase.shape) != 0:
        raise ValueError(f"The input tensor {cat_phase} is not a scalar.")
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


def cat_new_edges_to_int(cat_new_edges: torch.Tensor) -> int:
    if len(cat_new_edges.shape) != 0:
        raise ValueError(f"The input tensor {cat_new_edges} is not a scalar.")
    # The number of new edges is the category value plus one, since the new edge distribution is a categorical distribution
    # with categories 0, 1, 2, ..., n, where n + 1 is the max number of possible new edges.
    return int(cat_new_edges) + 1


def bernoulli_transfer_edges_to_tuple(bernoulli_transfer_edges: torch.Tensor) -> tuple[int, int]:
    pass