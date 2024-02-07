import torch
import torch.nn.functional as F

from models.alpha_zx_distribution import gather_phase_probs


def rand_probs(size: torch.Size) -> torch.Tensor:
    """
    Generate a tensor with random values, normalized to sum to 1 along the last dimension. This function supports
    generating 1D, 2D, and 3D tensors. For a 1D tensor, each element sums to 1. For a 2D tensor, each row sums to 1.
    For a 3D tensor, each row in every 2D tensor sums to 1.

    :param size: The size of the tensor to generate. This can be a 1D, 2D, or 3D shape, specified as a `torch.Size`.
                 For a 1D tensor, `size` should have one dimension. For a 2D tensor, `size` should have two dimensions.
                 For a 3D tensor, `size` should have three dimensions.
    :returns: A `torch.Tensor` of the specified `size`, where values are normalized to sum to 1 along the last dimension.
              For a 1D tensor, the sum of the tensor will be 1. For a 2D tensor, the sum of each row will be 1.
              For a 3D tensor, the sum of each row in every 2D tensor within it will be 1.
    Raises:
    - ValueError: If `size` is not for a 1D, 2D, or 3D tensor, indicating that the provided `size` argument does not
                  conform to the expected dimensions.
    """
    # Generate random values based on the provided size
    random_values = torch.rand(size)
    # Normalize the values to sum to 1 along the last dimension
    if len(size) == 1:
        # For a 1D tensor, simply divide by the sum
        distribution = random_values / random_values.sum()
    elif len(size) == 2:
        # For a 2D tensor, divide each row by its sum to get row-wise distributions
        distribution = random_values / random_values.sum(dim=1, keepdim=True)
    elif len(size) == 3:
        # For a 3D tensor, divide each row by its sum within each 2D tensor to get row-wise distributions
        distribution = random_values / random_values.sum(dim=2, keepdim=True)
    else:
        raise ValueError("Size must be for a 1D, 2D, or 3D tensor")
    return distribution


def rand_probs_with_zeros(size: torch.Size, zero_frac=0.3) -> torch.Tensor:
    """
    Generate a tensor with random values normalized to sum to 1 along the last dimension,
    with a specified fraction of entries set to zero. Supports generating 1D, 2D, and 3D tensors.

    :param size: The size of the tensor to generate. Can be a 1D, 2D, or 3D shape, specified as a `torch.Size`.
    :param zero_frac: The fraction of entries to set to zero, specified as a float between 0 and 1.
    :returns: A `torch.Tensor` of the specified `size`, normalized along the last dimension,
              including some zero entries according to `zero_frac`.
    """
    if len(size) not in [1, 2, 3]:
        raise ValueError("Size must be for a 1D, 2D, or 3D tensor")
    # Generate random values
    random_values = torch.rand(size)
    # Apply zeros to random positions
    flat_random_values = random_values.view(-1)
    num_zeros = int(flat_random_values.numel() * zero_frac)
    if num_zeros > 0:
        zero_indices = torch.randperm(flat_random_values.numel())[:num_zeros]
        flat_random_values[zero_indices] = 0
    random_values = flat_random_values.view(size)
    # Normalize the tensor
    if len(size) == 1:
        distribution = random_values / random_values.sum()
    else:
        # Normalize 2D or 3D tensor row-wise
        dim = -2 if len(size) == 3 else 1
        sums = random_values.sum(dim=dim, keepdim=True)
        # Avoid division by zero by adding a small epsilon where sums are zero
        distribution = random_values / torch.where(sums != 0, sums, torch.ones_like(sums))
    # Ensure at least one non-zero entry per row for 2D and 3D tensors
    if len(size) in [2, 3]:
        for i in range(size[0] if len(size) == 3 else 1):  # Iterate through batches if 3D
            for j in range(size[1] if len(size) == 3 else size[0]):  # Iterate through rows
                if torch.all(distribution[i][j] == 0):
                    # Assign a small value to a random position in the row to avoid all zeros
                    rand_pos = torch.randint(0, size[-1], (1,))
                    distribution[i][j][rand_pos] = 1e-6
        # Re-normalize after ensuring no all-zero rows
        if len(size) == 3:
            distribution = distribution / distribution.sum(dim=2, keepdim=True)
        else:
            distribution = distribution / distribution.sum(dim=1, keepdim=True)
    return distribution


def rand_mixture_probs(batch_size: int, num_action_types: int) -> torch.Tensor:
    return rand_probs(torch.Size([batch_size, num_action_types]))


def rand_node_probs(batch_size: int, num_nodes: int) -> torch.Tensor:
    return rand_probs(torch.Size([batch_size, num_nodes]))


def rand_phase_probs(batch_size: int, num_nodes: int, num_phase_buckets: int) -> torch.Tensor:
    return rand_probs(torch.Size([batch_size, num_nodes, num_phase_buckets]))


def rand_new_edge_probs(batch_size: int, num_nodes: int, num_new_edge_buckets: int) -> torch.Tensor:
    return rand_probs(torch.Size([batch_size, num_nodes, num_new_edge_buckets]))


def rand_o_edge_probs(nodes: int, o_edge_buckets: int) -> torch.Tensor:
    o_edge_probs = []
    for _ in range(nodes):
        o_edge_count = torch.randint(0, o_edge_buckets + 1, (1,)).item()
        selection_probs = rand_probs(torch.Size([o_edge_count, o_edge_count]))
        mixture_probs = rand_probs(torch.Size([o_edge_count]))
        all_probs = prepend_column(selection_probs, mixture_probs)
        o_edge_padding = o_edge_buckets - o_edge_count
        padded_probs = F.pad(all_probs, (0, o_edge_padding, 0, o_edge_padding + 1))
        o_edge_probs.append(padded_probs)
    return torch.stack(o_edge_probs)


def prepend_column(tensor: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
    """
    Prepend a column to a 2D tensor.

    :param tensor: A 2D tensor to which the column will be prepended. This tensor should have dimensions [n, m],
                   where `n` is the number of rows and `m` is the number of columns.
    :param column: A 1D tensor representing the column to prepend. It must have the same height (number of rows)
                   as the input tensor, with length `n`.
    :returns: A new tensor with the column prepended, having dimensions [n, m+1], where the new column is added
              as the first column of the tensor.

    Raises:
    - ValueError: If the input `tensor` is not 2D, or if `column` is not 1D.
    - ValueError: If `column` does not have the same number of rows as `tensor`.

    Example:
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        column = torch.tensor([0, 0])
        result = prepend_column(tensor, column)
        # result is now torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])
    """
    if tensor.dim() != 2 or column.dim() != 1:
        raise ValueError("Input tensor must be 2D and column must be 1D")
    if tensor.size(0) != column.size(0):
        raise ValueError("Column must have the same number of rows as the input tensor")
    # Reshape the column to be a 2D tensor with a single column
    column = column.unsqueeze(1)
    # Concatenate the column tensor and the original tensor along dimension 1 (columns)
    result = torch.cat((column, tensor), dim=1)
    return result


def recover_original_tensor_batch(padded_tensors: torch.Tensor) -> list[torch.Tensor]:
    """
    Recover the original tensors from a batch of padded tensors. Each original tensor is assumed to have
    been padded to the same dimensions with all-zero rows for padding. The function identifies the original
    tensor size by finding the first all-zero row, which indicates the start of padding, and slices each tensor
    to its original dimensions of N x (N+1).

    :param padded_tensors: A batch of 2D padded tensors, where each tensor is assumed to have been padded
                           with all-zero rows to a uniform size. The tensor batch is represented as a 3D tensor
                           with dimensions [batch_size, max_rows, max_cols], where `batch_size` is the number
                           of tensors, and `max_rows` and `max_cols` are the dimensions to which the tensors
                           have been padded.
    :returns: A list of 2D tensors, where each tensor has been trimmed to its original size of N x (N+1).
              The original size is determined by identifying the first all-zero row in each padded tensor,
              assuming that the original height of the tensor is N (the number of non-zero rows) and the
              original width is N+1.
    """
    original_tensors = []
    for tensor in padded_tensors:
        # Find N, the number of non-zero rows which represents original height
        n = next((i for i, row in enumerate(tensor) if torch.all(row == 0)), tensor.shape[0])
        # Assuming the width is N+1, slice the tensor accordingly
        original_tensors.append(tensor[:n, :n + 1])
    return original_tensors


def replace_zero_rows_with_uniform(t: torch.Tensor) -> torch.Tensor:
    """
    Replace all-zero rows in a 2D tensor or in each 2D tensor within a 3D batch with a uniform distribution.

    :param t: A 2D or 3D tensor. For a 2D tensor, each row is checked individually, and if a row consists
              entirely of zeros, it is replaced with a uniform distribution. For a 3D tensor, the operation
              is applied to each 2D tensor in the batch independently.
    :returns: The modified tensor with uniform distributions replacing all-zero rows. The output tensor retains
              the same shape as the input tensor, but with rows that were originally all zeros now containing
              values from a uniform distribution that sums to 1.
    """
    if t.dim() not in [2, 3]:
        raise ValueError("Input must be a 2D or 3D tensor")
    # Handle both 2D and 3D tensors by adding a dummy batch dimension to 2D tensors
    if t.dim() == 2:
        t = t.unsqueeze(0)
    # Identify rows that are all zeroes across the batch
    zero_rows_mask = torch.all(t == 0, dim=-1)
    # Calculate the uniform distribution for a single row
    num_columns = t.size(-1)
    uniform_row = torch.full((1, 1, num_columns), fill_value=1 / num_columns, device=t.device)
    # Use broadcasting to replace zero rows with the uniform distribution
    # Expanding zero_rows_mask for broadcasting
    zero_rows_mask_expanded = zero_rows_mask.unsqueeze(-1)
    # Perform the replacement
    t = torch.where(zero_rows_mask_expanded, uniform_row.expand_as(t), t)
    # Remove the dummy batch dimension if it was added
    if t.size(0) == 1 and t.dim() == 3:
        t = t.squeeze(0)
    return t


import torch


def rand_sampled_actions_batch(batch_size: int, num_actions: int, action_length: int, num_action_types: int,
                               num_nodes: int, num_phase_buckets: int,
                               non_parametric_action_types: list[int]) -> torch.Tensor:
    if action_length < 4:
        raise ValueError("Action length must be at least 4.")

    # Generate random action types for the first column
    first_types = torch.randint(low=0, high=num_action_types, size=(batch_size, num_actions, 1), dtype=torch.float)

    # Generate random second entries without repetition
    second_types = torch.zeros(batch_size, num_actions, 1, dtype=torch.float)
    for i in range(batch_size):
        second_types[i] = torch.randperm(num_nodes)[:num_actions].unsqueeze(-1).float()

    # Generate random third entries
    third_types = torch.randint(low=0, high=num_phase_buckets, size=(batch_size, num_actions, 1), dtype=torch.float)

    # Generate random values for the rest of each row
    values = torch.rand(batch_size, num_actions, action_length - 3)

    # Concatenate the components to form the complete tensor
    batch = torch.cat((first_types, second_types, third_types, values), dim=2)

    # Apply non-parametric action rule
    for np_type in non_parametric_action_types:
        mask = (batch[:, :, 0] == np_type).unsqueeze(-1)
        zeroed_values = torch.zeros(batch_size, num_actions, action_length - 2, dtype=torch.float)
        batch = torch.where(mask, torch.cat((batch[:, :, :2], zeroed_values), dim=2), batch)

    return batch

#
# Example parameters
batch_size = 3
num_actions = 3
action_length = 6
num_action_types = 4
num_nodes = 5
num_phase_buckets = 3
non_parametric_action_types = [1, 2]
#
# # Generate the batch
sampled_actions_batch = rand_sampled_actions_batch(batch_size, num_actions, action_length, num_action_types,
                                   num_nodes, num_phase_buckets, non_parametric_action_types)
# print(sampled_actions_batch)
#
# phase_probs_batch = rand_phase_probs(batch_size, num_nodes, num_phase_buckets)
# print(phase_probs_batch)

print('sampled_actions = ', sampled_actions_batch)

phase_probs_batch = rand_phase_probs(batch_size, num_nodes, num_phase_buckets)

print('phase_probs_batch = ', phase_probs_batch)

print(gather_phase_probs(phase_probs_batch, sampled_actions_batch))