import torch


def rand_distribution_with_zeros(size: torch.Size, zero_prob: float = 0.3) -> torch.Tensor:
    """
    Generate a tensor of specified size where the last dimension represents a valid probability distribution,
    potentially including zero entries. This function supports generating 1D, 2D, 3D, and 4D tensors.

    :param size: A torch.Size object specifying the dimensions of the tensor.
    :param zero_prob: The probability of an entry being set to zero, controlling the sparsity of the tensor.
    :returns: A tensor of the specified size where each row in the last dimension sums to 1.
    """
    # Validate inputs
    if not isinstance(size, torch.Size):
        raise ValueError("size must be a torch.Size object.")
    if not 1 <= len(size) <= 4:
        raise ValueError("This function supports 1D, 2D, 3D, and 4D tensors only.")
    if not (0 <= zero_prob <= 1):
        raise ValueError("zero_prob must be between 0 and 1.")
    if len(size) == 1 and size[0] == 1:
        return torch.tensor([1.])
    # Generate initial random values
    distribution = torch.rand(size)
    # Apply zero probability
    zero_mask = torch.rand(size) < zero_prob
    distribution[zero_mask] = 0
    # Normalize the last dimension to sum to 1
    sum_last_dim = distribution.sum(dim=-1, keepdim=True)
    # Avoid division by zero by setting zero rows' sums to 1 (since dividing by 1 has no effect)
    sum_last_dim[sum_last_dim == 0] = 1
    distribution /= sum_last_dim
    return distribution


def rand_distribution(size: torch.Size) -> torch.Tensor:
    """
    Generate a tensor of specified size where the last dimension represents a valid probability distribution and has no
    zero entries. This function supports generating 1D, 2D, 3D, and 4D tensors.

    :param size: A torch.Size object specifying the dimensions of the tensor.
    :returns: A tensor of the specified size where each row in the last dimension sums to 1.
    """
    return rand_distribution_with_zeros(size, 0.)


def rand_mixture_probs(batch_size: int, num_action_types: int) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_action_types]))


def rand_node_probs(batch_size: int, num_nodes: int) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes]))


def rand_phase_probs(batch_size: int, num_nodes: int, num_phase_buckets: int) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes, num_phase_buckets]))


def rand_new_edge_probs(batch_size: int, num_nodes: int, num_new_edge_buckets: int) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes, num_new_edge_buckets]))


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


def rand_single_node_transfer_edge_probs(incident_edges: int, zero_prob: float = 0.3) -> torch.Tensor:
    """
    Generate a 2D tensor of shape (incident_edges, incident_edges + 1) where:
    - The first column is a normalized probability distribution with potentially zero entries.
    - For every row, the entries after the first form a normalized probability distribution, also with potentially zero entries.

    :param incident_edges: Number of incident edges, determining the tensor's shape.
    :param zero_prob: Probability of an entry being set to zero, controlling the sparsity of the tensor.
    :returns: A 2D tensor with the specified properties.
    """
    return prepend_column(rand_distribution_with_zeros(torch.Size([incident_edges, incident_edges]), zero_prob),
                          rand_distribution_with_zeros(torch.Size([incident_edges]), zero_prob))


def generate_distribution_adjusted(size: torch.Size, zero_prob: float = 0.5) -> torch.Tensor:
    """
    Adjusted function to generate distributions ensuring the first column is a valid distribution,
    and for 2D tensors, ensuring the first two entries of each row can't be both zero.
    """
    distribution = torch.rand(size)
    zero_mask = torch.rand(size) < zero_prob
    distribution[zero_mask] = 0

    # Ensure the first entry in each distribution isn't zero (for at least one incident edge)
    if len(size) == 2:
        distribution[:, 0] = torch.where(distribution[:, 0] == 0, torch.rand(size[0]), distribution[:, 0])

    # Normalize
    sum_last_dim = distribution.sum(dim=-1, keepdim=True)
    sum_last_dim[sum_last_dim == 0] = 1  # Avoid division by zero
    distribution /= sum_last_dim
    return distribution


def pad_tensor_to_size(input_tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """
    Pads a given tensor of shape (x, x + 1) with zeros to be of shape (y, y + 1).

    :param input_tensor: The input tensor to pad, assumed to have shape (x, x + 1).
    :param new_size: The desired size 'y' for the new tensor. Must be >= x.
    :return: A new tensor of shape (y, y + 1) with the original tensor's values and additional zeros.
    """
    # Extract the current size 'x' from the input tensor
    x = input_tensor.shape[0]
    # Validate the new size
    if new_size < x:
        raise ValueError("new_size must be greater than or equal to the input tensor's size")
    # Create a new zero tensor of the desired size (y, y + 1)
    padded_tensor = torch.zeros(new_size, new_size + 1, dtype=input_tensor.dtype)
    # Copy the values from the input tensor to the top-left corner of the new tensor
    padded_tensor[:x, :x + 1] = input_tensor
    return padded_tensor


def rand_transfer_edge_probs(batch_size: int, max_num_nodes:int, max_incident_edges: int, zero_prob: float = 0.3) -> torch.Tensor:
    # Initialize the output tensor with zeros
    output_tensor = torch.zeros((batch_size, max_num_nodes, max_incident_edges, max_incident_edges + 1))
    for b in range(batch_size):
        num_nodes = torch.randint(1, max_num_nodes + 1, (1,)).item()  # Randomly select the number of nodes for this batch
        print('num_nodes = ', num_nodes)
        for n in range(num_nodes):
            num_incident_edges = torch.randint(1, max_incident_edges + 1, (1,)).item()  # Randomly select the number of incident edges for this node
            print('num_incident_edges = ', num_incident_edges)
            mixture_probs = rand_distribution_with_zeros(torch.Size([num_incident_edges]), zero_prob)
            print('mixture_probs = ', mixture_probs)
            remaining_probs = torch.rand(num_incident_edges, num_incident_edges)
            combined_probs = prepend_column(remaining_probs, mixture_probs)
            padded_probs = pad_tensor_to_size(combined_probs, max_incident_edges)
            print('padded_probs = ', padded_probs)
            if (padded_probs == 0.).all():
                padded_probs[0, 0] = 1.
            print('padded_probs = ', padded_probs)
            output_tensor[b, n, :, :] = padded_probs
    return output_tensor


# Example usage
batch_size = 2
max_num_nodes = 4
max_incident_edges = 3
transfer_edge_probs = rand_transfer_edge_probs(batch_size, max_num_nodes, max_incident_edges)
print('transfer_edge_probs = ', transfer_edge_probs)
print("Corrected Transfer edge probabilities tensor shape:", transfer_edge_probs.shape)

# Validate the distributions
for b in range(batch_size):
    for n in range(max_num_nodes):
        print(f"Batch {b}, Node {n}, Sum of first column:", transfer_edge_probs[b, n, :, 0].sum())
        for e in range(max_incident_edges):
            print(f"  Edge {e}, Sum of row:", transfer_edge_probs[b, n, e, :].sum())


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


# TODO: Have this function generate realistic samples, i.e., samples where batches have different node counts and are
#       padded to a uniform dimension.
def rand_sampled_actions_batch(batch_size: int, num_actions: int, action_length: int, num_action_types: int,
                               num_nodes: int, num_phase_buckets: int, num_new_edge_buckets: int,
                               non_parametric_action_types: list[int]) -> torch.Tensor:
    """
    Generate a batch of random tensors with the first entry in each row representing the type of the action,
    the second entry randomly sampled from a predefined set of consecutive integers [0, num_nodes - 1] without repetition,
    and the third entry randomly selected from a set of predefined consecutive integers [0, num_phase_buckets - 1].
    For non-parametric action types, all entries in the row after the second are zeroed out.

    :param batch_size: Number of tensors in the batch.
    :param num_actions: Number of actions in each batch, assumes num_actions <= num_nodes.
    :param action_length: Length of each row tensor, must be at least 4.
    :param num_action_types: Number of action types to choose from.
    :param num_nodes: Number of nodes, used to generate second entries.
    :param num_phase_buckets: Number of possible phases, used to generate third entries.
    :param num_new_edge_buckets: Numer of possible new edges, used to generate fourth entries.
    :param non_parametric_action_types: Action types that trigger zeroing out all subsequent entries after the second entry.
    :returns: A batch of tensors with dimensions (batch_size, num_actions, action_length). Rows in each tensor are sorted
              according to the second entry (node index).
    """
    if action_length < 4:
        raise ValueError("Action length must be at least 4.")
    # Generate random action types for the first column
    first_types = torch.randint(low=0, high=num_action_types, size=(batch_size, num_actions, 1), dtype=torch.float)
    # Generate random second entries without repetition and then sort them
    second_types = torch.zeros(batch_size, num_actions, 1, dtype=torch.float)
    for i in range(batch_size):
        perm = torch.randperm(num_nodes)[:num_actions]
        sorted_perm, _ = torch.sort(perm)
        second_types[i] = sorted_perm.unsqueeze(-1).float()
    # Generate random third entries
    third_types = torch.randint(low=0, high=num_phase_buckets, size=(batch_size, num_actions, 1), dtype=torch.float)
    fourth_types = torch.randint(low=0, high=num_new_edge_buckets, size=(batch_size, num_actions, 1), dtype=torch.float)
    # Generate random values for the rest of each row
    values = torch.rand(batch_size, num_actions, action_length - 4)
    # Concatenate the components to form the complete tensor
    batch = torch.cat((first_types, second_types, third_types, fourth_types, values), dim=2)
    # Apply non-parametric action rule
    for np_type in non_parametric_action_types:
        mask = (batch[:, :, 0] == np_type).unsqueeze(-1)
        zeroed_values = torch.zeros(batch_size, num_actions, action_length - 2, dtype=torch.float)
        batch = torch.where(mask, torch.cat((batch[:, :, :2], zeroed_values), dim=2), batch)
    return batch


# Example parameters
# batch_size = 3
# num_actions = 7
# action_length = 6
# num_action_types = 2
# num_nodes = 8
# num_phase_buckets = 4
# num_new_edges_buckets = 10
# non_parametric_action_types = [1]
#
# sampled_actions_batch = rand_sampled_actions_batch(batch_size, num_actions, action_length, num_action_types,
#                                                    num_nodes, num_phase_buckets, num_new_edges_buckets,
#                                                    non_parametric_action_types)
#
# print('sampled_actions = ', sampled_actions_batch)
#
# phase_probs_batch = rand_phase_probs(batch_size, num_nodes, num_phase_buckets)
#
# print('phase_probs_batch = ', phase_probs_batch)
#
# print(gather_phase_probs(phase_probs_batch, sampled_actions_batch))
