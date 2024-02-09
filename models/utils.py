import torch


def rand_distribution_with_zeros(size: torch.Size, zero_prob: float = 0.3) -> torch.Tensor:
    # Check if the size argument and zero_prob are valid
    if not (1 <= len(size) <= 4):
        raise ValueError("The length of the size argument must be between 1 and 4.")
    if not (0 <= zero_prob <= 1):
        raise ValueError("zero_prob must be between 0 and 1.")
    # Generate a random tensor
    distributions = torch.rand(size)
    # Insert zeros based on zero_prob
    zero_mask = torch.rand(size) < zero_prob
    distributions[zero_mask] = 0
    # Normalize the distributions
    if len(size) > 1:
        sums = distributions.sum(dim=-1, keepdim=True)
        # Check for distributions where the sum is 0 and set at least one value to ensure sum is not 0
        for idx, s in enumerate(sums.reshape(-1)):
            if s.item() == 0:
                distributions.reshape(-1, size[-1])[idx, torch.randint(0, size[-1], (1,))] = 1.0
        sums = distributions.sum(dim=-1, keepdim=True)  # Recalculate sums after adjustments
        distributions = distributions / sums
    else:
        sum_val = distributions.sum()
        if sum_val == 0:  # Adjust if sum is 0
            distributions[torch.randint(0, size[-1], (1,))] = 1.0
            sum_val = distributions.sum()
        distributions = distributions / sum_val
    assert torch.all(torch.isclose(distributions.sum(dim=-1, keepdim=True), torch.tensor(1.0), atol=1e-6)), \
        f'Produced invalid probability distribution {distributions} of size {size}'
    return distributions


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


def rand_inclusive(size: torch.Size) -> torch.Tensor:
    # Generate values in the range [0, 1)
    values = torch.rand(size)
    # Randomly set one value to 1, regardless of tensor size
    # For demonstration, this randomly chooses whether to set a value to 1
    should_set_one = torch.rand(1) < 0.05  # 5% chance to set a value to 1
    if should_set_one:
        if torch.numel(values) == 1:
            values[0] = 1.0
        else:
            indices = torch.randint(0, torch.numel(values), (1,))
            values.view(-1)[indices] = 1.0
    return values


def rand_transfer_edge_probs(batch_size: int, max_num_nodes: int, max_incident_edges: int):
    """
    Generates valid parameter sets for a MultivariateBernoulliMixture distribution with specific modifications.
    Each innermost 2D tensor, representing a distribution, is modified by zeroing out certain rows and columns
    based on a randomly selected integer, creating a structure within the parameter tensor where only a subset
    of the potential distributions are effectively used.

    For each innermost 2D tensor of dimension `(x, x + 1)`, a random integer `y` is sampled from the range `[1, x - 1]` inclusive.
    All rows from `y` downwards (including `y`) and all columns from `y` (excluding `y`) are set to zero. The first column
    in each 2D tensor, representing mixture weights, is adjusted so that its elements sum to 1, ensuring the validity
    of the mixture model parameters.

    :param batch_size: The number of batches to generate.
    :type batch_size: int
    :param max_num_nodes: The maximum number of nodes (distributions) in each batch.
    :type max_num_nodes: int
    :param max_incident_edges: The maximum number of incident edges (events) in each distribution.
    :type max_incident_edges: int
    :returns: A 4D tensor of shape (batch_size, max_num_nodes, max_incident_edges, max_incident_edges + 1).
              The first column of each innermost 2D tensor sums to 1, with other specified modifications
              applied to simulate a mixture of Bernoulli distributions with structured sparsity.
    :rtype: torch.Tensor
    """
    if 1 == batch_size == max_num_nodes == max_incident_edges:
        params = torch.tensor([[[[1., 0.]]]])
        params[0, 0, 0, 1] = rand_inclusive(torch.Size([]))
        return params
    # Initialize the parameters tensor with random values from [0, 1]
    params = rand_inclusive(torch.Size([batch_size, max_num_nodes, max_incident_edges, max_incident_edges + 1]))
    for b in range(batch_size):
        for n in range(max_num_nodes):
            # Sample a random integer y from [1, e_incident - 1] inclusive
            if max_incident_edges == 1:
                y = 1
            else:
                y = torch.randint(1, max_incident_edges, (1,)).item()
            # Zero rows from y down (including y)
            params[b, n, y:, :] = 0
            # Zero columns from y (not including y)
            params[b, n, :, y:] = 0
            # Ensure the mixture parameters sum to 1 for the truncated distributions
            params[b, n, :, 0] /= params[b, n, :, 0].sum()
    return params


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
