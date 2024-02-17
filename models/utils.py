import torch


def rand_distribution_with_zeros(size: torch.Size, zero_prob: float = 0.05) -> torch.Tensor:
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


def rand_distribution(size: torch.Size, zero_prob: float = 0.) -> torch.Tensor:
    """
    Generate a tensor of specified size where the last dimension represents a valid probability distribution and has no
    zero entries. This function supports generating 1D, 2D, 3D, and 4D tensors.

    :param zero_prob:
    :param size: A torch.Size object specifying the dimensions of the tensor.
    :returns: A tensor of the specified size where each row in the last dimension sums to 1.
    """
    return rand_distribution_with_zeros(size, zero_prob)


def rand_mixture_probs(batch_size: int, num_action_types: int, zero_prob: float = 0.) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_action_types]), zero_prob)


def rand_node_probs(batch_size: int, num_nodes: int, zero_prob: float = 0.) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes]), zero_prob)


def rand_phase_probs(batch_size: int, num_nodes: int, num_phase_buckets: int, zero_prob: float = 0.) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes, num_phase_buckets]), zero_prob)


def rand_new_edge_probs(batch_size: int, num_nodes: int, num_new_edge_buckets: int,
                        zero_prob: float = 0.) -> torch.Tensor:
    return rand_distribution(torch.Size([batch_size, num_nodes, num_new_edge_buckets]), zero_prob)


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
    :param max_num_nodes: The maximum number of nodes (distributions) in each batch.
    :param max_incident_edges: The maximum number of incident edges (events) in each distribution.
    :returns: A 4D tensor of shape (batch_size, max_num_nodes, max_incident_edges, max_incident_edges + 1).
              The first column of each innermost 2D tensor sums to 1, with other specified modifications
              applied to simulate a mixture of Bernoulli distributions with structured sparsity.
    """
    if 1 == batch_size == max_num_nodes == max_incident_edges:
        params = torch.tensor([[[[1., 0.]]]])
        params[..., 1] = rand_inclusive(torch.Size([1]))
        return params
    # Initialize the parameters tensor with random values from [0, 1]
    params = rand_inclusive(torch.Size([batch_size, max_num_nodes, max_incident_edges, max_incident_edges + 1]))
    for b in range(batch_size):
        for n in range(max_num_nodes):
            if max_incident_edges == 1:
                y = 1
            else:
                # Sample a random integer y from [1, e_incident] inclusive
                y = torch.randint(1, max_incident_edges + 1, (1,)).item()
            # Zero rows from y down (including y)
            params[b, n, y:, :] = 0
            # Zero columns from y (not including y)
            params[b, n, :, y + 1:] = 0
            # Ensure the mixture parameters sum to 1 for the truncated distributions
            params[b, n, :, 0] /= params[b, n, :, 0].sum()
    return params


def insert_random_integers(tensor: torch.Tensor, num_inserts: int, insert_val: int) -> torch.Tensor:
    """
    Inserts a specified number of integers into random positions within a 1D tensor.

    :param tensor: The original 1D tensor where integers will be inserted.
    :param num_inserts: The number of integers to insert into the tensor.
    :param insert_val: The integer value to be inserted.
    :returns: A new tensor with integers inserted at random positions.

    The function creates a new tensor that is longer than the original tensor by `num_inserts` length.
    The specified integer values are inserted at random positions throughout the tensor, while the original elements are preserved.
    """
    # Length of the original tensor
    original_length = tensor.size(0)
    # New length after inserting integers
    new_length = original_length + num_inserts
    # Create an index tensor of size new_length, filled with values that are within the range of original_length
    # This will be used to scatter the original tensor values into the new tensor, leaving spaces for the inserted integers
    indices = torch.randperm(new_length)[:original_length]
    # Create a new tensor filled with the insert_val with the new length
    new_tensor = torch.full((new_length,), insert_val, dtype=tensor.dtype)
    # Scatter the original tensor values into the new tensor
    new_tensor.scatter_(0, indices, tensor)
    return new_tensor


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


def pad_and_stack_tensors(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Pads a list of tensors of the same shape (1D, 2D, or 3D) with zeros to make them all the same size,
    and then stacks them into a single tensor.

    :param tensor_list: A list of tensors to be padded and stacked. The tensors can be 1D, 2D, or 3D,
                        and any of the 1D tensors could be empty.
    :type tensor_list: list[torch.Tensor]
    :returns: A single tensor with all input tensors padded to the same size and then stacked.
    :rtype: torch.Tensor

    The function determines the maximum size in each dimension across all tensors, pads each tensor
    to match this maximum size using zeros, and finally stacks them along a new dimension.
    """
    if not tensor_list:
        raise ValueError("The tensor list must not be empty.")
    # Determine the maximum size in each dimension
    max_size = [0] * (max(tensor.ndim for tensor in tensor_list if tensor.numel() > 0))
    for tensor in tensor_list:
        for i, size in enumerate(tensor.size()):
            max_size[i] = max(max_size[i], size)
    # Pad each tensor to the maximum size and collect them in a new list
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = [(0, max_size[i] - size) for i, size in enumerate(tensor.size())]
        # Flatten the list of tuples to match the expected input format for F.pad
        pad_size_flat = [size for sizes in reversed(pad_size) for size in sizes]  # Reverse the padding order for F.pad
        padded_tensor = torch.nn.functional.pad(tensor, pad_size_flat, "constant", 0)
        padded_tensors.append(padded_tensor)
    # Stack the padded tensors
    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor


def adjust_all_zero_rows(tensor: torch.Tensor) -> torch.Tensor:
    """
    Correctly adjusts a tensor by setting the first entry of each innermost row, which consists entirely of zeros,
    to 1, while leaving other rows unchanged. Correctly handles 3D tensors.

    :param tensor: A 3D tensor to be adjusted.
    :returns: A new tensor with specified adjustments applied.

    This function iterates over each row of the tensor across all dimensions. If a row consists entirely of zeros,
    it sets the first entry of such a row to 1.
    """
    # Create a mask to identify rows that consist entirely of zeros
    zero_rows_mask = (tensor == 0).all(dim=-1)
    # Iterate over each slice and row to correctly adjust the first entry
    for i in range(tensor.size(0)):  # Iterate over the first dimension
        for j in range(tensor.size(1)):  # Iterate over the second dimension
            if zero_rows_mask[i, j]:
                tensor[i, j, 0] = 1.0  # Adjust the first entry of the row
    return tensor


def adjust_all_zero_matrices(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjusts a 4D tensor by setting the upper-left entry of each 2D inner tensor to 1 if the entire 2D inner tensor is all zeros.

    :param tensor: A 4D tensor to be adjusted.
    :returns: A new tensor with specified adjustments applied.

    This function iterates over each 2D inner tensor of the 4D input tensor. If a 2D inner tensor consists entirely of zeros,
    it sets its upper-left entry to 1.
    """
    # Create a mask to identify 2D inner tensors that consist entirely of zeros
    zero_2d_tensors_mask = (tensor == 0).all(dim=-1).all(dim=-1)
    # Iterate over the first two dimensions to correctly adjust the upper-left entry
    for i in range(tensor.size(0)):  # Iterate over the first dimension
        for j in range(tensor.size(1)):  # Iterate over the second dimension
            if zero_2d_tensors_mask[i, j]:
                tensor[i, j, 0, 0] = 1.0  # Adjust the upper-left entry of the 2D inner tensor
    return tensor


def point_dist(length: int) -> torch.Tensor:
    return torch.tensor([1.0] + [0.0] * (length - 1))


def rand_azx_dist_params(batch_size: int,
                         max_frz_nodes: int,
                         max_flz_nodes: int,
                         num_phases: int,
                         num_new_edges: int,
                         max_incident_edges: int,
                         zero_prob: float = 0.05) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixture_dist_params = []
    frz_node_dist_params = []
    flz_node_dist_params = []
    phase_dist_params = []
    new_edges_dist_params = []
    transfer_edges_dist_params = []
    for b in range(batch_size):
        b_flz_num_nodes = torch.randint(0, max_flz_nodes + 1, (1,)).item()
        b_frz_num_nodes = torch.randint(1, max_frz_nodes + 1, (1,)).item()
        b_mixture_dist_params = torch.tensor([1.0, 0.0]) if b_flz_num_nodes == 0 else rand_mixture_probs(1, 2,
                                                                                                         zero_prob).squeeze(
            0)
        # Insert random -1s into the frz node distribution indicate where flz nodes are located.
        b_frz_node_dist_params = insert_random_integers(rand_node_probs(1, b_frz_num_nodes, 0.05).squeeze(0),
                                                        b_flz_num_nodes, -1)
        b_flz_node_indices = b_frz_node_dist_params == -1
        b_frz_node_indices = b_frz_node_dist_params != -1
        if b_flz_num_nodes == 0:
            # If there are no flz nodes, the flz node distribution is a point distribution. Since the mixture
            # distribution is [1, 0], the flz node distribution is not used.
            b_flz_node_dist_params = point_dist(b_frz_num_nodes)
        else:
            # Otherwise, insert random probabilities at the -1 positions in the frz node distribution. All other entries
            # are set to 0.
            b_flz_node_dist_params = torch.zeros([b_flz_num_nodes + b_frz_num_nodes])
            b_flz_node_dist_params[b_flz_node_indices] = rand_node_probs(1, b_flz_num_nodes, zero_prob).squeeze(0)
        # Remove the -1 entries.
        b_frz_node_dist_params[b_flz_node_indices] = 0.

        b_phase_dist_params = torch.zeros([b_flz_num_nodes + b_frz_num_nodes, num_phases])
        b_phase_dist_params[b_frz_node_indices] = rand_phase_probs(1, b_frz_num_nodes, num_phases, zero_prob).squeeze(0)
        b_phase_dist_params = adjust_all_zero_rows(b_phase_dist_params.unsqueeze(0)).squeeze(0)

        b_new_edges_dist_params = torch.zeros([b_flz_num_nodes + b_frz_num_nodes, num_new_edges])
        b_new_edges_dist_params[b_frz_node_indices] = rand_new_edge_probs(1, b_frz_num_nodes, num_new_edges, zero_prob).squeeze(0)
        b_new_edges_dist_params = adjust_all_zero_rows(b_new_edges_dist_params.unsqueeze(0)).squeeze(0)

        b_transfer_edges_dist_params = torch.zeros([b_flz_num_nodes + b_frz_num_nodes, max_incident_edges, max_incident_edges + 1])
        b_transfer_edges_dist_params[b_frz_node_indices] = rand_transfer_edge_probs(1, b_frz_num_nodes, max_incident_edges).squeeze(0)
        b_transfer_edges_dist_params = adjust_all_zero_matrices(b_transfer_edges_dist_params.unsqueeze(0)).squeeze(0)

        mixture_dist_params.append(b_mixture_dist_params)
        frz_node_dist_params.append(b_frz_node_dist_params)
        flz_node_dist_params.append(b_flz_node_dist_params)
        phase_dist_params.append(b_phase_dist_params)
        new_edges_dist_params.append(b_new_edges_dist_params)
        transfer_edges_dist_params.append(b_transfer_edges_dist_params)

    mixture_dist_params = pad_and_stack_tensors(mixture_dist_params)
    frz_node_dist_params = pad_and_stack_tensors(frz_node_dist_params)
    flz_node_dist_params = pad_and_stack_tensors(flz_node_dist_params)
    phase_dist_params = adjust_all_zero_rows(pad_and_stack_tensors(phase_dist_params))
    new_edges_dist_params = adjust_all_zero_rows(pad_and_stack_tensors(new_edges_dist_params))
    transfer_edges_dist_params = adjust_all_zero_matrices(pad_and_stack_tensors(transfer_edges_dist_params))

    return (mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params,
            transfer_edges_dist_params)
