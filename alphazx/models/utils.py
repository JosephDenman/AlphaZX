import torch


def cat_aggregate(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Number of groups and the number of features in each row of x
    num_groups = index.max().item() + 1
    num_features = x.size(1)
    # Compute the maximum number of elements in any group
    group_sizes = torch.zeros(num_groups, dtype=torch.long, device=x.device)
    group_sizes.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
    # Prepare the output tensor, padded with zeros
    max_num_elements = group_sizes.max()
    result = torch.zeros(num_groups, max_num_elements, num_features, dtype=x.dtype, device=x.device)
    # Positions to fill in the result tensor
    positions = group_sizes.clone().fill_(0)  # Current fill position in each group
    # Fill the tensor
    for i in range(x.size(0)):
        group_id = index[i]
        result[group_id, positions[group_id]] = x[i]
        positions[group_id] += 1
    return result
