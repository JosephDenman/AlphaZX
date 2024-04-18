import torch


def cat_aggregate(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    print('x = ', x)
    print('index = ', index)
    max_num_elements = torch.max(torch.bincount(index))
    print('max_num_elements = ', max_num_elements)
    # Number of groups
    num_groups = index.max().item() + 1
    # Sort index and x in order of index
    sorted_indices, sorted_idx = index.sort()
    sorted_x = x[sorted_idx]
    # Calculate sizes and maximum group size
    group_sizes = torch.zeros(num_groups, dtype=torch.long, device=x.device)
    group_sizes.scatter_add_(0, sorted_indices, torch.ones_like(sorted_indices))
    # Prepare the output tensor, padded with zeros
    result = torch.zeros((num_groups, max_num_elements), dtype=x.dtype, device=x.device)
    # Calculate group start indices and place values
    start_indices = torch.zeros(num_groups, dtype=torch.long, device=x.device).scatter_(
        0, torch.arange(num_groups, device=x.device),
        torch.cat([torch.tensor([0], device=x.device), group_sizes[:-1].cumsum(0)])
    )
    flat_indices = (sorted_indices * max_num_elements + torch.arange(sorted_x.size(0), device=x.device) - start_indices[
        sorted_indices])
    print('flat_indices = ', flat_indices)
    print('sorted_x = ', sorted_x)
    result.view(-1).scatter_(0, flat_indices, sorted_x)
    return result
