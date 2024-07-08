import torch
import torch_geometric as pyg

from alphazx.diagram.match import METADATA
torch.set_printoptions(threshold=10_000)


def concatenate_by_group(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = torch.full_like(x[0], -torch.inf).repeat(fill_count.sum(), *([1] * (len(x.shape) - 1)))
    fill_index = torch.arange(0, fill_count.shape[0]).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
    return x_


def concatenate_neighbor_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    neighbor_x = torch.index_select(x, 0, edge_index[0])
    x_, mask = pyg.utils.to_dense_batch(neighbor_x, edge_index[1])
    row_mask = torch.any(mask, dim=1)
    x_ = x_[row_mask]
    mask = mask[row_mask]
    return x_


def concatenate_with_neighbor_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    neighbor_x = concatenate_neighbor_features(x, edge_index)
    x_ = torch.cat([x.unsqueeze(dim=1), neighbor_x], dim=1)
    return x_


def pad_and_stack(tensors: list[torch.Tensor], pad_value=0.) -> torch.Tensor:
    if not tensors:
        raise ValueError("The input list of tensors cannot be empty.")
    # Identify common device - assuming all tensors are on the same device or moving them to a common device
    common_device = tensors[0].device
    # Calculate the maximum number of dimensions and their maximum size
    max_dims = max(len(tensor.shape) for tensor in tensors)
    max_shape = [max([tensor.shape[i] if i < len(tensor.shape) and tensor.numel() > 0 else 0 for tensor in tensors]) for
                 i in range(max_dims)]
    # Pad each tensor to the maximum shape
    padded_tensors = []
    for tensor in tensors:
        tensor = tensor.to(common_device)  # Ensure tensor is on the common device
        if tensor.numel() == 0:  # Check if the tensor is empty
            # Create a new tensor of the max shape with the padding value on the correct device
            new_shape = [max_shape[i] for i in range(max_dims)]
            padded_tensor = torch.full(new_shape, pad_value, dtype=tensor.dtype, device=common_device)
        else:
            pad_sizes = [(0, max_shape[i] - tensor.shape[i]) if i < len(tensor.shape) else (0, max_shape[i]) for i in
                         reversed(range(max_dims))]
            pad_sizes_flat = [item for sublist in pad_sizes for item in sublist]  # Flatten the list of tuples
            padded_tensor = torch.nn.functional.pad(tensor, pad_sizes_flat, 'constant', pad_value)
        padded_tensors.append(padded_tensor)
    # Stack all the padded tensors
    result = torch.stack(padded_tensors)
    return result


def mask_non_basis_edges(edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    print('torch.tensor(METADATA.non_basis_node_type_indices) = ', torch.tensor(METADATA.non_basis_node_type_indices))
    return mask_edges_by_type(edge_index, node_types, torch.tensor(METADATA.non_basis_node_type_indices))


def mask_non_super_nodes(x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.full_like(node_types, False, dtype=torch.bool)
    for super_node_index in METADATA.super_node_type_indices:
        mask = mask | (node_types == super_node_index)
    masked_x = x[mask]
    masked_edge_index = mask_edges_by_type(edge_index, node_types, torch.tensor(METADATA.non_super_node_type_indices))
    masked_node_types = node_types[mask]
    masked_batch = batch[mask]
    return masked_x, masked_edge_index, masked_node_types, masked_batch


def mask_non_basis_nodes(x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.full_like(node_types, False, dtype=torch.bool)
    for super_node_index in METADATA.basis_node_type_indices:
        mask = mask | (node_types == super_node_index)
    masked_x = x[mask]
    masked_edge_index = mask_edges_by_type(edge_index, node_types, torch.tensor(METADATA.non_super_node_type_indices))
    masked_node_types = node_types[mask]
    masked_batch = batch[mask]
    return masked_x, masked_edge_index, masked_node_types, masked_batch


def mask_edges_by_type(edge_index: torch.Tensor, node_types: torch.Tensor,
                       node_types_to_mask: torch.Tensor) -> torch.Tensor:
    """
    Masks out all columns of `edge_index` which contain a node with a type in `node_types`.
    :param edge_index: Two-dimensional 2 x N tensor where N is the number of nodes in the graph. Each column corresponds
                       to an edge with the top item of the column being the source node of the edge the bottom item of the
                       column being the target node of the edge.
    :param node_types: One-dimensional N-length tensor where each item represents the type of the node.
    :param node_types_to_mask: One-dimensional L-length tensor containing the node types used to mask out `edge_index`.
    :return: Two-dimensional 2 x M tensor which is identical to `edge_index` except that every column in which the bottom
             or top item has a node type in `node_types_to_mask` is omitted.
    """
    column_mask = compute_column_mask_for_values(edge_index_as_node_types(edge_index, node_types), node_types_to_mask)
    filtered_edge_index = edge_index[:, column_mask]
    return filtered_edge_index


def edge_index_as_node_types(edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    src_node_types = torch.index_select(node_types, 0, edge_index[0])
    tgt_node_types = torch.index_select(node_types, 0, edge_index[1])
    edge_indexed_node_types = torch.stack([src_node_types, tgt_node_types], dim=0)
    # print('edge_indexed_node_types = ', edge_indexed_node_types)
    return edge_indexed_node_types


def compute_column_mask_for_values(t: torch.Tensor, values_to_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes a mask indicating which columns of `t` have no elements of `values_to_mask`.

    Parameters:
        t (torch.Tensor): The input tensor from which columns are to be removed.
        values_to_mask (torch.Tensor): A 1D tensor of values based on which columns will be removed.

    Returns:
        torch.Tensor: A mask indicating which columns of `t` should be kept.
    """
    assert len(t.shape) == 2, f"Expected `t` to be two-dimensional, got ${t.shape}."
    assert len(
        values_to_mask.shape) == 1, f"Expected `values_to_mask` to be one-dimensional, got ${values_to_mask.shape}."
    if len(values_to_mask) == 0:
        return t
    # Check each element if it is in `values` and create a mask
    # This results in a mask of shape [rows, columns, values.size(0)]
    mask = t.unsqueeze(-1) == values_to_mask.unsqueeze(0).unsqueeze(0)
    # Reduce across the last dimension to see if any value matches in the values tensor
    # This collapses the mask to [rows, columns], being True where any match was found
    any_match = mask.any(dim=-1)
    # Use `all()` along the rows (dim=0) to find columns where no element matches any of the values
    valid_columns = (~any_match).all(dim=0)
    # Return the mask
    return valid_columns


def compute_basis_neighbors(edge_index: torch.Tensor, node: int, node_types: torch.Tensor) -> torch.Tensor:
    edge_index = mask_non_basis_edges(edge_index, node_types)
    return edge_index[0][edge_index[1] == node]


def throw_on_nan(x: torch.Tensor) -> None:
    if torch.isnan(x).any():
        raise Exception(f'Input tensor {x} contains NaN values')
    if (x == torch.inf).any():
        raise Exception(f'Input tensor {x} contains NaN values')
    if (x == -torch.inf).any():
        raise Exception(f'Input tensor {x} contains NaN values')
