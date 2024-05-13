import torch
import torch_geometric as pyg
from torch_geometric.nn.to_hetero_with_bases_transformer import split_output
from torch_geometric.typing import NodeType

from alphazx.diagram.match import FRightZMatch, FRightXMatch


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
    x_[~mask] = -torch.inf
    return x_


def concatenate_with_neighbor_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    neighbor_x = concatenate_neighbor_features(x, edge_index)
    x_ = torch.cat([x.unsqueeze(dim=1), neighbor_x], dim=1)
    return x_


def split_features(x: torch.Tensor, offsets: dict[NodeType, int]) -> dict[NodeType, torch.Tensor]:
    return split_output(x, offsets)


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


def mask_edges_by_type(edge_index: torch.Tensor, node_types: torch.Tensor, node_types_to_mask: torch.Tensor) -> torch.Tensor:
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
    # Get the node types for the source and target nodes of each edge
    source_types = node_types[edge_index[0]]
    target_types = node_types[edge_index[1]]
    # Create a mask to determine which edges have node types to mask
    mask_source = source_types.unsqueeze(1) == node_types_to_mask.unsqueeze(0)
    mask_target = target_types.unsqueeze(1) == node_types_to_mask.unsqueeze(0)
    # Determine the final mask by checking if any source or target node needs to be masked
    mask_edges = mask_source.any(dim=1) | mask_target.any(dim=1)
    # Use the mask to filter out edges
    filtered_edges = edge_index[:, ~mask_edges]
    return filtered_edges


def mask_batch_by_type(batch: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    return batch[(node_types == FRightZMatch.index) | (node_types == FRightXMatch.index)]
