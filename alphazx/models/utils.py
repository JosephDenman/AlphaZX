import torch
import torch_geometric as pyg
from torch import index_select
from torch_geometric.typing import NodeType

from alphazx.diagram.zx_match_diagram import ZXMatchDiagram


def concatenate_by_group(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = torch.zeros_like(x[0]).repeat(fill_count.sum(), *([1]*(len(x.shape)-1)))
    fill_index = torch.arange(0, fill_count.shape[0]).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max(), *x.shape[1:])
    return x_


def concatenate_neighbor_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    neighbor_x = index_select(x, 0, edge_index[0])
    x_ = concatenate_by_group(neighbor_x, edge_index[1])
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


def get_expected_node_features(zx_match_diagram: ZXMatchDiagram, data: pyg.data.Data) -> list[list[list[float]]] | \
                                                                                         dict[NodeType, list[
                                                                                             list[float]]]:
    indices = data['node_ids']
    match_to_index = {}
    original_nodes = list(zx_match_diagram.nodes(data=True))
    if isinstance(indices, dict):
        expected_features_dict = {}
        for ntype, indices in indices.items():
            expected_features = []
            for node_index in indices.tolist():
                n, ndata = original_nodes[node_index]
                match_to_index[n] = node_index
            for node_index in indices.tolist():
                n, ndata = original_nodes[node_index]
                pairs = []
                for m in zx_match_diagram.neighbors(n):
                    pairs.append((m, match_to_index[m]))
                pairs = sorted(pairs, key=lambda x: x[1])
                ordered_neighbor_ndata = [zx_match_diagram.nodes[m] for m, _ in pairs]
                expected_neighbor_features = [[ndata['type'], ndata['phase'].item()] for ndata in
                                              ordered_neighbor_ndata]
                expected_features.append(expected_neighbor_features)
            expected_features_dict[ntype] = expected_features
    else:
        expected_features = []
        for node_index in indices.tolist():
            n, ndata = original_nodes[node_index]
            match_to_index[n] = node_index
        for node_index in indices.tolist():
            n, ndata = original_nodes[node_index]
            pairs = []
            for m in zx_match_diagram.neighbors(n):
                pairs.append((m, match_to_index[m]))
            pairs = sorted(pairs, key=lambda x: x[1])
            ordered_neighbor_ndata = [zx_match_diagram.nodes[m] for m, _ in pairs]
            expected_neighbor_features = [[ndata['type'], ndata['phase'].item()] for ndata in ordered_neighbor_ndata]
            expected_features.append(expected_neighbor_features)
        return expected_features
