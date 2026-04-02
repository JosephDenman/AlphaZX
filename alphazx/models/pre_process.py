import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj, degree

from alphazx.diagram.match import METADATA


def with_laplacian_pe(data: pyg.data.Data, pe_dimension: int) -> pyg.data.Data:
    return pyg.transforms.AddLaplacianEigenvectorPE(k=pe_dimension, attr_name='pe', is_undirected=data.is_undirected())(
        data)


def with_random_walk_pe(data: pyg.data.Batch, walk_length: int) -> pyg.data.Batch:
    """Random walk PE using PyG's built-in sparse AddRandomWalkPE transform.

    This replaces the original dense O(N^2 * walk_length) implementation with
    PyG's sparse implementation, which is O(E * walk_length) — much faster for
    the sparse graphs typical in ZX diagrams.

    Falls back to the dense implementation if the sparse version fails (e.g.
    MKL sparse issues on some macOS configurations).
    """
    try:
        transform = pyg.transforms.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
        data = transform(data)
        # Ensure float32 output
        data.pe = data.pe.float()
        return data
    except Exception:
        # Fallback to dense implementation for environments where sparse ops fail
        return _with_random_walk_pe_dense(data, walk_length)


def _with_random_walk_pe_dense(data: pyg.data.Batch, walk_length: int) -> pyg.data.Batch:
    """Dense fallback for random walk PE (original implementation)."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = 1.0 / deg.clamp(min=1)

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    transition = adj * deg_inv.unsqueeze(1)

    pe = torch.zeros(num_nodes, walk_length, device=edge_index.device, dtype=torch.float)
    walk = torch.eye(num_nodes, device=edge_index.device, dtype=torch.float)

    for k in range(walk_length):
        walk = walk @ transition
        pe[:, k] = walk.diag()

    data.pe = pe
    return data


def with_embeddable_feats(data: pyg.data.Batch) -> pyg.data.Batch:
    node_feature_idxs = []
    for node_feature in data.x:
        match_idx = int(node_feature[0].item())
        phase = node_feature[1].item()
        node_feature_idxs.append(METADATA.node_feat_to_index_dict[(match_idx, phase)])
    data.x = torch.tensor(node_feature_idxs)
    edge_feature_idxs = []
    for edge_feature in data.edge_attr:
        edge_idx = int(edge_feature[0].item())
        size = edge_feature[1].item()
        edge_feature_idxs.append(METADATA.edge_feat_to_index_dict[(edge_idx, size)])
    data.edge_attr = torch.tensor(edge_feature_idxs)
    return data


def pre_process_single(data: pyg.data.Data, pe_dimension: int) -> pyg.data.Data:
    """Pre-process a single Data object."""
    data = with_embeddable_feats_single(data)
    data = with_random_walk_pe(data, pe_dimension)
    return data


def with_embeddable_feats_single(data: pyg.data.Data) -> pyg.data.Data:
    """Process embeddable features for a single Data object."""
    node_feature_idxs = []
    for node_feature in data.x:
        match_idx = int(node_feature[0].item())
        phase = node_feature[1].item()
        node_feature_idxs.append(METADATA.node_feat_to_index_dict[(match_idx, phase)])
    data.x = torch.tensor(node_feature_idxs)
    edge_feature_idxs = []
    for edge_feature in data.edge_attr:
        edge_idx = int(edge_feature[0].item())
        size = edge_feature[1].item()
        edge_feature_idxs.append(METADATA.edge_feat_to_index_dict[(edge_idx, size)])
    data.edge_attr = torch.tensor(edge_feature_idxs)
    return data


def pre_process(data, pe_dimension: int):
    """Pre-process graph data by processing each graph individually.

    Accepts either a single Data object or a Batch. For a single Data object,
    delegates directly to pre_process_single. For a Batch, splits into individual
    Data objects, processes each, and re-batches.
    """
    if isinstance(data, pyg.data.Batch):
        # Split into individual Data objects
        data_list = data.to_data_list()
        processed_list = [pre_process_single(d, pe_dimension) for d in data_list]
        return pyg.data.Batch.from_data_list(processed_list)
    else:
        # Single Data object
        return pre_process_single(data, pe_dimension)
