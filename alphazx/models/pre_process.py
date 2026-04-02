import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj, degree

from alphazx.diagram.match import METADATA


def with_laplacian_pe(data: pyg.data.Data, pe_dimension: int) -> pyg.data.Data:
    return pyg.transforms.AddLaplacianEigenvectorPE(k=pe_dimension, attr_name='pe', is_undirected=data.is_undirected())(
        data)


def with_random_walk_pe(data: pyg.data.Batch, walk_length: int) -> pyg.data.Batch:
    """Custom random walk PE that uses dense operations to avoid MKL sparse issues on macOS."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    batch = data.batch if hasattr(data, 'batch') else torch.zeros(num_nodes, dtype=torch.long)

    # Compute row-normalized adjacency matrix (transition matrix) using dense operations
    # Get degree for normalization
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = 1.0 / deg.clamp(min=1)  # Avoid division by zero

    # Create dense adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [num_nodes, num_nodes]

    # Row-normalize to get transition matrix
    transition = adj * deg_inv.unsqueeze(1)

    # Compute random walk PE: powers of transition matrix
    pe = torch.zeros(num_nodes, walk_length, device=edge_index.device, dtype=torch.float)

    # Start with identity (walk of length 0 returns to self with prob 1)
    walk = torch.eye(num_nodes, device=edge_index.device, dtype=torch.float)

    for k in range(walk_length):
        walk = walk @ transition
        pe[:, k] = walk.diag()  # Probability of returning to starting node

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


def pre_process(data: pyg.data.Batch, pe_dimension: int) -> pyg.data.Batch:
    """Pre-process a batch by processing each graph individually, then re-batching.

    This ensures pe attributes are properly preserved when later splitting/re-batching.
    """
    # Split into individual Data objects
    data_list = data.to_data_list()

    # Process each individually
    processed_list = [pre_process_single(d, pe_dimension) for d in data_list]

    # Re-batch
    return pyg.data.Batch.from_data_list(processed_list)
