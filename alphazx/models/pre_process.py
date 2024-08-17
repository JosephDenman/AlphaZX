import torch
import torch_geometric as pyg

from alphazx.diagram.match import METADATA


def with_laplacian_pe(data: pyg.data.Data, pe_dimension: int) -> pyg.data.Data:
    return pyg.transforms.AddLaplacianEigenvectorPE(k=pe_dimension, attr_name='pe', is_undirected=data.is_undirected())(
        data)


def with_random_walk_pe(data: pyg.data.Data, walk_length: int) -> pyg.data.Data:
    return pyg.transforms.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')(data)


def with_embeddable_feats(data: pyg.data.Data) -> pyg.data.Data:
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


def pre_process(data: pyg.data.Data, pe_dimension: int):
    data = with_embeddable_feats(data)
    data = with_random_walk_pe(data, pe_dimension)
    return data
