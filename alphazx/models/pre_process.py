import torch
import torch_geometric as pyg

from alphazx.diagram.match import NODE_FEATURE_PAIR_TO_INDEX_METADATA


def with_laplacian_pe(data: pyg.data.Data, pe_dimension: int) -> pyg.data.Data:
    return pyg.transforms.AddLaplacianEigenvectorPE(k=pe_dimension, attr_name='pe', is_undirected=data.is_undirected())(data)


def with_embeddable_feats(data: pyg.data.Data) -> pyg.data.Data:
    feature_idxs = []
    for feature in data.x:
        match_idx = int(feature[0].item())
        phase = feature[1].item()
        feature_idxs.append(NODE_FEATURE_PAIR_TO_INDEX_METADATA[(match_idx, phase)])
    data.x = torch.tensor(feature_idxs)
    return data


# TODO: Implement
def with_one_hot_types(data: pyg.data.Data) -> pyg.data.Data:
    pass
