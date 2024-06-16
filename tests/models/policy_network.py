import torch_geometric as pyg

from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import NODE_METADATA, POSSIBLE_PHASES
from alphazx.models.homogeneous.mcts.policy_network import PolicyNetwork
from alphazx.models.pre_process import with_embeddable_feats, with_laplacian_pe


def create_data_loader(num_diagrams: int, batch_size: int, num_qubits: int, depth: int) -> pyg.loader.DataLoader:
    dataset = []
    for _ in range(num_diagrams):
        d = clifford_pyg_zx_match_diagram(num_qubits, depth)
        d = with_embeddable_feats(d)
        d = with_laplacian_pe(d, 2)
        dataset.append(d)
    return pyg.loader.DataLoader(dataset, batch_size)


def policy_network():
    # num_diagrams = 2
    # batch_size = 2
    num_node_types = len(NODE_METADATA)
    num_possible_phases = len(POSSIBLE_PHASES)
    num_possible_new_edges = 10
    node_embedding_channels = 1
    num_pooling_encoder_blocks = 2
    num_pooling_heads = 1
    pooling_layer_norm = True
    pooling_dropout = 0.0
    return PolicyNetwork(num_node_types,
                         num_possible_phases,
                         num_possible_new_edges,
                         node_embedding_channels,
                         num_pooling_encoder_blocks,
                         num_pooling_heads,
                         pooling_layer_norm,
                         pooling_dropout)
    # dataloader = create_data_loader(num_diagrams, batch_size, num_qubits, depth)
    # for batch in dataloader:
    #     batch = batch.sort(False)
    #     azx_dist = AlphaZXDistribution(model(batch))
    #     print('samples = ', azx_dist.sample(8))

# policy_network(10, 10)
