import torch_geometric as pyg

from alphazx.diagram import NODE_METADATA, POSSIBLE_PHASES, clifford_pyg_zx_match_diagram
from alphazx.models import with_embeddable_feats, with_laplacian_pe
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

num_node_types = len(NODE_METADATA)
num_possible_phases = len(POSSIBLE_PHASES)
num_possible_new_edges = 10
node_embedding_channels = (1 + len(POSSIBLE_PHASES) + 10)
repr_gps_channels = (1 + len(POSSIBLE_PHASES) + 10 + 2)
repr_gps_edge_in_channels = 2
repr_gps_edge_out_channels = 2
repr_gps_pe_in_channels = 2
repr_gps_pe_out_channels = 2
repr_gps_num_layers = 5
repr_gps_bias = False
repr_gps_num_attn_heads = 1
repr_gps_attn_type = 'multihead'
repr_gps_attn_kwargs = {}
repr_gps_mlp_hidden_channels = 64
policy_num_pooling_encoder_blocks = 4
policy_num_pooling_heads = 3
policy_pooling_layer_norm = True
policy_pooling_dropout = 0.1
value_gmt_num_encoder_blocks = 4
value_gmt_num_heads = 3
value_gmt_layer_norm = True
value_gmt_dropout = 0.1

model = AlphaZXModel(num_node_types,
                     num_possible_phases,
                     num_possible_new_edges,
                     node_embedding_channels,
                     repr_gps_channels,
                     repr_gps_edge_in_channels,
                     repr_gps_edge_out_channels,
                     repr_gps_pe_in_channels,
                     repr_gps_pe_out_channels,
                     repr_gps_num_layers,
                     repr_gps_bias,
                     repr_gps_num_attn_heads,
                     repr_gps_attn_type,
                     repr_gps_attn_kwargs,
                     repr_gps_mlp_hidden_channels,
                     policy_num_pooling_encoder_blocks,
                     policy_num_pooling_heads,
                     policy_pooling_layer_norm,
                     policy_pooling_dropout,
                     value_gmt_num_encoder_blocks,
                     value_gmt_num_heads,
                     value_gmt_layer_norm,
                     value_gmt_dropout)


def create_data_loader(num_diagrams: int, batch_size: int, num_qubits: int, depth: int) -> pyg.loader.DataLoader:
    dataset = []
    for _ in range(num_diagrams):
        d = clifford_pyg_zx_match_diagram(num_qubits, depth)
        d = with_embeddable_feats(d)
        d = with_laplacian_pe(d, 2)
        dataset.append(d)
    return pyg.loader.DataLoader(dataset, batch_size)


num_diagrams = 8
batch_size = 4
num_qubits = 10
depth = 10
dataloader = create_data_loader(num_diagrams, batch_size, num_qubits, depth)
for batch in dataloader:
    batch = batch.sort(False)
    azx_dist_params, value = model(batch)