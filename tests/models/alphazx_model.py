import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES, clifford_pyg_zx_match_diagram, clifford_pyg_zx_diagram, \
    clifford_zx_diagram, to_zx_match_diagram
from alphazx.distributions import AlphaZXDistribution
from alphazx.game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import with_embeddable_feats, with_laplacian_pe, assert_unique_elements
from alphazx.models.homogeneous.mcts.alphazx_model import AlphaZXModel


num_node_types = len(METADATA.node_type_abbrevs)
num_possible_phases = len(POSSIBLE_PHASES)
num_possible_new_edges = 10
pe_dim = 40
repr_gps_embedding_out_channels = 8
repr_gps_node_out_channels = 64
repr_gps_edge_in_channels = 2
repr_gps_edge_out_channels = 8
repr_gps_pe_in_channels = pe_dim
repr_gps_pe_out_channels = pe_dim
repr_gps_num_layers = 5
repr_gps_bias = True
repr_gps_num_attn_heads = 4
repr_gps_attn_type = 'multihead'
repr_gps_attn_kwargs = {}
repr_gps_mlp_hidden_channels = 64

policy_num_pooling_encoder_blocks = 4
policy_num_pooling_heads = 4
policy_pooling_layer_norm = True
policy_pooling_dropout = 0.1
value_gmt_num_encoder_blocks = 4
value_gmt_num_heads = 4
value_gmt_layer_norm = True
value_gmt_dropout = 0.1

model = AlphaZXModel(num_node_types,
                     num_possible_phases,
                     num_possible_new_edges,
                     repr_gps_embedding_out_channels,
                     repr_gps_node_out_channels,
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
    for i in range(num_diagrams):
        d = clifford_zx_diagram(num_qubits, depth)
        remove_isolated_nodes(d)
        remove_self_loop_edges(d)
        remove_isolated_components(d)
        md = to_zx_match_diagram(d).to_pyg_data()
        md = with_embeddable_feats(md)
        dataset.append(md)
    return pyg.loader.DataLoader(dataset, batch_size)


num_diagrams = 8
batch_size = 4
num_qubits = 10
depth = 10
data_loader = create_data_loader(num_diagrams, batch_size, num_qubits, depth)
for batch in data_loader:
    batch = batch.sort(False)
    batch = with_laplacian_pe(batch, pe_dim)
    azx_dist_params, value = model(batch)
    azx_dist = AlphaZXDistribution(azx_dist_params)
    print(azx_dist.sample(1))
