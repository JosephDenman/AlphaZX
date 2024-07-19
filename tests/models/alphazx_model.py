from random import random

import torch_geometric as pyg
import torch
from alphazx.diagram import METADATA, POSSIBLE_PHASES, clifford_zx_diagram, to_zx_match_diagram
from alphazx.distributions import AlphaZXDistribution, AlphaZXDistributionParams
from alphazx.game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import pre_process
from alphazx.models.homogeneous.mcts.alphazx_model import AlphaZXModel

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(10)
torch.set_printoptions(threshold=60_000)

num_node_types = len(METADATA.node_type_abbrevs)
num_possible_phases = len(POSSIBLE_PHASES)
num_possible_new_edges = 5
pe_dim = 20
repr_gps_node_embedding_out_channels = 64
repr_gps_node_out_channels = 64
repr_gps_num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
repr_gps_edge_embedding_out_channels = 64
repr_gps_edge_out_channels = 64
repr_gps_pe_in_channels = pe_dim
repr_gps_pe_out_channels = pe_dim
repr_gps_num_layers = 2
repr_gps_bias = True
repr_gps_num_attn_heads = 4
repr_gps_attn_type = 'multihead'
repr_gps_attn_kwargs = {}
repr_gps_mlp_hidden_channels = 64
policy_num_pooling_encoder_blocks = 2
policy_num_pooling_heads = 32
policy_pooling_layer_norm = True
policy_pooling_dropout = 0.15
value_gmt_num_encoder_blocks = 2
value_gmt_num_heads = 4
value_gmt_layer_norm = True
value_gmt_dropout = 0.15

model = AlphaZXModel(num_node_types,
                     num_possible_phases,
                     num_possible_new_edges,
                     repr_gps_node_embedding_out_channels,
                     repr_gps_node_out_channels,
                     repr_gps_num_edge_embeddings,
                     repr_gps_edge_embedding_out_channels,
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


def create_batch_list(num_batches: int, batch_size: int, num_qubits: int, depth: int) -> list[pyg.data.Batch]:
    batch_list = []
    for _ in range(num_batches):
        data_list = []
        for _ in range(batch_size):
            d = clifford_zx_diagram(num_qubits, depth)
            remove_isolated_nodes(d)
            remove_self_loop_edges(d)
            remove_isolated_components(d)
            md = to_zx_match_diagram(d).to_pyg_data()
            data_list.append(md)
        batch_list.append(pyg.data.Batch.from_data_list(data_list))
    return batch_list


def check_model_consistency(batch: pyg.data.Batch, azx_dist_params: AlphaZXDistributionParams, sampled_actions: torch.Tensor = None) -> None:
    # Where mixture parameters are non-zero, there should be the same number of nodes in the data.
    # Only mixture parameters for match nodes should be present.
    # All non-zero node rows should have the same number of nodes of that type in the graph
    # All zero node rows should have no nodes of that type in the graph
    # The last dimension of the node parameters batch should be as long as the maximum number of nodes of all graphs in the batch
    # All non-zero edge parameter rows should have
    B = batch.batch_size

    for params in list(azx_dist_params):
        # print('params = ', params)
        assert params.shape[0] == B, f'Expected batch dimension of parameters {params} to be {B}'

    for b_idx in range(B):
        data = batch.get_example(b_idx)
        x, edge_index, edge_attr, node_type, batch_tensor = data.x, data.edge_index, data.edge_attr, data.node_type, data.batch
        mixture_dist_probs = azx_dist_params.mixture_dist_probs[b_idx]
        for md_idx in range(len(mixture_dist_probs)):
            mixture_component = mixture_dist_probs[md_idx]
            num_nodes_for_component = torch.sum(node_type == md_idx)
            if mixture_component != 0:
                assert num_nodes_for_component.item() != 0, f'Expected number of nodes for mixture component {md_idx} with value {mixture_component} to be non-zero, node_types ={node_type}'
            else:
                assert num_nodes_for_component.item() == 0, f'Expected number of nodes for mixture component {md_idx} with value {mixture_component} to be zero, got {num_nodes_for_component}, node_types ={node_type}'
        #
        # node_dist_probs = azx_dist_params.node_dist_probs[b_idx]
        # new_edge_dist_probs = azx_dist_params.new_edge_dist_probs[b_idx]
        # phase_dist_probs = azx_dist_params.phase_dist_probs[b_idx]
        # transfer_edge_dist_probs = azx_dist_params.transfer_edge_dist_probs[b_idx]

    # pass


num_batches = 2
batch_size = 4
num_qubits = 2
depth = 4
batch_list = create_batch_list(num_batches, batch_size, num_qubits, depth)
for batch in batch_list:
    print('batch = ', batch)
    batch = batch.sort(False)
    batch = pre_process(batch, pe_dim)
    azx_dist_params, value = model(batch.x, batch.edge_index, batch.edge_attr, batch.node_type, batch.batch, batch.pe)
    #print('azx_dist_params.mixture_dist_params = ', azx_dist_params.mixture_dist_probs)
    #print('azx_dist_params.node_dist_params = ', azx_dist_params.node_dist_probs)
    check_model_consistency(batch, azx_dist_params)
    azx_dist = AlphaZXDistribution(azx_dist_params)
    sampled_actions = azx_dist.sample(1)
    check_model_consistency(batch, azx_dist_params, sampled_actions)

