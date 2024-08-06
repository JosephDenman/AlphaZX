import torch
import torch_geometric as pyg
import gymnasium as gym

from alphazx.diagram import METADATA, POSSIBLE_PHASES, clifford_zx_diagram, to_zx_match_diagram
from alphazx.distributions import AlphaZXDistributionParams, AlphaZXDistribution
from alphazx.game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import is_all_zero, pre_process
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


def check_model_consistency(batch: pyg.data.Batch,
                            azx_dist_params: AlphaZXDistributionParams,
                            num_samples: int = None,
                            sampled_rewrite_types_batch: torch.Tensor = None,
                            sampled_nodes_batch: torch.Tensor = None,
                            action: torch.Tensor = None) -> None:
    # Where mixture parameters are non-zero, there should be the same number of nodes in the data.
    # Only mixture parameters for match nodes should be present.
    # All non-zero node rows should have the same number of nodes of that type in the graph
    # All zero node rows should have no nodes of that type in the graph
    # The last dimension of the node parameters batch should be as long as the maximum number of nodes of all graphs in the batch
    # All non-zero edge parameter rows should have

    B = batch.batch_size

    for params in list(azx_dist_params):
        assert params.shape[0] == B, f'Expected batch dimension of parameters {params} to be {B}'

    batch_list = batch.to_data_list()
    for b_idx in range(len(batch_list)):
        data = batch_list[b_idx]
        x, edge_index, edge_attr, node_type = data.x, data.edge_index, data.edge_attr, data.node_type

        mixture_dist_probs = azx_dist_params.mixture_dist_probs[b_idx]
        node_dist_probs = azx_dist_params.node_dist_probs[b_idx]

        for mixture_component in range(len(mixture_dist_probs)):
            mixture_prob = mixture_dist_probs[mixture_component].item()
            if (mixture_component >= 12) and (mixture_component <= 21) or mixture_component == 0:
                expected_num_nodes_for_component = torch.sum(node_type == mixture_component)
                if mixture_component == 0:
                    assert mixture_prob == 0., (f'Expected mixture prob to be zero'
                                                f'\nmixture_component = {mixture_component}'
                                                f'\nmixture_prob = {mixture_prob}'
                                                f'\nnode_types = {node_type}')
                else:
                    num_nodes_for_component = torch.sum(node_type == mixture_component - 11)
                    if num_nodes_for_component > 0:
                        assert expected_num_nodes_for_component == 1, (
                            f'Expected one super node, got {expected_num_nodes_for_component}'
                            f'\nmixture_component = {mixture_component}'
                            f'\nmixture_prob = {mixture_prob}'
                            f'\nnode_types = {node_type}')
                        assert mixture_prob != 0., (f'Expected mixture prob to be non-zero'
                                                    f'\nmixture_component = {mixture_component}'
                                                    f'\nmixture_prob = {mixture_prob}'
                                                    f'\nnode_types = {node_type}')
                    else:
                        assert expected_num_nodes_for_component == 0, (
                            f'Expected zero super nodes, got {expected_num_nodes_for_component}'
                            f'\nmixture_component = {mixture_component}'
                            f'\nmixture_prob = {mixture_prob}'
                            f'\nnode_types = {node_type}')
                        assert mixture_prob == 0., (f'Expected mixture prob to be zero'
                                                    f'\nmixture_component = {mixture_component}'
                                                    f'\nmixture_prob = {mixture_prob}'
                                                    f'\nnode_types = {node_type}')
            else:
                assert mixture_prob == 0., f'Expected mixture prob to be zero, got {mixture_prob}\nnode_types ={node_type}'

        for mixture_component in range(len(node_dist_probs)):
            node_dist_probs_row = node_dist_probs[mixture_component][:len(node_type)]
            if (mixture_component >= 1) & (mixture_component <= 10):
                num_nodes_for_component = torch.sum(node_type == mixture_component)
                node_dist_probs_row_for_component = node_dist_probs_row[node_type == mixture_component]
                summed_node_dist_probs_row_for_component = torch.sum(node_dist_probs_row_for_component)
                if num_nodes_for_component > 0:
                    assert torch.isclose(summed_node_dist_probs_row_for_component, torch.tensor(1.)), (
                        f'Expected node probs to be normalized\n'
                        f'\nmixture_component = {mixture_component}'
                        f'\nsummed_node_dist_probs_row_for_component = {summed_node_dist_probs_row_for_component}'
                        f'\nnum_nodes_for_component = {num_nodes_for_component}'
                        f'\nnode_dist_probs_row = {node_dist_probs_row}'
                        f'\nnode_dist_probs_row_for_component = {node_dist_probs_row_for_component}'
                        f'\nnode_types = {node_type}')
                else:
                    assert summed_node_dist_probs_row_for_component == 0., (
                        f'Expected node probs to be all zero\n'
                        f'\nmixture_component = {mixture_component}'
                        f'\nnode_dist_probs_row = {node_dist_probs_row}'
                        f'\nnode_dist_probs_row_for_component = {node_dist_probs_row_for_component}'
                        f'\nnode_types = {node_type}')
                actual_non_zero_node_prob_idxs = node_dist_probs_row != 0.
                expected_non_zero_node_prob_idxs = node_type == mixture_component
                assert torch.equal(actual_non_zero_node_prob_idxs, expected_non_zero_node_prob_idxs), (
                    f'Expected node probs {actual_non_zero_node_prob_idxs} to be {expected_non_zero_node_prob_idxs}'
                    f'\nmixture_component = {mixture_component}'
                    f'\nactual_non_zero_node_prob_idxs = {actual_non_zero_node_prob_idxs}'
                    f'\nexpected_non_zero_node_prob_idxs = {expected_non_zero_node_prob_idxs}'
                    f'\nnode_type = {node_type}')
            else:
                assert is_all_zero(
                    node_dist_probs_row), f'Expected node probs for type {mixture_component} to be all zero, got {node_dist_probs_row}\nnode_type = {node_type}'

    def batch_list_string():
        s = ''
        for b in batch_list:
            s = s + f'\n{b.to_dict()}'
        return s


    if sampled_rewrite_types_batch is not None:
        SA_B, SA_A = sampled_rewrite_types_batch.shape
        assert SA_B == B, f'Expected sample actions and data batch sizes to be equal\nB = {B}\nSA_B = {SA_B}'
        if num_samples is not None:
            assert SA_A == num_samples, f'Expected {num_samples} in sampled actions {sampled_rewrite_types_batch}, got {SA_A}'
        for b_idx in range(SA_B):
            sampled_rewrite_types = sampled_rewrite_types_batch[b_idx]
            for sampled_rewrite_type in sampled_rewrite_types:
                rewrite_probs = azx_dist_params.mixture_dist_probs[b_idx]
                rewrite_prob = rewrite_probs[sampled_rewrite_type]
                assert rewrite_prob != 0., (f'Expected non-zero rewrite probability'
                                            f'\nbatch = {b_idx}'
                                            f'\nrewrite_probs = {rewrite_probs}'
                                            f'\nrewrite_prob = {rewrite_prob}'
                                            f'\nsampled_rewrite_type = {sampled_rewrite_type}'
                                            f'\n'
                                            f'\nParams:'
                                            f'\ngraph_id = {azx_dist_params.graph_ids[b_idx]}'
                                            f'\nmixture_dist_probs = {azx_dist_params.mixture_dist_probs[b_idx]}'
                                            f'\nnode_dist_probs = {azx_dist_params.node_dist_probs[b_idx]}'
                                            f'\nphase_dist_probs = {azx_dist_params.phase_dist_probs[b_idx]}'
                                            f'\nnew_edge_dist_probs = {azx_dist_params.new_edge_dist_probs[b_idx]}'
                                            f'\ntransfer_edge_dist_probs = {azx_dist_params.transfer_edge_dist_probs[b_idx]}'
                                            f'\n'
                                            f'\nBatch:'
                                            f'\n{batch_list[b_idx].to_dict()}'
                                            f'\n'
                                            f'\nAction:'
                                            f'\n{action[b_idx]}')

    pyg.data.Batch({})
    if sampled_nodes_batch is not None:
        SA_B, SA_A = sampled_nodes_batch.shape
        assert SA_B == B, f'Expected sample actions and data batch sizes to be equal\nB = {B}\nSA_B = {SA_B}'
        if num_samples is not None:
            assert SA_A == num_samples, f'Expected {num_samples} in sampled actions {sampled_nodes_batch}, got {SA_A}'
        for b_idx in range(SA_B):
            sampled_rewrite_types = sampled_rewrite_types_batch[b_idx]
            sampled_nodes = sampled_nodes_batch[b_idx]
            for sampled_rewrite_type in sampled_rewrite_types:
                for sampled_node in sampled_nodes:
                    node_probs = azx_dist_params.node_dist_probs[b_idx][sampled_rewrite_type - 11]
                    node_prob = node_probs[sampled_node]
                    assert node_prob != 0., (f'Expected non-zero node probability'
                                             f'\nbatch = {b_idx}'
                                             f'\nsampled_rewrite_type = {sampled_rewrite_type}'
                                             f'\nnode_probs = {node_probs}'
                                             f'\nnode_prob = {node_prob}'
                                             f'\nsampled_node = {sampled_node}'
                                             f'\nrewrite_dist_params = {azx_dist_params.mixture_dist_probs[b_idx]}'
                                             f'\nnode_dist_params = {azx_dist_params.node_dist_probs[b_idx]}')


# num_batches = 32
# batch_size = 16
# num_qubits = 10
# depth = 10
# batch_list = create_batch_list(num_batches, batch_size, num_qubits, depth)
# print('created batch')
# for i, b in enumerate(batch_list):
#     print(f'processing batch {i}')
#     b = b.sort(False)
#     print('sorted batch')
#     b = pre_process(b, pe_dim)
#     print('pre-processed batch')
#     azx_dist_params, value = model(b.x, b.edge_index, b.edge_attr, b.node_type, b.batch, b.pe)
#     print('ran model')
#     # print('azx_dist_params.mixture_dist_params = ', azx_dist_params.mixture_dist_probs)
#     # print('azx_dist_params.node_dist_params = ', azx_dist_params.node_dist_probs)
#     check_model_consistency(b, azx_dist_params)
#     print('checked consistency')
#     azx_dist = AlphaZXDistribution(azx_dist_params)
#     # sampled_actions = azx_dist.sample(1)
#     # print('sampled_actions = ', sampled_actions)
#     # print('azx_dist_param = ', azx_dist_params)
#     sampled_rewrite_types = azx_dist.sample_rewrite_types(1)
#     # print('sampled_rewrite_types = ', sampled_rewrite_types)
#     # check_model_consistency(b, azx_dist_params, num_samples=1, sampled_rewrite_types_batch=sampled_rewrite_types)
#     sampled_nodes = azx_dist.sample_nodes(sampled_rewrite_types)
#     check_model_consistency(b, azx_dist_params, 1, sampled_rewrite_types, sampled_nodes)
#     # print('sampled_nodes = ', sampled_nodes)
#     # check_model_consistency(b, azx_dist_params, num_samples=1, sampled_rewrite_types_batch=sampled_rewrite_types)
