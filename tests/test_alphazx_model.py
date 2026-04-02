import torch
import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES, clifford_zx_diagram, to_zx_match_diagram
from alphazx.distributions import AlphaZXDistributionParams
from alphazx.game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import is_all_zero
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

torch.set_default_dtype(torch.float64)
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

model = AlphaZXModel(num_node_types,
                     num_possible_phases,
                     num_possible_new_edges,
                     repr_gps_node_embedding_out_channels,
                     repr_gps_num_edge_embeddings,
                     repr_gps_edge_embedding_out_channels,
                     repr_gps_pe_in_channels,
                     repr_gps_pe_out_channels)


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


def check_model_consistency(batch: pyg.data.Batch, azx_dist_params: AlphaZXDistributionParams, num_samples: int = None,
                            sampled_rewrite_types_batch: torch.Tensor = None) -> None:
    # Where mixture parameters are non-zero, there should be the same number of nodes in the data.
    # Only mixture parameters for match nodes should be present.
    # All non-zero node rows should have the same number of nodes of that type in the graph
    # All zero node rows should have no nodes of that type in the graph
    # The last dimension of the node parameters batch should be as long as the maximum number of nodes of all graphs in the batch
    # All non-zero edge parameter rows should have
    B = batch.batch_size

    for params in list(azx_dist_params):
        assert params.shape[0] == B, f'Expected batch dimension of parameters {params} to be {B}'

    batch = batch.to_data_list()
    for b_idx in range(len(batch)):
        data = batch[b_idx]
        x, edge_index, edge_attr, node_type = data.x, data.edge_index, data.edge_attr, data.node_type

        mixture_dist_probs = azx_dist_params.mixture_dist_probs[b_idx]
        for md_idx in range(len(mixture_dist_probs)):
            mixture_component = mixture_dist_probs[md_idx].item()
            expected_num_nodes_for_component = torch.sum(node_type == md_idx).item()
            if (md_idx >= 12) and (md_idx <= 21) or mixture_component != 0:
                if mixture_component != 0:
                    assert expected_num_nodes_for_component != 0, f'Expected number of nodes for mixture component {md_idx} with value {mixture_component} to be non-zero\nnode_types ={node_type}'
                else:
                    assert expected_num_nodes_for_component == 0, f'Expected number of nodes for mixture component {md_idx} with value {mixture_component} to be zero, got {expected_num_nodes_for_component}\nnode_types ={node_type}'
            else:
                assert mixture_component == 0., f'Expected mixture component {md_idx} to be zero, got {mixture_component}\nnode_types ={node_type}'

        node_dist_probs = azx_dist_params.node_dist_probs[b_idx]
        for nd_idx in range(len(node_dist_probs)):
            node_dist_probs_row = node_dist_probs[nd_idx][:len(node_type)]
            if (nd_idx >= 1) & (nd_idx <= 10):
                actual_non_zero_node_prob_idxs = node_dist_probs_row != 0.
                expected_non_zero_node_prob_idxs = node_type == nd_idx
                assert torch.equal(actual_non_zero_node_prob_idxs,
                                   expected_non_zero_node_prob_idxs), f'Expected node probs {actual_non_zero_node_prob_idxs} for type {nd_idx} to be {expected_non_zero_node_prob_idxs}\nnode_type = {node_type}'
            else:
                assert is_all_zero(
                    node_dist_probs_row), f'Expected node probs for type {nd_idx} to be all zero, got {node_dist_probs_row}\nnode_type = {node_type}'

    if sampled_rewrite_types_batch is not None:
        assert sampled_rewrite_types_batch.shape[1] == len(
            batch), f'Expected {len(batch)} batches in sampled actions {sampled_rewrite_types_batch} but got {sampled_rewrite_types_batch.shape[0]}'
        if num_samples is not None:
            assert sampled_rewrite_types_batch.shape[
                       0] == num_samples, f'Expected {num_samples} in sampled actions {sampled_rewrite_types_batch} but got {sampled_rewrite_types_batch.shape[1]}'
        for sample_idx in range(sampled_rewrite_types_batch.shape[0]):
            for b_idx in range(sampled_rewrite_types_batch.shape[1]):
                sampled_rewrite_type = sampled_rewrite_types_batch[sample_idx][b_idx]
                rewrite_type_dist_component = azx_dist_params.mixture_dist_probs[b_idx][sampled_rewrite_type]
                assert rewrite_type_dist_component != 0., f'Expected non-zero entry rewrite type component for rewrite type {sampled_rewrite_type} in batch {b_idx}\nazx_dist_params = {azx_dist_params.mixture_dist_probs}'

#
# num_envs = 8
# max_episode_length = 1000
# num_qubits = 10
# depth = 10
# envs = [ZXGame(num_qubits, depth, max_episode_length=max_episode_length, pe_dim=walk_length) for _ in range(num_envs)]
#
# step_data = [env.reset() for env in envs]
# while not any([step_datum[2] for step_datum in step_data]):
#     b = pyg.data.Batch.from_data_list([step_datum[0] for step_datum in step_data])
#     b = pre_process(b, walk_length)
#     azx_dist_params, value = model(b.x, b.edge_index, b.edge_attr, b.node_type, b.batch, b.pe)
#     azx_dist = AlphaZXDistribution(azx_dist_params)
#     sampled_actions = azx_dist.sample(1)
#     print('sampled_actions = ', sampled_actions)
#     probs = torch.exp(azx_dist.log_prob(sampled_actions))
#     print('probs = ', probs)
#     for env_id, action in enumerate(sampled_actions[0]):
#         action = tuple(action.tolist())
#         step_datum = envs[env_id].step(action)
#         data, reward, done, stats = step_datum
#         print('env_id = ', env_id)
#         print('action = ', action)
#         print('reward = ', reward)
#         print('done = ', done)
#         print('stats = ', stats)
#         step_data[env_id] = step_datum
