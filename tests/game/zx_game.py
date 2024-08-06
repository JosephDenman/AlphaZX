import gym.vector
import torch
import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.distributions import AlphaZXDistribution
from alphazx.game import ZXGame
from alphazx.models import pre_process
from alphazx.models.homogeneous.mcts.alphazx_model import AlphaZXModel
from tests.models.alphazx_model import check_model_consistency

torch.set_default_tensor_type(torch.DoubleTensor)

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

num_envs = 8
max_episode_length = 100
num_qubits = 10
depth = 10
zx_env = gym.vector.SyncVectorEnv(
    [lambda: ZXGame(num_qubits, depth, max_episode_length=max_episode_length, pe_dim=pe_dim) for _ in range(num_envs)])
data, reward, done, stats = zx_env.reset()
while not done:
    data = pre_process(data, pe_dim)
    azx_dist_params, value = model(data.x, data.edge_index, data.edge_attr, data.node_type, data.b, data.pe)
    check_model_consistency(pyg.data.Batch.from_data_list([data]), azx_dist_params)
    azx_dist = AlphaZXDistribution(azx_dist_params)
    sampled_action = azx_dist.sample(1)
    sampled_rewrite_types = sampled_action[:, 0]
    sampled_nodes = sampled_action[:, 1]
    action = sampled_action.numpy()
    data, reward, done, stats = zx_env.step(action)
    print('action = ', action)
    print('reward = ', reward)
    print('done = ', done)
    print('stats = ', stats)
