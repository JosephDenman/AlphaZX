import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

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
value_hidden_channels = 64

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
                     value_hidden_channels)

# max_episode_length = 1000
# num_qubits = 10
# depth = 10
# zx_game = ZXGame(num_qubits, depth, max_episode_length=max_episode_length, pe_dim=pe_dim)
# data, reward, done, stats = zx_game.reset()
# while not done:
#     data = pre_process(data, pe_dim)
#     azx_dist_params, value = model(data.x, data.edge_index, data.edge_attr, data.node_type, data.b, data.pe)
#     azx_dist = AlphaZXDistribution(azx_dist_params)
#     action = tuple(azx_dist.sample(1).squeeze().tolist())
#     print('prob = ', torch.exp(azx_dist.log_prob(torch.tensor(action).unsqueeze(dim=0).unsqueeze(dim=0))))
#     data, reward, done, stats = zx_game.step(action)
#     print('action = ', action)
#     print('reward = ', reward)
#     print('done = ', done)
#     print('stats = ', stats)
