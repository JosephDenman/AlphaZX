from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.distributions import AlphaZXDistribution
from alphazx.game import ZXGame
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


max_episode_length = 1000
num_qubits = 200
depth = 200
zx_game = ZXGame(num_qubits, depth, max_episode_length=max_episode_length, pe_dim=pe_dim)
data, reward, done, stats = zx_game.reset()
while not done:
    azx_dist_params, value = model(data)
    azx_dist = AlphaZXDistribution(azx_dist_params)
    action = tuple(azx_dist.sample(1).squeeze().tolist())
    data, reward, done, stats = zx_game.step(action)
    print('action = ', action)
    print('reward = ', reward)
    print('done = ', done)
    print('stats = ', stats)

