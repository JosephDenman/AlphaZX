import torch
import unittest
from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.distributions import AlphaZXDistribution
from alphazx.game import ZXGame
from alphazx.models import pre_process
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

torch.set_default_dtype(torch.float64)

num_node_types = len(METADATA.node_type_abbrevs)
num_possible_phases = len(POSSIBLE_PHASES)
num_possible_new_edges = 5
pe_dim = 20
repr_gps_node_embedding_out_channels = 64
repr_gps_num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
repr_gps_edge_embedding_out_channels = 64
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


class UtilsTest(unittest.TestCase):
    @staticmethod
    def test_run_zx_game():
        max_episode_length = 100
        num_qubits = 10
        depth = 10
        zx_game = ZXGame(num_qubits, depth, max_episode_length=max_episode_length, pe_dim=pe_dim)
        data, reward, done, stats = zx_game.reset()
        while not done:
            data = pre_process(data, pe_dim)
            azx_dist_params, value = model(data.x, data.edge_index, data.edge_attr, data.node_type, torch.zeros_like(data.x, dtype=torch.int64), data.pe, data.id)
            print('value = ', value)
            azx_dist = AlphaZXDistribution(azx_dist_params)
            action = tuple(azx_dist.sample(1).squeeze().tolist())
            print('action = ', action)
            data, reward, done, stats = zx_game.step(action)
            print('data = ', data)
            print('reward = ', reward)
            print('done = ', done)
            print('stats = ', stats)
