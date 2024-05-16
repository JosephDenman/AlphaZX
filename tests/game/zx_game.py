import torch

from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution
from alphazx.game.zx_game import ZXGame
from tests.models.policy_network import policy_network

num_qubits = 100
depth = 100

while True:
    model = policy_network(num_qubits, depth)
    zx_game = ZXGame(num_qubits, depth)
    data, reward, done = zx_game.reset()
    while not done:
        params = model(data)
        azx_dist = AlphaZXDistribution(params)
        action = tuple(azx_dist.sample(1)[0][0].tolist())
        print('action = ', action)
        step_result = zx_game.step(action)
        data = step_result['observation']
        reward = step_result['reward']
        print('reward = ', reward)
        done = step_result['done']
        print('done = ', done)
        print('episode_return = ', zx_game.episode_return)
        print('num_nodes = ', data.num_nodes)

