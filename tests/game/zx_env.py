import torch
import gymnasium as gym
import torch_geometric as pyg

from gymnasium.spaces import Discrete, Box, Graph

from alphazx.game import ZXGame


class AlphaZXEnv(gym.Env):
    def __init__(self,
                 qubits: int,
                 depth: int,
                 t_gates: bool = True,
                 one_hot_types: bool = False,
                 step_penalty: int = 1,
                 num_possible_new_edges: int = 11,
                 num_possible_phases: int = 16,
                 max_episode_length: int = 75,
                 pe_dim: int = 40,
                 device: torch.device = 'cpu'):
        self.action_space = Discrete(1)
        self.observation_space = Graph(node_space=Box(low=-1, high=1, shape=(3,)), edge_space=Discrete(3), seed=42)
        self.qubits = qubits
        self.depth = depth
        self.t_gates = t_gates
        self.one_hot_types = one_hot_types
        self.step_penalty = step_penalty
        self.num_possible_new_edges = num_possible_new_edges
        self.num_possible_phases = num_possible_phases
        self.device = device
        self.zx_game = ZXGame(qubits, depth, t_gates, one_hot_types, step_penalty, max_episode_length, pe_dim)

    def step(self, action: tuple) -> tuple[pyg.data.Data, int, bool, bool, dict]:
        obs, reward, done, info = self.zx_game.step(action[0])
        return obs, reward, done, False, info

    def reset(self, seed: int = None, options: dict = None) -> tuple[pyg.data.Data, dict]:
        obs, reward, done, info = self.zx_game.reset(None)
        return obs, info