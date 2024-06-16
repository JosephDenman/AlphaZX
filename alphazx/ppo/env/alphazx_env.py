import copy
from typing import Any, Optional, Tuple

import numpy as np
from alphazx.diagram import ZXDiagram
from alphazx.game.zx_game import ZXGame
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym import spaces


@ENV_REGISTRY.register('alphazx')
class AlphaZXEnv(BaseEnv):
    """
    Overview:
        An AlphaZX environment that inherits from the BaseEnv. This environment can be used for training and
        evaluating AI players for the game of Gomoku.

    .. note::
        For the latest macOS, you should set context='spawn' or 'forkserver' in ppo/envs/env_manager/subprocess_env_manager.py
        to be able to use subprocess env_manager.
    """

    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_id="alphazx",
        # (int) The maximum number of qubits in the initial graph.
        max_num_qubits=50,
        # (int) The maximum gate depth of the initial graph.
        max_circuit_depth=50,
        # (bool) Whether to include T-gates in the initial graph.
        t_gates=True,
        # (int) The maximum number of new edges generated in a fission action.
        max_num_new_edges=10,
        # (int) The number of phase buckets used to discretize the phase space.
        num_phase_buckets=10,
        # (float) The reward for completely simplifying the graph.
        done_reward=1.,
        # (float) The penalty for each step taken.
        step_penalty=-1.,
        # (int) The maximum number of steps before the game is considered complete.
        max_num_steps=100,
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (float) The scale of the render screen.
        screen_scaling=9,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=False,
        # (bool) Whether to scale the observation.
        scale=True,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0,
        # (bool) Whether to use the MCTS ctree in AlphaZX. If True, then the AlphaZero MCTS ctree will be used.
        alphazx_mcts_ctree=False,
    )

    def __init__(self, cfg: dict = None):
        self._cfg = cfg
        self.max_num_qubits = cfg.max_num_qubits
        self.max_circuit_depth = cfg.max_circuit_depth
        self.t_gates = cfg.t_gates
        self.max_num_new_edges = cfg.max_num_new_edges
        self.num_phase_buckets = cfg.num_phase_buckets
        self.done_reward = cfg.done_reward
        self.step_penalty = cfg.step_penalty
        self.max_num_steps = cfg.max_num_steps
        self._seed = None
        self._dynamic_seed = None
        self._observation_space = spaces.Graph(spaces.Discrete(11 * 16), spaces.Discrete(2))
        # TODO: FIX THE ACTION SPACE
        self._action_space = spaces.Tuple([spaces.Discrete(11), spaces.Discrete(self.num_phase_buckets),
                                           spaces.Discrete(self.max_num_new_edges)])
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.zx_game = ZXGame(self.max_num_qubits, self.max_circuit_depth, self.t_gates, False, self.step_penalty,
                              self.max_num_steps,
                              self.done_reward)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def current_state(self):
        return self.zx_game.data, self.zx_game.data

    def reset(self, start_state: ZXDiagram = None) -> Any:
        # if start_state is not None:
        #     print(f'Resetting to diagram {start_state.id}')
        # else:
        #     print(f'Resetting with fresh diagram')
        return self.zx_game.reset(start_state)[0]

    def close(self) -> None:
        pass

    def step(self, action: tuple) -> BaseEnvTimestep:
        diagram_id = action[-1]
        action = action[:-1]
        assert self.zx_game.zx_diagram.id == diagram_id, f'Expected action diagram id {diagram_id} to equal actual diagram id {self.zx_game.zx_diagram.id}'
        observation, reward, done = self.zx_game.step(action)
        return BaseEnvTimestep(observation, reward, done,
                               {'eval_episode_return': self.zx_game.episode_return} if done else {})

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        print('self = ', self)
        print('seed = ', seed)
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        pass

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def get_done_reward(self) -> Tuple[bool, Optional[float]]:
        return self.zx_game.done, self.done_reward if self.zx_game.done else None
