from typing import Type

import networkx as nx
import torch
from torch_geometric.data import Data

from alphazx.diagram.action_decoder import compute_f_right_params
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.match import MatchNode, FRightZMatch, FLeftZMatch, FRightXMatch, FLeftXMatch, \
    BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch, YRightXMatch, YLeftXMatch
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import ZXMatchDiagram, to_zx_match_diagram, DataIndexToMatch
from alphazx.rewriting.utils import rewrite, FRightParameters


def node_index_to_match(node_index: int, match_diagram: ZXMatchDiagram) -> MatchNode:
    return list(match_diagram.nodes)[node_index]


def assert_correct_match_instance(expected_class: Type[MatchNode], match: MatchNode) -> None:
    if not isinstance(match, expected_class):
        raise ValueError(f'Expected {expected_class} but got {match}')


def tuple_to_match(zx_match_diagram: ZXMatchDiagram, data: Data, action: tuple, data_index: DataIndexToMatch) -> tuple[
    MatchNode, FRightParameters | None]:
    # In this function, the batch dimension of 'action' is always one.
    action_type = action[0] - 11
    match = data_index[action[1]]
    if action_type == FRightZMatch.index or action_type == FRightXMatch.index:
        if action_type == FRightZMatch:
            assert_correct_match_instance(FRightZMatch, match)
        elif action_type == FRightXMatch:
            assert_correct_match_instance(FRightXMatch, match)
        return match, compute_f_right_params(action, data, data_index, zx_match_diagram)
    elif action_type == FLeftZMatch.index:
        assert_correct_match_instance(FLeftZMatch, match)
        return match, None
    elif action_type == FLeftXMatch.index:
        assert_correct_match_instance(FLeftXMatch, match)
        return match, None
    elif action_type == BRightMatch.index:
        assert_correct_match_instance(BRightMatch, match)
        return match, None
    elif action_type == BLeftMatch.index:
        assert_correct_match_instance(BLeftMatch, match)
        return match, None
    elif action_type == YRightZMatch.index:
        assert_correct_match_instance(YRightZMatch, match)
        return match, None
    elif action_type == YLeftZMatch.index:
        assert_correct_match_instance(YLeftZMatch, match)
        return match, None
    elif action_type == YRightXMatch.index:
        assert_correct_match_instance(YRightXMatch, match)
        return match, None
    elif action_type == YLeftXMatch.index:
        assert_correct_match_instance(YLeftXMatch, match)
        return match, None
    else:
        raise ValueError(f'Unexpected action type {action_type}')


def num_non_clifford_gates(diagram: ZXDiagram) -> int:
    if diagram.number_of_nodes() == diagram.num_b_nodes():
        return 0
    else:
        return sum([1 if p % 0.5 != 0 else 0 for p in diagram.phases().values()])


def is_simplified(zx_diagram: ZXDiagram) -> bool:
    return num_non_clifford_gates(zx_diagram) == 0


class DiagramStats:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        self.num_nodes = zx_match_diagram.zx_diagram.number_of_nodes()
        self.num_b_nodes = zx_match_diagram.zx_diagram.num_b_nodes()
        self.num_z_nodes = zx_match_diagram.zx_diagram.num_z_nodes()
        self.num_x_nodes = zx_match_diagram.zx_diagram.num_x_nodes()
        self.num_non_clifford_gates = num_non_clifford_gates(zx_match_diagram.zx_diagram)
        self.num_edges = zx_match_diagram.zx_diagram.num_edges()
        self.num_b_left_matches = len(zx_match_diagram.bl_nodes)

    def to_dict(self) -> dict[str, int]:
        return {
            'num_nodes': self.num_nodes,
            'num_b_nodes': self.num_b_nodes,
            'num_z_nodes': self.num_z_nodes,
            'num_x_nodes': self.num_x_nodes,
            'num_non_clifford_gates': self.num_non_clifford_gates,
            'num_edges': self.num_edges,
            'num_b_left_matches': self.num_b_left_matches
        }


def calculate_reward(old_diagram_stats: DiagramStats, new_diagram_stats: DiagramStats) -> int:
    """
    -2 for every node
    -3 for every new edge
    -5 for every new non-Clifford gate.
    """
    return 3 * (old_diagram_stats.num_edges - new_diagram_stats.num_edges) + 5 * (
            old_diagram_stats.num_non_clifford_gates - new_diagram_stats.num_non_clifford_gates) + 2 * (
            old_diagram_stats.num_z_nodes - new_diagram_stats.num_z_nodes) + 2 * (
            old_diagram_stats.num_x_nodes - new_diagram_stats.num_x_nodes)


class EpisodeStats:
    def __init__(self):
        pass

    def reset(self):
        pass

    def to_dict(self) -> dict[str, int]:
        pass


class BestActionStats:
    def __init__(self):
        pass

    def reset(self):
        pass

    def to_dict(self) -> dict[str, int]:
        pass


def remove_isolated_nodes(zx_diagram: ZXDiagram) -> None:
    zx_diagram.remove_nodes_from(list(nx.isolates(zx_diagram)))


def remove_self_loop_edges(zx_diagram: ZXDiagram) -> None:
    zx_diagram.remove_edges_from(list(nx.selfloop_edges(zx_diagram, keys=True)))


def remove_isolated_components(zx_diagram: ZXDiagram) -> None:
    if zx_diagram.num_b_nodes() == 0 or zx_diagram.num_b_nodes() == 1:
        raise ValueError('Valid diagrams always have at least two boundary nodes')
    b_nodes = zx_diagram.b_nodes()
    for c in nx.connected_components(zx_diagram.copy()):
        if b_nodes.isdisjoint(c):
            zx_diagram.remove_nodes_from(c)


class ZXGame:
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 t_gates: bool = True,
                 step_penalty: int = 1,
                 max_episode_length: int = 100,
                 simplified_reward: int = 1000,
                 pe_dim: int = 40):

        self.num_qubits = num_qubits
        self.depth = depth
        self.t_gates = t_gates
        self.step_penalty = step_penalty
        self.max_episode_length = max_episode_length
        self.simplified_reward = simplified_reward
        self.pe_dim = pe_dim

        self.episode_length = 0
        self.done = False
        self.previous_reward = 0.
        self.episode_return = 0.

        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        remove_isolated_nodes(self.zx_diagram)
        remove_self_loop_edges(self.zx_diagram)
        remove_isolated_components(self.zx_diagram)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.diagram_stats = DiagramStats(self.zx_match_diagram)
        data, self.data_index = self.zx_match_diagram.to_pyg_data(True)
        # data = pre_process(data, self.pe_dim)
        data.b = torch.zeros_like(data.node_type)
        self.data = data

    def step(self, action: tuple) -> tuple[Data, int, bool, dict]:

        match, params = tuple_to_match(self.zx_match_diagram, self.data, action, self.data_index)
        rewrite(self.zx_diagram, match, params)
        remove_isolated_nodes(self.zx_diagram)
        remove_self_loop_edges(self.zx_diagram)
        remove_isolated_components(self.zx_diagram)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)

        self.episode_length += 1

        old_diagram_stats = self.diagram_stats
        self.diagram_stats = DiagramStats(self.zx_match_diagram)
        self.previous_reward = calculate_reward(old_diagram_stats, self.diagram_stats)

        done_reward = 0
        self.done = False
        if is_simplified(self.zx_diagram):
            self.done = True
            done_reward = self.simplified_reward
        elif self.episode_length == self.max_episode_length:
            self.done = True

        # self.previous_reward += done_reward if self.done else -self.step_penalty * self.episode_length
        self.episode_return += self.previous_reward

        data, self.data_index = self.zx_match_diagram.to_pyg_data(True)
        data.b = torch.zeros_like(data.node_type)
        self.data = data
        return self.data, self.previous_reward, self.done, self.diagram_stats.to_dict()

    def reset(self, start_state: ZXDiagram = None) -> tuple[Data, int, bool, dict]:
        self.episode_return = 0.
        self.previous_reward = 0.
        self.episode_length = 0
        self.zx_diagram = start_state.copy() if start_state is not None else clifford_zx_diagram(self.num_qubits,
                                                                                                 self.depth,
                                                                                                 self.t_gates)
        remove_isolated_nodes(self.zx_diagram)
        remove_self_loop_edges(self.zx_diagram)
        remove_isolated_components(self.zx_diagram)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.diagram_stats = DiagramStats(self.zx_match_diagram)

        done_reward = 0
        self.done = False
        if is_simplified(self.zx_diagram):
            self.done = True
            done_reward = self.simplified_reward
        elif self.episode_length == self.max_episode_length:
            self.done = True

        data, self.data_index = self.zx_match_diagram.to_pyg_data(True)
        data.b = torch.zeros_like(data.node_type)
        self.data = data
        return self.data, done_reward, self.done, self.diagram_stats.to_dict()
