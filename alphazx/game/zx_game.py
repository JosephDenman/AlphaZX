from typing import Type

import networkx as nx
import torch
from torch_geometric.data import Data

from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.action_decoder import compute_new_phase, compute_num_new_edges, \
    compute_transfer_edges, compute_f_right_params
from alphazx.diagram.match import Match, FRightZMatch, FLeftZMatch, FRightXMatch, FLeftXMatch, \
    BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch, YRightXMatch, YLeftXMatch
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import ZXMatchDiagram, to_zx_match_diagram, DataIndexToMatch
from alphazx.models.pre_process import with_embeddable_feats, with_laplacian_pe, pre_process
from alphazx.rewriting.utils import rewrite, FRightParameters


def node_index_to_match(node_index: int, match_diagram: ZXMatchDiagram) -> Match:
    return list(match_diagram.nodes)[node_index]


def assert_correct_match_instance(expected_class: Type[Match], match: Match) -> None:
    if not isinstance(match, expected_class):
        raise ValueError(f'Expected {expected_class} but got {match}')


def tuple_to_match(zx_match_diagram: ZXMatchDiagram, data: Data, action: tuple, data_index: DataIndexToMatch) -> tuple[
    Match, FRightParameters | None]:
    # In this function, the batch dimension of 'action' is always one.
    action_type = action[0]
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


def diagram_value(diagram: ZXDiagram) -> int:
    """
    TODO - Maybe not this simple...
    -1 for every node
    -1 for every edge
    -1 for every non-Clifford gate.
    """
    return - diagram.number_of_nodes() - sum(
        [1 if p % 0.5 != 0 else 0 for p in diagram.phases().values()]) - len(diagram.edges())


def is_simplified(zx_diagram: ZXDiagram) -> bool:
    num_non_zero_phases = 0
    for n, phase in zx_diagram.phases().items():
        if phase != 0.:
            num_non_zero_phases += 1
    return num_non_zero_phases == 0


class ZXGame:
    def __init__(self, num_qubits: int, depth: int, t_gates: bool = True, one_hot_types: bool = False,
                 step_penalty: int = 1, simplified_reward: int = 1):
        self.episode_return = 0.
        self.previous_reward = 0.
        self.done = False
        self.num_qubits = num_qubits
        self.depth = depth
        self.t_gates = t_gates
        self.one_hot_types = one_hot_types
        self.simplified_reward = simplified_reward
        self.step_penalty = step_penalty
        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        self.num_nodes = self.zx_diagram.number_of_nodes()
        self.__remove_isolated_nodes()
        self.__remove_self_loop_edges()
        self.__remove_isolated_components()
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.previous_value = diagram_value(self.zx_diagram)
        data, self.data_index = self.zx_match_diagram.to_pyg_data(True, False)
        data = pre_process(data)
        data.batch = torch.zeros_like(data.node_type)
        self.data = data

    def __remove_isolated_nodes(self) -> None:
        self.zx_diagram.remove_nodes_from(list(nx.isolates(self.zx_diagram)))

    def __remove_self_loop_edges(self) -> None:
        self.zx_diagram.remove_edges_from(list(nx.selfloop_edges(self.zx_diagram, keys=True)))

    def __remove_isolated_components(self) -> None:
        if self.zx_diagram.num_b_nodes() == 0 or self.zx_diagram.num_b_nodes() == 1:
            raise ValueError('Valid diagrams always have at least two boundary nodes')
        b_nodes = self.zx_diagram.b_nodes()
        for c in nx.connected_components(self.zx_diagram.copy()):
            if b_nodes.isdisjoint(c):
                self.zx_diagram.remove_nodes_from(c)

    def step(self, action: tuple):
        match, params = tuple_to_match(self.zx_match_diagram, self.data, action, self.data_index)
        rewrite(self.zx_diagram, match, params)
        self.__remove_isolated_nodes()
        self.__remove_self_loop_edges()
        self.__remove_isolated_components()
        self.num_nodes = self.zx_diagram.number_of_nodes()
        self.done = is_simplified(self.zx_diagram)
        current_value = diagram_value(self.zx_diagram)
        self.previous_reward = self.previous_value - current_value + (0 if self.done else -self.step_penalty)
        self.episode_return += self.previous_reward
        self.previous_value = current_value
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        data, self.data_index = self.zx_match_diagram.to_pyg_data(True, False)
        data = pre_process(data)
        data.batch = torch.zeros_like(data.node_type)
        self.data = data
        return {
            'observation': self.data,
            'reward': self.previous_reward,
            'done': self.done,
        }

    def reset(self) -> tuple[Data, int, bool]:
        self.episode_return = 0.
        self.previous_reward = 0.
        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        self.__remove_isolated_nodes()
        self.__remove_self_loop_edges()
        self.__remove_isolated_components()
        self.num_nodes = self.zx_diagram.number_of_nodes()
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.previous_value = diagram_value(self.zx_diagram)
        self.done = is_simplified(self.zx_diagram)
        data, self.data_index = self.zx_match_diagram.to_pyg_data(True, False)
        data = pre_process(data)
        data.batch = torch.zeros_like(data.node_type)
        self.data = data
        return self.data, 0, self.done
