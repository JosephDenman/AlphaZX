from typing import Type

import networkx as nx
from torch import Tensor
from torch_geometric.data import Data

from diagram.feature_conversions import cat_phase_to_float, cat_new_edges_to_int, bernoulli_transfer_edges_to_tuple
from diagram.match import Match, FRightZMatch, FLeftZMatch, FRightXMatch, FLeftXMatch, \
    BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch, YRightXMatch, YLeftXMatch
from diagram.pyzx_graph_generator import clifford_zx_diagram
from diagram.zx_diagram import ZXDiagram
from diagram.zx_match_diagram import ZXMatchDiagram, to_zx_match_diagram
from rewriting.util import rewrite, FRightParameters


def node_index_to_match(node_index: int, match_diagram: ZXMatchDiagram) -> Match:
    return list(match_diagram.nodes)[node_index]


def assert_correct_match_instance(expected_class: Type[Match], match: Match) -> None:
    if not isinstance(match, expected_class):
        raise ValueError(f'Expected {expected_class} but got {type(match)}')


def tensor_to_match(zx_match_diagram: ZXMatchDiagram, action: Tensor) -> tuple[Match, FRightParameters | None]:
    action_type = action[0]
    node = action[1]
    match = node_index_to_match(node, zx_match_diagram)
    if action_type == FRightZMatch.index:
        assert_correct_match_instance(FRightZMatch, match)
        phase = cat_phase_to_float(action[2], zx_match_diagram.phase_denominator)
        new_edges = cat_new_edges_to_int(action[3])
        transfer_edges = bernoulli_transfer_edges_to_tuple(action[4:])
        # TODO: BIG - How to convert the transfer_edges to the correct representation?
        return match, FRightParameters(phase, new_edges, transfer_edges)
    elif action_type == FLeftZMatch.index:
        assert_correct_match_instance(FLeftZMatch, match)
        raise Exception('Not implemented')
    elif action_type == FRightXMatch.index:
        assert_correct_match_instance(FRightXMatch, match)
        raise Exception('Not implemented')
    elif action_type == FLeftXMatch.index:
        assert_correct_match_instance(FLeftXMatch, match)
        raise Exception('Not implemented')
    elif action_type == BRightMatch.index:
        assert_correct_match_instance(BRightMatch, match)
        raise Exception('Not implemented')
    elif action_type == BLeftMatch.index:
        assert_correct_match_instance(BLeftMatch, match)
        raise Exception('Not implemented')
    elif action_type == YRightZMatch.index:
        assert_correct_match_instance(YRightZMatch, match)
        raise Exception('Not implemented')
    elif action_type == YLeftZMatch.index:
        assert_correct_match_instance(YLeftZMatch, match)
        raise Exception('Not implemented')
    elif action_type == YRightXMatch.index:
        assert_correct_match_instance(YRightXMatch, match)
        raise Exception('Not implemented')
    elif action_type == YLeftXMatch.index:
        assert_correct_match_instance(YLeftXMatch, match)
        raise Exception('Not implemented')
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


def is_simplified(diagram: ZXDiagram) -> bool:
    # TODO - Maybe not this simple...
    return diagram.number_of_nodes() == diagram.num_b_nodes()


class ZXGame:
    def __init__(self, num_qubits: int, depth: int, t_gates: bool = True, one_hot_types: bool = False,
                 step_penalty: int = 1, simplified_reward: int = 1):
        self.zx_diagram: ZXDiagram | None = None
        self.zx_match_diagram: ZXMatchDiagram | None = None
        self.previous_value: int | None = None
        self.num_qubits = num_qubits
        self.depth = depth
        self.t_gates = t_gates
        self.one_hot_types = one_hot_types
        self.simplified_reward = simplified_reward
        self.step_penalty = step_penalty

    def _remove_isolated_nodes(self) -> None:
        self.zx_diagram.remove_nodes_from(list(nx.isolates(self.zx_diagram)))

    def _remove_self_loop_edges(self) -> None:
        self.zx_diagram.remove_edges_from(list(nx.selfloop_edges(self.zx_diagram, keys=True)))

    def _remove_isolated_components(self) -> None:
        # TODO: Any isolated connected components (subgraphs representing scalars) should be removed, since scalars can
        #       be recovered for any scalar free diagram.
        pass

    def step(self, action: Tensor) -> tuple[Data, int, bool]:
        match, params = tensor_to_match(self.zx_match_diagram, action)
        rewrite(self.zx_diagram, match, params)
        current_value = diagram_value(self.zx_diagram)
        self._remove_isolated_nodes()
        self._remove_self_loop_edges()
        done = is_simplified(self.zx_diagram)
        reward = self.previous_value - current_value + (self.simplified_reward if done else -self.step_penalty)
        self.previous_value = current_value
        self.zx_match_diagram = ZXMatchDiagram(self.zx_diagram, self.one_hot_types)
        return self.zx_diagram.to_pyg_data(self.one_hot_types), reward, done

    def reset(self) -> Data:
        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram, self.one_hot_types)
        self.previous_value = diagram_value(self.zx_diagram)
        return self.zx_match_diagram.to_pyg_data()
