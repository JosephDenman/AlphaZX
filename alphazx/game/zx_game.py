from typing import Type

import networkx as nx
from torch_geometric.data import Data

from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.match import MatchNode, FRightZMatch, FLeftZMatch, FRightXMatch, FLeftXMatch, \
    BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch, YRightXMatch, YLeftXMatch, METADATA
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import ZXMatchDiagram, to_zx_match_diagram, DataIndexToMatch
from alphazx.diagram.zx_match_diagram import compute_f_right_params
from alphazx.rewriting.efficient_rewrite import (
    efficient_rewrite,
    get_matches_involving_nodes,
    remove_match_from_diagram,
)


def node_index_to_match(node_index: int, match_diagram: ZXMatchDiagram) -> MatchNode:
    return list(match_diagram.nodes)[node_index]


def assert_correct_match_instance(expected_class: Type[MatchNode], match: MatchNode) -> None:
    if not isinstance(match, expected_class):
        raise ValueError(f'Expected {expected_class} but got {match}')


def tuple_to_match(zx_match_diagram: ZXMatchDiagram, data: Data, action: tuple, data_index: DataIndexToMatch) -> tuple[
    MatchNode, tuple[float, int, set[int]] | None]:
    # In this function, the batch dimension of 'action' is always one.
    # Action type is now directly the index from the policy network (0-9)
    action_type = action[1]
    match = data_index[action[2]]
    # Map action type indices to match node indices
    # action_type 0 corresponds to FRightZMatch (index 1)
    # action_type 1 corresponds to FRightXMatch (index 2)
    # etc.
    match_type_index = action_type + 1
    if match_type_index == FRightZMatch.index or match_type_index == FRightXMatch.index:
        if match_type_index == FRightZMatch.index:
            assert_correct_match_instance(FRightZMatch, match)
        elif match_type_index == FRightXMatch.index:
            assert_correct_match_instance(FRightXMatch, match)
        return match, compute_f_right_params(action, data, data_index, zx_match_diagram)
    elif match_type_index == FLeftZMatch.index:
        assert_correct_match_instance(FLeftZMatch, match)
        return match, None
    elif match_type_index == FLeftXMatch.index:
        assert_correct_match_instance(FLeftXMatch, match)
        return match, None
    elif match_type_index == BRightMatch.index:
        assert_correct_match_instance(BRightMatch, match)
        return match, None
    elif match_type_index == BLeftMatch.index:
        assert_correct_match_instance(BLeftMatch, match)
        return match, None
    elif match_type_index == YRightZMatch.index:
        assert_correct_match_instance(YRightZMatch, match)
        return match, None
    elif match_type_index == YLeftZMatch.index:
        assert_correct_match_instance(YLeftZMatch, match)
        return match, None
    elif match_type_index == YRightXMatch.index:
        assert_correct_match_instance(YRightXMatch, match)
        return match, None
    elif match_type_index == YLeftXMatch.index:
        assert_correct_match_instance(YLeftXMatch, match)
        return match, None
    else:
        raise ValueError(f'Unexpected action type {action_type} (match_type_index {match_type_index})')


def num_non_clifford_gates(diagram: ZXDiagram) -> int:
    if diagram.number_of_nodes() == diagram.num_b_nodes():
        return 0
    else:
        return sum([1 if p % 0.5 != 0 else 0 for p in diagram.phases().values()])


def is_simplified(zx_diagram: ZXDiagram) -> bool:
    return num_non_clifford_gates(zx_diagram) == 0


class DiagramStats:
    def __init__(self, zx_match_diagram: ZXMatchDiagram):
        for match_node_type_abbrev in METADATA.match_node_type_abbrevs:
            attr_name = f'{match_node_type_abbrev}_nodes'
            setattr(self, attr_name, len(getattr(zx_match_diagram, attr_name)))
        self.num_nodes = zx_match_diagram.zx_diagram.number_of_nodes()
        self.num_edges = zx_match_diagram.zx_diagram.num_edges()
        self.num_non_clifford_gates = num_non_clifford_gates(zx_match_diagram.zx_diagram)


def rewrite_weight(diagram_stats_key: str) -> int:
    if diagram_stats_key == 'num_non_clifford_gates':
        return 20
    else:
        return 2


def calculate_reward_simple(old_diagram_stats: DiagramStats, new_diagram_stats: DiagramStats) -> int:
    old_diagram_stats_dict = vars(old_diagram_stats)
    new_diagram_stats_dict = vars(new_diagram_stats)
    return old_diagram_stats_dict['num_non_clifford_gates'] - new_diagram_stats_dict['num_non_clifford_gates']


class RewardBreakdown:
    """Detailed breakdown of reward components for logging."""
    def __init__(self):
        self.t_gate_reward = 0.0
        self.node_reward = 0.0
        self.edge_reward = 0.0
        self.match_reward = 0.0
        self.total = 0.0

    def to_dict(self) -> dict:
        return {
            't_gate_reward': self.t_gate_reward,
            'node_reward': self.node_reward,
            'edge_reward': self.edge_reward,
            'match_reward': self.match_reward,
            'total_reward': self.total
        }


def calculate_reward(old_diagram_stats: DiagramStats, new_diagram_stats: DiagramStats) -> tuple[float, RewardBreakdown]:
    """
    Shaped reward function that provides more frequent feedback.
    Returns both the total reward and a breakdown of components.
    """
    breakdown = RewardBreakdown()

    # Primary reward: Non-Clifford gate reduction (T-gates)
    t_gate_reduction = old_diagram_stats.num_non_clifford_gates - new_diagram_stats.num_non_clifford_gates
    breakdown.t_gate_reward = t_gate_reduction * 10.0

    # Secondary rewards for progress indicators
    node_reduction = old_diagram_stats.num_nodes - new_diagram_stats.num_nodes
    edge_reduction = old_diagram_stats.num_edges - new_diagram_stats.num_edges

    # Small positive rewards for simplification
    if node_reduction > 0:
        breakdown.node_reward = node_reduction * 0.1
    # Penalty for increasing complexity
    if node_reduction < 0:
        breakdown.node_reward = node_reduction * 0.2

    if edge_reduction > 0:
        breakdown.edge_reward = edge_reduction * 0.05
    if edge_reduction < 0:
        breakdown.edge_reward = edge_reduction * 0.1

    # Small rewards for reducing specific match types (indicates progress)
    old_stats_dict = vars(old_diagram_stats)
    new_stats_dict = vars(new_diagram_stats)

    # Reward reduction in complex matches (indicates simplification)
    for match_type in ['br_nodes', 'bl_nodes', 'yrz_nodes', 'ylz_nodes', 'yrx_nodes', 'ylx_nodes']:
        if match_type in old_stats_dict and match_type in new_stats_dict:
            reduction = old_stats_dict[match_type] - new_stats_dict[match_type]
            if reduction > 0:
                breakdown.match_reward += reduction * 0.2

    breakdown.total = breakdown.t_gate_reward + breakdown.node_reward + breakdown.edge_reward + breakdown.match_reward
    return breakdown.total, breakdown


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


def remove_isolated_nodes(zx_diagram: ZXDiagram) -> set[int]:
    """Remove isolated nodes and return the set of removed node IDs."""
    isolated = set(nx.isolates(zx_diagram))
    zx_diagram.remove_nodes_from(list(isolated))
    return isolated


def remove_self_loop_edges(zx_diagram: ZXDiagram) -> None:
    zx_diagram.remove_edges_from(list(nx.selfloop_edges(zx_diagram, keys=True)))


def remove_isolated_components(zx_diagram: ZXDiagram) -> set[int]:
    """Remove isolated components and return the set of removed node IDs."""
    if zx_diagram.num_b_nodes() == 0 or zx_diagram.num_b_nodes() == 1:
        raise ValueError('Valid diagrams always have at least two boundary nodes')
    b_nodes = zx_diagram.b_nodes()
    removed = set()
    # nx.connected_components is read-only; no need to copy the graph.
    # ZXDiagram extends nx.MultiGraph (undirected), so this works directly.
    for c in list(nx.connected_components(zx_diagram)):
        if b_nodes.isdisjoint(c):
            removed.update(c)
            zx_diagram.remove_nodes_from(c)
    return removed


def update_match_diagram_for_removed_nodes(
    zx_match_diagram: ZXMatchDiagram,
    removed_nodes: set[int]
) -> None:
    """Update match diagram after nodes have been removed from the ZX diagram."""
    if not removed_nodes:
        return
    # Find and remove all matches involving the removed nodes
    matches_to_remove = get_matches_involving_nodes(zx_match_diagram, removed_nodes)
    for match in matches_to_remove:
        remove_match_from_diagram(zx_match_diagram, match)


class ZXGame:
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 t_gates: bool = True,
                 step_penalty: int = 1,
                 max_episode_length: int = 100,
                 simplified_reward: int = 1000,
                 pe_dim: int = 40):
        """
        Initialize a ZXGame.

        :param num_qubits: Number of qubits in the circuit
        :param depth: Depth of the circuit
        :param t_gates: Whether to include T-gates
        :param step_penalty: Penalty per step
        :param max_episode_length: Maximum number of steps per episode
        :param simplified_reward: Reward for fully simplifying the diagram
        :param pe_dim: Dimension of positional encoding
        """
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

        # Initialize cumulative reward trackers
        self.cumulative_t_gate_reward = 0.0
        self.cumulative_node_reward = 0.0
        self.cumulative_edge_reward = 0.0
        self.cumulative_match_reward = 0.0

        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        remove_isolated_nodes(self.zx_diagram)
        remove_self_loop_edges(self.zx_diagram)
        remove_isolated_components(self.zx_diagram)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.diagram_stats = DiagramStats(self.zx_match_diagram)
        self.data, self.data_index = self.zx_match_diagram.to_pyg_data(True)

        # Capture initial stats for episode summary
        self.initial_t_gates = self.diagram_stats.num_non_clifford_gates
        self.initial_nodes = self.diagram_stats.num_nodes
        self.initial_edges = self.diagram_stats.num_edges

    def __calculate_done_reward(self) -> tuple[bool, int]:
        done_reward = 0
        done = False
        if is_simplified(self.zx_diagram):
            done = True
            done_reward = self.simplified_reward
        elif self.episode_length == self.max_episode_length:
            done = True
        return done, done_reward

    def step(self, action: tuple) -> tuple[Data, float, bool, dict]:

        match, params = tuple_to_match(self.zx_match_diagram, self.data, action, self.data_index)

        # Use incremental match diagram update
        efficient_rewrite(self.zx_diagram, self.zx_match_diagram, match, params)

        # Handle cleanup operations with match diagram updates
        removed_isolated = remove_isolated_nodes(self.zx_diagram)
        update_match_diagram_for_removed_nodes(self.zx_match_diagram, removed_isolated)

        remove_self_loop_edges(self.zx_diagram)

        removed_components = remove_isolated_components(self.zx_diagram)
        update_match_diagram_for_removed_nodes(self.zx_match_diagram, removed_components)

        self.episode_length += 1

        old_diagram_stats = self.diagram_stats
        self.diagram_stats = DiagramStats(self.zx_match_diagram)
        # Use the shaped reward function - now returns (reward, breakdown)
        self.previous_reward, reward_breakdown = calculate_reward(old_diagram_stats, self.diagram_stats)
        self.done, done_reward = self.__calculate_done_reward()
        # Step penalty is now applied as a fixed cost per step
        self.episode_return += self.previous_reward + done_reward - self.step_penalty

        # Track cumulative reward components
        self.cumulative_t_gate_reward += reward_breakdown.t_gate_reward
        self.cumulative_node_reward += reward_breakdown.node_reward
        self.cumulative_edge_reward += reward_breakdown.edge_reward
        self.cumulative_match_reward += reward_breakdown.match_reward

        self.data, self.data_index = self.zx_match_diagram.to_pyg_data(True)

        info = {
            'diagram_stats': vars(self.diagram_stats),
            'reward_breakdown': reward_breakdown.to_dict(),
        }

        # Add episode summary on termination
        # Note: Use 'episode_info' to avoid conflict with Gymnasium's RecordEpisodeStatistics wrapper
        if self.done:
            info['episode_info'] = {
                'length': self.episode_length,
                'return': self.episode_return,
                'initial_t_gates': self.initial_t_gates,
                'final_t_gates': self.diagram_stats.num_non_clifford_gates,
                't_gates_reduced': self.initial_t_gates - self.diagram_stats.num_non_clifford_gates,
                'initial_nodes': self.initial_nodes,
                'final_nodes': self.diagram_stats.num_nodes,
                'initial_edges': self.initial_edges,
                'final_edges': self.diagram_stats.num_edges,
                'cumulative_t_gate_reward': self.cumulative_t_gate_reward,
                'cumulative_node_reward': self.cumulative_node_reward,
                'cumulative_edge_reward': self.cumulative_edge_reward,
                'cumulative_match_reward': self.cumulative_match_reward,
                'done_reward': done_reward,
                'simplified': is_simplified(self.zx_diagram),
            }

        return self.data, self.previous_reward, self.done, info

    def reset(self, start_state: ZXDiagram = None) -> tuple[Data, int, bool, dict]:
        self.episode_return = 0.
        self.previous_reward = 0.
        self.episode_length = 0

        # Reset cumulative reward trackers
        self.cumulative_t_gate_reward = 0.0
        self.cumulative_node_reward = 0.0
        self.cumulative_edge_reward = 0.0
        self.cumulative_match_reward = 0.0

        self.zx_diagram = start_state.copy() if start_state is not None else clifford_zx_diagram(self.num_qubits,
                                                                                                 self.depth,
                                                                                                 self.t_gates)
        remove_isolated_nodes(self.zx_diagram)
        remove_self_loop_edges(self.zx_diagram)
        remove_isolated_components(self.zx_diagram)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.diagram_stats = DiagramStats(self.zx_match_diagram)

        # Capture initial stats for episode summary
        self.initial_t_gates = self.diagram_stats.num_non_clifford_gates
        self.initial_nodes = self.diagram_stats.num_nodes
        self.initial_edges = self.diagram_stats.num_edges

        self.done, done_reward = self.__calculate_done_reward()
        self.data, self.data_index = self.zx_match_diagram.to_pyg_data(True)
        return self.data, done_reward, self.done, {'diagram_stats': vars(self.diagram_stats)}
