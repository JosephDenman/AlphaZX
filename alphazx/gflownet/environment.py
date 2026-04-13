"""
GFlowNet environment wrapping the ZX-calculus GameState.

Adapts AlphaZX's GameState for GFlowNet trajectory sampling.  The action
space is a hybrid:

- Non-F-Right (types 2-9): flat (action_type, node) pairs, fully enumerable
  from the ZXMatchDiagram.
- F-Right (types 0-1): autoregressive decomposition (action_type, node, phase,
  new_edge, *transfer_edges) — parameter space too large to enumerate.

The environment delegates all ZX-calculus logic to GameState and the rewriting
infrastructure in alphazx.shared / alphazx.rewriting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from alphazx.diagram.diagram_generators import (
    clifford_zx_diagram,
    clifford_zx_diagram_with_pyzx,
    cliffordT_zx_diagram,
    cliffordT_zx_diagram_with_pyzx,
    cnot_had_phase_zx_diagram,
    cnot_had_phase_zx_diagram_with_pyzx,
)
from alphazx.game.zx_game import num_non_clifford_gates
from alphazx.shared.game_state import GameState
from alphazx.shared.config import CircuitConfig
from alphazx.shared.constants import ACTION_TYPE_NAMES, N_COMPONENTS_BY_ACTION_TYPE

logger = logging.getLogger(__name__)

# Boundary match nodes have index 0; action types start at index 1.
# action_type i in [0..9] corresponds to match_type_index = i + 1.
_BOUNDARY_INDEX = 0


@dataclass
class Transition:
    """A single (state, action, next_state, reward, done) transition."""
    state: GameState
    action: tuple
    next_state: GameState
    reward: float
    done: bool


@dataclass
class Trajectory:
    """A complete trajectory from initial state to terminal state."""
    transitions: list[Transition] = field(default_factory=list)
    initial_t_gates: int = 0
    final_t_gates: int = 0

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def t_gate_reduction(self) -> int:
        return self.initial_t_gates - self.final_t_gates

    @property
    def states(self) -> list[GameState]:
        """All states including the terminal one."""
        if not self.transitions:
            return []
        result = [t.state for t in self.transitions]
        result.append(self.transitions[-1].next_state)
        return result

    @property
    def actions(self) -> list[tuple]:
        return [t.action for t in self.transitions]

    def __len__(self) -> int:
        return len(self.transitions)


class ZXGFlowNetEnv:
    """GFlowNet environment for ZX-calculus simplification.

    Provides:
    - generate_state(): create a random initial GameState
    - enumerate_actions(): list all valid actions at a state
    - step(): apply an action, returning (next_state, reward, done)
    - terminal_reward(): compute R(x) for a completed trajectory
    """

    def __init__(self, config: CircuitConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Circuit generation
    # ------------------------------------------------------------------

    def generate_state(self) -> GameState:
        """Generate a random initial GameState from the circuit config.

        Retries up to max_circuit_retries times to find a circuit with at
        least min_initial_t_gates T-gates.
        """
        cfg = self.config
        diagram = None
        for _attempt in range(cfg.max_circuit_retries):
            if cfg.circuit_type == 'cnot_had_phase':
                diagram = cnot_had_phase_zx_diagram(
                    cfg.num_qubits, cfg.depth, cfg.p_had, cfg.p_t,
                )
            elif cfg.circuit_type == 'clifford':
                diagram = clifford_zx_diagram(
                    cfg.num_qubits, cfg.depth, t_gates=True,
                )
            elif cfg.circuit_type == 'cliffordT':
                diagram = cliffordT_zx_diagram(
                    cfg.num_qubits, cfg.depth,
                    p_t=cfg.p_t, p_s=cfg.p_s, p_hsh=cfg.p_hsh,
                )
            else:
                raise ValueError(f"Unknown circuit_type: {cfg.circuit_type}")

            t_count = num_non_clifford_gates(diagram)
            if t_count >= cfg.min_initial_t_gates:
                break

        assert diagram is not None
        return GameState.from_diagram(diagram)

    def generate_state_with_pyzx(self) -> tuple[GameState, object]:
        """Generate a random initial GameState AND the original PyZX graph.

        The PyZX graph is the circuit graph *before* any simplification,
        allowing an apples-to-apples comparison with ``pyzx.full_reduce()``.

        Returns:
            (game_state, pyzx_graph) where *pyzx_graph* is a ``pyzx.Graph``.
        """
        cfg = self.config
        diagram, pyzx_graph = None, None
        for _attempt in range(cfg.max_circuit_retries):
            if cfg.circuit_type == 'cnot_had_phase':
                diagram, pyzx_graph = cnot_had_phase_zx_diagram_with_pyzx(
                    cfg.num_qubits, cfg.depth, cfg.p_had, cfg.p_t,
                )
            elif cfg.circuit_type == 'clifford':
                diagram, pyzx_graph = clifford_zx_diagram_with_pyzx(
                    cfg.num_qubits, cfg.depth, t_gates=True,
                )
            elif cfg.circuit_type == 'cliffordT':
                diagram, pyzx_graph = cliffordT_zx_diagram_with_pyzx(
                    cfg.num_qubits, cfg.depth,
                    p_t=cfg.p_t, p_s=cfg.p_s, p_hsh=cfg.p_hsh,
                )
            else:
                raise ValueError(f"Unknown circuit_type: {cfg.circuit_type}")

            t_count = num_non_clifford_gates(diagram)
            if t_count >= cfg.min_initial_t_gates:
                break

        assert diagram is not None
        return GameState.from_diagram(diagram), pyzx_graph

    # ------------------------------------------------------------------
    # Action enumeration
    # ------------------------------------------------------------------

    def enumerate_flat_actions(self, state: GameState) -> list[tuple]:
        """Enumerate all non-F-Right actions (types 2-9) at the current state.

        Each action is a tuple:
            (0, action_type, node_index, 0, 0)

        where graph_id is always 0 (single graph), action_type is in [2..9],
        and node_index is the PyG index of the match node.  Phase, new_edge,
        and transfer_edge fields are 0 (ignored for non-F-Right).

        Returns an empty list if no non-F-Right actions are available.
        """
        data, data_index = state.ensure_data()
        actions = []
        num_nodes = data.x.shape[0]
        for i in range(num_nodes):
            match = data_index[i]
            match_type_idx = match.index  # 0=boundary, 1=FRZ, 2=FRX, ... 11+=super
            if match_type_idx <= 2 or match_type_idx > 10:
                # Skip boundary (0), F-Right (1=FRZ, 2=FRX), and super nodes (11+)
                continue
            action_type = match_type_idx - 1  # match index → action type
            actions.append((0, action_type, i, 0, 0))
        return actions

    def enumerate_f_right_nodes(self, state: GameState) -> list[tuple[int, int]]:
        """Enumerate all F-Right (type, node_index) pairs.

        Returns list of (action_type, node_index) where action_type is 0 or 1.
        The phase/edge/transfer parameters must be sampled from the policy.
        """
        data, data_index = state.ensure_data()
        pairs = []
        num_nodes = data.x.shape[0]
        for i in range(num_nodes):
            match = data_index[i]
            if match.index == 1:  # FRightZMatch
                pairs.append((0, i))
            elif match.index == 2:  # FRightXMatch
                pairs.append((1, i))
        return pairs

    def has_actions(self, state: GameState) -> bool:
        """Check if ANY action (flat or F-Right) is available."""
        return state.has_legal_actions()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, state: GameState, action: tuple,
    ) -> tuple[GameState, float, bool]:
        """Apply an action, returning (next_state, reward, done).

        Clones the state so the original is preserved.
        """
        next_state = state.clone()
        reward, done = next_state.apply_action(action)

        # Check early termination: too many T-gates added.
        if not done and self.config.max_t_gate_increase > 0:
            initial_t = state.num_non_clifford
            current_t = next_state.num_non_clifford
            if current_t - initial_t > self.config.max_t_gate_increase:
                done = True

        return next_state, reward, done

    def is_terminal(self, state: GameState) -> bool:
        """Check if a state is terminal (no T-gates or no legal actions)."""
        return state.is_terminal() or not state.has_legal_actions()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def terminal_reward(
        self,
        initial_t_gates: int,
        final_t_gates: int,
        reward_exponent: float = 4.0,
        min_reward: float = 1e-6,
    ) -> float:
        """Compute terminal reward R(x) for a completed trajectory.

        R(x) = max(min_reward, (T_reduced / T_initial) ^ reward_exponent)

        The min_reward floor is applied AFTER exponentiation to prevent
        astronomically small rewards (which produce huge log(R) values
        that destabilise the TB loss).

        Higher reward_exponent concentrates sampling on the best
        trajectories.  The exponent effectively controls the "temperature"
        of the GFlowNet's distribution over trajectories.
        """
        if initial_t_gates == 0:
            return min_reward
        reduction_ratio = max(0.0, (initial_t_gates - final_t_gates) / initial_t_gates)
        return max(min_reward, reduction_ratio ** reward_exponent)
