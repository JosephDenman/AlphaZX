"""
Self-play data generation for AlphaZero-style training.

This replaces PPO's rollout collection. Instead of running the policy greedily
and computing advantages, we play full episodes where each move is selected by MCTS.

SelfPlayWorker: plays a single episode, producing a list of training examples.
SelfPlayManager: orchestrates multiple workers, feeding examples into the replay buffer.

The value target z is filled in retroactively after the episode ends. For ZX-calculus
simplification, z is the total T-gate reduction achieved during the episode, normalized
to a reasonable range.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from collections import Counter

from alphazx.diagram.diagram_generators import clifford_zx_diagram, cnot_had_phase_zx_diagram
from alphazx.game.zx_game import num_non_clifford_gates as num_non_clifford_gates_diagram
from alphazx.game.zx_game import num_non_clifford_gates
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.game_state import GameState
from alphazx.mcts.search import MCTS
from alphazx.mcts.replay_buffer import TrainingExample, ReplayBuffer
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Human-readable names for action type indices (action_type field in the action tuple).
# action_type 0 → FRightZ (index 1), action_type 1 → FRightX (index 2), etc.
ACTION_TYPE_NAMES = {
    0: "f-right-z",
    1: "f-right-x",
    2: "f-left-z",
    3: "f-left-x",
    4: "b-right",
    5: "b-left",
    6: "y-right-z",
    7: "y-left-z",
    8: "y-right-x",
    9: "y-left-x",
}


@dataclass
class EpisodeResult:
    """Summary statistics for a completed self-play episode."""
    num_steps: int
    total_reward: float
    initial_t_gates: int
    final_t_gates: int
    t_gates_reduced: int
    simplified: bool
    wall_time: float  # seconds
    examples: list[TrainingExample]


class SelfPlayWorker:
    """Plays a single self-play episode using MCTS and produces training examples.

    Each step:
    1. Run MCTS from the current state → get visit-count policy π.
    2. Record (preprocessed_state_data, π, None) — value target filled in later.
    3. Sample an action from π (with temperature), apply it, continue.
    4. When the episode ends, compute outcome z and fill it into all examples.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MCTSConfig,
        device: torch.device = torch.device('cpu'),
    ):
        self.mcts = MCTS(model, config)
        self.config = config
        self.device = device
        self._game_counter = 0

    def play_episode(
        self,
        start_diagram=None,
    ) -> EpisodeResult:
        """Play a complete self-play episode.

        :param start_diagram: Optional ZXDiagram to start from. If None, generates
                              a random Clifford+T circuit using config parameters.
        :return: EpisodeResult with training examples and episode statistics.
        """
        start_time = time.time()
        self._game_counter += 1

        # Create initial state
        if start_diagram is not None:
            state = GameState.from_diagram(start_diagram.copy())
        else:
            diagram = self._generate_circuit()
            state = GameState.from_diagram(diagram)

        initial_t_gates = state.num_non_clifford

        # Collect examples: (state_data, mcts_policy, step_reward)
        # Value targets are filled in retroactively after the episode ends,
        # using discounted cumulative future reward from each step.
        pending_examples: list[TrainingExample] = []
        step_rewards: list[float] = []
        total_reward = 0.0
        num_steps = 0
        max_increase = self.config.max_t_gate_increase

        early_terminated_degenerate = False

        # Per-step timing accumulators for profiling
        _t_mcts = 0.0
        _t_preprocess = 0.0
        _t_apply = 0.0

        # Track action types used during this episode
        _action_type_counts = Counter()

        while num_steps < self.config.max_episode_length:
            if state.is_terminal() or not state.has_legal_actions():
                break

            # Early termination: stop if T-gates have increased too far
            # beyond the starting count.  This prevents the untrained agent
            # from spending 100 steps making the diagram progressively worse.
            # Use >= so that we stop AS SOON as the threshold is reached,
            # not one action later.
            if max_increase > 0:
                current_t = state.num_non_clifford
                if current_t >= initial_t_gates + max_increase:
                    logger.debug(
                        f"Early termination (pre-action): T-gates {current_t} >= "
                        f"initial {initial_t_gates} + {max_increase}"
                    )
                    early_terminated_degenerate = True
                    break

            # Run MCTS from current state
            # Use exploration temperature for training data generation
            _t0 = time.time()
            action, policy, _ = self.mcts.select_action(state, self.device)
            _t_mcts += time.time() - _t0

            if not policy:
                # No legal actions found by MCTS (edge case)
                break

            # Preprocess and snapshot the state BEFORE applying the action.
            # This is the state the model should learn to evaluate.
            _t0 = time.time()
            state_data = self._preprocess_state(state)
            _t_preprocess += time.time() - _t0

            # Record the training example (value target TBD)
            example = TrainingExample(
                state_data=state_data,
                mcts_policy=policy,
                value_target=None,
                game_id=self._game_counter,
            )
            pending_examples.append(example)

            # Apply the selected action to advance the game
            # We mutate state in place since we already captured the snapshot
            _t0 = time.time()
            try:
                reward, done = state.apply_action(action)
            except (ValueError, KeyError, IndexError, AssertionError) as e:
                logger.warning(f"Action application failed at step {num_steps}: {e}")
                step_rewards.append(0.0)
                _t_apply += time.time() - _t0
                break
            _t_apply += time.time() - _t0

            step_rewards.append(reward)
            total_reward += reward
            num_steps += 1

            # Log per-step action detail
            action_type_idx = action[1] if len(action) > 1 else -1
            action_name = ACTION_TYPE_NAMES.get(action_type_idx, f"unknown({action_type_idx})")
            _action_type_counts[action_name] += 1
            new_t = state.num_non_clifford
            logger.debug(
                f"  step {num_steps}: {action_name}, "
                f"reward={reward:+.2f}, "
                f"t_gates={new_t} ({new_t - initial_t_gates:+d})"
            )

            if done:
                break

            # Post-action check: also stop if THIS action pushed us over.
            # This catches single actions that jump past the threshold
            # (e.g., f-right rewrites that add multiple T-gates at once).
            if max_increase > 0:
                current_t = state.num_non_clifford
                if current_t >= initial_t_gates + max_increase:
                    logger.debug(
                        f"Early termination (post-action): T-gates {current_t} >= "
                        f"initial {initial_t_gates} + {max_increase}"
                    )
                    early_terminated_degenerate = True
                    break

        # Compute the episode outcome and fill value targets into all examples.
        final_t_gates = state.num_non_clifford
        t_gates_reduced = initial_t_gates - final_t_gates
        simplified = state.is_terminal()

        # --- Value targets: discounted cumulative future reward ---
        # Instead of a uniform episode outcome for every step, we use the
        # discounted sum of future step rewards from each position:
        #   v_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
        #
        # This gives more informative targets than uniform assignment:
        # - Steps that immediately precede simplification get high values
        # - Steps early in a degenerate episode get less negative signal
        # - The per-step shaped reward (from calculate_reward) provides
        #   gradient even when the episode outcome is neutral
        #
        # We normalize by max(initial_t_gates, 1) * 10 to keep values in
        # a reasonable range (calculate_reward uses 10x multiplier for T-gates).
        gamma = self.config.gamma
        # The primary reward multiplier is 10x per T-gate (from calculate_reward).
        # Add ~20% headroom for secondary rewards (node/edge/match reductions).
        normalizer = max(initial_t_gates, 1) * 12.0

        if pending_examples and step_rewards:
            # If the episode was cut short because T-gates increased beyond
            # the allowed threshold, add a terminal penalty.  This gives the
            # value network a clear negative signal for states that lead to
            # degenerate behavior, bootstrapping the concept of "this path is
            # bad" into the value estimates much faster than relying solely on
            # the step-level shaped rewards.
            if early_terminated_degenerate:
                terminal_penalty = -max_increase * 10.0  # Same scale as T-gate reward
                step_rewards.append(terminal_penalty)

            # Compute discounted returns from the end backward
            T = len(step_rewards)
            discounted_returns = [0.0] * T
            running_return = 0.0
            for t in reversed(range(T)):
                running_return = step_rewards[t] + gamma * running_return
                discounted_returns[t] = running_return

            # Assign normalized value targets
            for i, example in enumerate(pending_examples):
                if i < T:
                    example.value_target = max(-1.0, min(1.0,
                        discounted_returns[i] / normalizer
                    ))
                else:
                    # Edge case: more examples than rewards (break after snapshot)
                    example.value_target = 0.0
        else:
            # No steps taken — trivial episode
            for example in pending_examples:
                example.value_target = 0.0

        wall_time = time.time() - start_time

        # Log per-episode timing breakdown for profiling
        if num_steps > 0:
            action_dist = ", ".join(
                f"{name}={count}" for name, count in _action_type_counts.most_common()
            )
            logger.debug(
                f"Episode timing: mcts={_t_mcts:.2f}s, "
                f"preprocess={_t_preprocess:.2f}s, "
                f"apply_action={_t_apply:.2f}s, "
                f"other={wall_time - _t_mcts - _t_preprocess - _t_apply:.2f}s "
                f"(total={wall_time:.2f}s, {num_steps} steps)"
            )
            logger.debug(f"Episode actions: {action_dist}")

        return EpisodeResult(
            num_steps=num_steps,
            total_reward=total_reward,
            initial_t_gates=initial_t_gates,
            final_t_gates=final_t_gates,
            t_gates_reduced=t_gates_reduced,
            simplified=simplified,
            wall_time=wall_time,
            examples=pending_examples,
        )

    def _generate_circuit(self):
        """Generate a fresh ZX diagram for a new self-play episode.

        Retries up to max_circuit_retries times to find a circuit with at
        least min_initial_t_gates T-gates. Circuits with 0-1 T-gates provide
        negligible learning signal and waste compute.

        Uses the configured circuit_type:
        - 'cnot_had_phase': Real quantum circuits (CNOT + Hadamard + T gates)
          converted to ZX diagrams. These have realistic structure with plenty
          of T-gates to optimize. Recommended for training.
        - 'clifford': Random ZX graphs from pyzx.generate.cliffords. These are
          graph-level diagrams not derived from circuits, and may have fewer
          T-gates or less realistic structure.
        """
        cfg = self.config
        min_t = cfg.min_initial_t_gates
        max_retries = cfg.max_circuit_retries

        for attempt in range(max_retries):
            diagram = self._generate_circuit_once()
            t_count = num_non_clifford_gates_diagram(diagram)
            if t_count >= min_t:
                return diagram
            if attempt < max_retries - 1:
                logger.debug(
                    f"Circuit has {t_count} T-gates (need >={min_t}), "
                    f"re-rolling (attempt {attempt + 1}/{max_retries})"
                )

        # Fallback: return whatever we got
        logger.debug(f"Could not find circuit with >={min_t} T-gates after {max_retries} attempts")
        return diagram

    def _generate_circuit_once(self):
        """Generate a single random circuit (no retry logic)."""
        cfg = self.config
        if cfg.circuit_type == 'cnot_had_phase':
            return cnot_had_phase_zx_diagram(
                cfg.num_qubits, cfg.depth, cfg.p_had, cfg.p_t,
            )
        elif cfg.circuit_type == 'clifford':
            return clifford_zx_diagram(
                cfg.num_qubits, cfg.depth, t_gates=True,
            )
        else:
            raise ValueError(
                f"Unknown circuit_type '{cfg.circuit_type}'. "
                f"Expected 'cnot_had_phase' or 'clifford'."
            )

    def _preprocess_state(self, state: GameState) -> Data:
        """Preprocess a GameState's PyG data for storage in the replay buffer.

        We apply the positional encoding here so that stored examples are
        ready for batched training without recomputation. The PE is computed
        fresh for each snapshot because the graph changes at every step.

        OPTIMIZATION: If evaluate_state() already preprocessed this state
        during the MCTS search (which it always does in the normal flow),
        we reuse the cached result instead of recomputing PE from scratch.
        This eliminates ~50% of per-step preprocessing cost.
        """
        # Check if MCTS already preprocessed this state (evaluate_state caches it)
        cached = getattr(state, '_cached_preprocessed_data', None)
        if cached is not None:
            # Consume the cache (each state snapshot should only be used once)
            state._cached_preprocessed_data = None
            return cached

        # Fallback: preprocess from scratch (e.g. if called outside MCTS flow)
        from alphazx.mcts.evaluate import _preprocess_data_for_model
        data = state.data.clone()
        data = _preprocess_data_for_model(data, self.config.pe_dim)
        return data


class SelfPlayManager:
    """Orchestrates self-play game generation and feeds examples into the replay buffer.

    Currently runs games sequentially. The main optimization opportunity is batching
    neural network evaluations across multiple concurrent MCTS searches, but that
    requires virtual loss support and async evaluation — defer to Phase 6.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MCTSConfig,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device('cpu'),
    ):
        self.worker = SelfPlayWorker(model, config, device)
        self.config = config
        self.replay_buffer = replay_buffer
        self.device = device

        # Lifetime statistics
        self.total_games: int = 0
        self.total_examples: int = 0
        self.total_t_gates_reduced: int = 0
        self.total_simplified: int = 0

    def generate_games(
        self,
        num_games: int,
        start_diagrams: Optional[list] = None,
    ) -> list[EpisodeResult]:
        """Generate a batch of self-play games and store examples in the replay buffer.

        :param num_games: Number of episodes to play.
        :param start_diagrams: Optional list of ZXDiagrams. If provided, each game
                               starts from the corresponding diagram. If shorter than
                               num_games, random circuits are used for the remainder.
        :return: List of EpisodeResult summaries.
        """
        results = []

        for i in range(num_games):
            start_diagram = None
            if start_diagrams is not None and i < len(start_diagrams):
                start_diagram = start_diagrams[i]

            result = self.worker.play_episode(start_diagram)
            results.append(result)

            # Add examples to replay buffer
            self.replay_buffer.add_game(result.examples)

            # Update statistics
            self.total_games += 1
            self.total_examples += len(result.examples)
            self.total_t_gates_reduced += result.t_gates_reduced
            if result.simplified:
                self.total_simplified += 1

            logger.info(
                f"Game {self.total_games}: "
                f"steps={result.num_steps}, "
                f"t_gates={result.initial_t_gates}→{result.final_t_gates} "
                f"(-{result.t_gates_reduced}), "
                f"simplified={result.simplified}, "
                f"time={result.wall_time:.1f}s, "
                f"examples={len(result.examples)}"
            )

        return results

    def stats_summary(self) -> dict:
        """Return a summary of self-play statistics."""
        return {
            'total_games': self.total_games,
            'total_examples': self.total_examples,
            'total_t_gates_reduced': self.total_t_gates_reduced,
            'total_simplified': self.total_simplified,
            'simplification_rate': (
                self.total_simplified / max(1, self.total_games)
            ),
            'avg_t_gates_reduced': (
                self.total_t_gates_reduced / max(1, self.total_games)
            ),
            'buffer_size': len(self.replay_buffer),
            'buffer_total_added': self.replay_buffer.total_added,
        }
