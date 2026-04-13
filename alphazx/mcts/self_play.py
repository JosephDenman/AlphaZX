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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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


def _assign_value_targets(
    examples: list[TrainingExample],
    step_rewards: list[float],
    initial_t_gates: int,
    final_t_gates: int,
    config: MCTSConfig,
) -> None:
    """Assign value targets to training examples after an episode ends.

    Supports two modes (controlled by ``config.value_target_mode``):

    **discounted_return** (default, recommended):
        Compute per-step discounted returns from step rewards, then normalize
        by ``initial_t_gates`` and clamp to [-1, 1].

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        target_t = clamp(G_t / initial_t_gates, -1, 1)

        This gives the value head a per-position learning signal.  Early
        "sacrificial" moves (e.g. F-Right that increases T-gates) get
        near-zero or slightly negative targets when they enable later
        reductions, instead of the strongly negative uniform target they
        would receive under the outcome-based scheme.  This is critical
        for learning multi-step F-Right→F-Left strategies.

    **uniform_outcome** (original AlphaZero approach):
        All steps share the same target:
            z = clamp((initial_t - final_t) / initial_t, -1, 1)
    """
    if not examples:
        return

    if config.value_target_mode == 'discounted_return' and step_rewards:
        gamma = config.gamma
        T = len(step_rewards)
        returns = [0.0] * T

        # Backward pass to compute discounted returns
        G = 0.0
        for t in reversed(range(T)):
            G = step_rewards[t] + gamma * G
            returns[t] = G

        # Normalize by initial T-gates (same scale as uniform outcome)
        # and clamp to [-1, 1] for tanh value head.
        for t, example in enumerate(examples):
            if initial_t_gates > 0:
                example.value_target = max(-1.0, min(1.0, returns[t] / initial_t_gates))
            else:
                example.value_target = 0.0
    else:
        # Uniform outcome (standard AlphaZero)
        if initial_t_gates > 0:
            outcome = (initial_t_gates - final_t_gates) / initial_t_gates
            outcome = max(-1.0, min(1.0, outcome))
        else:
            outcome = -1.0 if final_t_gates > 0 else 0.0

        for example in examples:
            example.value_target = outcome


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

        # Collect examples: (state_data, mcts_policy, value_target=TBD)
        # Value targets are filled in retroactively after the episode ends.
        pending_examples: list[TrainingExample] = []
        step_rewards: list[float] = []  # per-step rewards for discounted returns
        total_reward = 0.0
        num_steps = 0
        max_increase = self.config.max_t_gate_increase
        effective_max_length = self.config.effective_max_episode_length

        # Per-step timing accumulators for profiling
        _t_mcts = 0.0
        _t_preprocess = 0.0
        _t_apply = 0.0

        # Track action types used during this episode
        _action_type_counts = Counter()

        while num_steps < effective_max_length:
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
                _t_apply += time.time() - _t0
                break
            _t_apply += time.time() - _t0

            total_reward += reward
            step_rewards.append(reward)
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
                    break

        # Compute the episode outcome and fill value targets into all examples.
        final_t_gates = state.num_non_clifford
        t_gates_reduced = initial_t_gates - final_t_gates
        simplified = state.is_terminal()

        # --- Value targets ---
        _assign_value_targets(
            pending_examples, step_rewards, initial_t_gates,
            final_t_gates, self.config,
        )

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


@dataclass
class _ActiveGame:
    """Mutable state for one game slot in MultiGameSelfPlayWorker."""
    state: GameState
    game_id: int
    initial_t_gates: int
    num_steps: int = 0
    total_reward: float = 0.0
    done: bool = False
    examples: list[TrainingExample] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    start_time: float = 0.0
    action_type_counts: Counter = field(default_factory=Counter)


class MultiGameSelfPlayWorker:
    """Plays multiple self-play episodes concurrently with cross-game batched MCTS.

    Instead of playing one game at a time (where each MCTS search batches
    leaf evaluations from a single tree), this worker maintains K active
    game slots.  At each step, all K games' MCTS searches are interleaved
    via :meth:`MCTS.search_batch`, collecting leaf nodes across all K trees
    into combined forward passes.  This typically gives 2–3× throughput
    improvement by increasing batch utilisation for CPU inference.

    Games that finish their episodes are replaced with fresh games until
    the requested total has been played.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MCTSConfig,
        device: torch.device = torch.device('cpu'),
        concurrent_games: int = 4,
    ):
        self.mcts = MCTS(model, config)
        self.config = config
        self.device = device
        self.concurrent_games = concurrent_games
        self._game_counter = 0

    def play_episodes(
        self,
        num_games: int,
        difficulty_overrides: list[tuple[int, int]] | None = None,
    ) -> list[EpisodeResult]:
        """Play *num_games* episodes, K at a time with batched MCTS.

        :param num_games: Total number of episodes to play.
        :param difficulty_overrides: Optional per-game (num_qubits, depth) list.
                                     Games are assigned overrides in order.
        :return: List of EpisodeResult objects (one per completed game).
        """
        all_results: list[EpisodeResult] = []
        K = min(self.concurrent_games, num_games)
        t_total_start = time.time()

        # Initialise first batch of K game slots
        slots: list[_ActiveGame | None] = [
            self._new_game(
                difficulty_overrides[i] if difficulty_overrides and i < len(difficulty_overrides) else None
            )
            for i in range(K)
        ]
        games_started = K
        step_count = 0  # total search_batch calls (for logging)

        logger.info(
            f"MultiGameSelfPlayWorker: {num_games} games, "
            f"K={K} concurrent, {self.config.num_simulations} sims/step"
        )

        while any(s is not None for s in slots):
            # Collect active (non-done) game indices
            active_indices = [
                i for i, s in enumerate(slots)
                if s is not None and not s.done
            ]
            if not active_indices:
                # All remaining slots are done — finalize below
                pass
            else:
                active_games = [slots[i] for i in active_indices]

                # --- Pre-action termination checks ---
                for game in active_games:
                    if game.state.is_terminal() or not game.state.has_legal_actions():
                        game.done = True
                    elif (self.config.max_t_gate_increase > 0
                          and game.state.num_non_clifford
                              >= game.initial_t_gates + self.config.max_t_gate_increase):
                        game.done = True

                # Re-filter after termination checks
                active_indices = [
                    i for i, s in enumerate(slots)
                    if s is not None and not s.done
                ]

                if active_indices:
                    active_games = [slots[i] for i in active_indices]
                    states = [g.state for g in active_games]

                    # --- Batched MCTS search across all active games ---
                    t_search = time.time()
                    policies = self.mcts.search_batch(states, self.device)
                    t_search = time.time() - t_search
                    step_count += 1

                    if step_count <= 3 or step_count % 10 == 0:
                        logger.debug(
                            f"  search_batch: {len(states)} games, "
                            f"{t_search:.2f}s (step {step_count})"
                        )

                    for game, policy in zip(active_games, policies):
                        if not policy:
                            game.done = True
                            continue

                        # Snapshot state before applying the action
                        state_data = self._preprocess_state(game.state)
                        example = TrainingExample(
                            state_data=state_data,
                            mcts_policy=policy,
                            value_target=None,
                            game_id=game.game_id,
                        )
                        game.examples.append(example)

                        # Sample action from MCTS policy
                        actions = list(policy.keys())
                        probs = list(policy.values())
                        idx = np.random.choice(len(actions), p=probs)
                        action = actions[idx]

                        # Apply action
                        try:
                            reward, done = game.state.apply_action(action)
                        except (ValueError, KeyError, IndexError, AssertionError) as e:
                            logger.warning(
                                f"Action failed at step {game.num_steps}: {e}"
                            )
                            game.done = True
                            continue

                        game.total_reward += reward
                        game.step_rewards.append(reward)
                        game.num_steps += 1

                        # Track action type
                        action_type_idx = action[1] if len(action) > 1 else -1
                        action_name = ACTION_TYPE_NAMES.get(
                            action_type_idx, f"unknown({action_type_idx})"
                        )
                        game.action_type_counts[action_name] += 1

                        if done or game.num_steps >= self.config.effective_max_episode_length:
                            game.done = True

                        # Post-action T-gate increase check
                        if (not game.done
                                and self.config.max_t_gate_increase > 0
                                and game.state.num_non_clifford
                                    >= game.initial_t_gates + self.config.max_t_gate_increase):
                            game.done = True

            # --- Finalise done games, start new ones ---
            for i in range(len(slots)):
                if slots[i] is not None and slots[i].done:
                    result = self._finalize_episode(slots[i])
                    all_results.append(result)

                    logger.info(
                        f"Game {len(all_results)}/{num_games}: "
                        f"steps={result.num_steps}, "
                        f"t_gates={result.initial_t_gates}"
                        f"→{result.final_t_gates} "
                        f"(-{result.t_gates_reduced}), "
                        f"time={result.wall_time:.1f}s"
                    )

                    if games_started < num_games:
                        diff = (
                            difficulty_overrides[games_started]
                            if difficulty_overrides and games_started < len(difficulty_overrides)
                            else None
                        )
                        slots[i] = self._new_game(diff)
                        games_started += 1
                    else:
                        slots[i] = None

        t_total = time.time() - t_total_start
        logger.info(
            f"MultiGameSelfPlayWorker done: {len(all_results)}/{num_games} "
            f"games in {t_total:.1f}s "
            f"({len(all_results) / max(t_total, 0.001):.1f} games/s)"
        )
        return all_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_game(
        self, difficulty: tuple[int, int] | None = None,
    ) -> _ActiveGame:
        """Create a fresh game, optionally overriding difficulty."""
        self._game_counter += 1
        cfg = self.config

        if difficulty:
            saved_q, saved_d = cfg.num_qubits, cfg.depth
            cfg.num_qubits, cfg.depth = difficulty

        try:
            diagram = self._generate_circuit()
            state = GameState.from_diagram(diagram)
        finally:
            if difficulty:
                cfg.num_qubits, cfg.depth = saved_q, saved_d

        return _ActiveGame(
            state=state,
            game_id=self._game_counter,
            initial_t_gates=state.num_non_clifford,
            start_time=time.time(),
        )

    def _generate_circuit(self):
        """Generate a random ZX circuit (same logic as SelfPlayWorker)."""
        cfg = self.config
        for attempt in range(cfg.max_circuit_retries):
            if cfg.circuit_type == 'cnot_had_phase':
                diagram = cnot_had_phase_zx_diagram(
                    cfg.num_qubits, cfg.depth, cfg.p_had, cfg.p_t,
                )
            elif cfg.circuit_type == 'clifford':
                diagram = clifford_zx_diagram(
                    cfg.num_qubits, cfg.depth, t_gates=True,
                )
            else:
                raise ValueError(f"Unknown circuit_type '{cfg.circuit_type}'")
            if num_non_clifford_gates_diagram(diagram) >= cfg.min_initial_t_gates:
                return diagram
        return diagram  # fallback

    def _preprocess_state(self, state: GameState) -> Data:
        """Preprocess a GameState for replay buffer storage."""
        cached = getattr(state, '_cached_preprocessed_data', None)
        if cached is not None:
            state._cached_preprocessed_data = None
            return cached
        from alphazx.mcts.evaluate import _preprocess_data_for_model
        data = state.data.clone()
        return _preprocess_data_for_model(data, self.config.pe_dim)

    def _finalize_episode(self, game: _ActiveGame) -> EpisodeResult:
        """Compute value targets and build EpisodeResult."""
        final_t = game.state.num_non_clifford
        t_reduced = game.initial_t_gates - final_t
        simplified = game.state.is_terminal()

        # Value targets: uses shared helper (supports discounted returns)
        _assign_value_targets(
            game.examples, game.step_rewards, game.initial_t_gates,
            final_t, self.config,
        )

        wall_time = time.time() - game.start_time

        return EpisodeResult(
            num_steps=game.num_steps,
            total_reward=game.total_reward,
            initial_t_gates=game.initial_t_gates,
            final_t_gates=final_t,
            t_gates_reduced=t_reduced,
            simplified=simplified,
            wall_time=wall_time,
            examples=game.examples,
        )


class SelfPlayManager:
    """Orchestrates self-play game generation and feeds examples into the replay buffer."""

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
