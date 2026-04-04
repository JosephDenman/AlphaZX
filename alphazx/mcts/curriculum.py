"""
Curriculum learning scheduler for AlphaZX training.

Progressively increases circuit difficulty (num_qubits and depth) based on the
agent's performance, so that the agent first learns to simplify small circuits
and gradually develops tactics for larger ones.

Two strategies are supported:

- **performance** (recommended): Advance to the next difficulty level when the
  agent achieves a sustained simplification rate above a configurable threshold
  for several consecutive iterations.  This ensures the agent has genuinely
  learned the current level before moving on.

- **linear**: Advance on a fixed iteration schedule regardless of performance.
  Simpler to configure but risks advancing before the agent is ready (or
  wasting iterations on already-mastered levels).

Both strategies support *mixed-difficulty sampling*: at each iteration, most
games are generated at the current difficulty level, but a configurable fraction
are drawn from easier or harder levels.  This keeps a mix of difficulties in
the replay buffer, smoothing the transition between levels and preventing
catastrophic forgetting of earlier skills.

Usage::

    from alphazx.mcts.curriculum import CurriculumScheduler, CurriculumConfig

    curriculum = CurriculumScheduler(CurriculumConfig(
        enabled=True,
        start_num_qubits=2, start_depth=3,
        target_num_qubits=10, target_depth=10,
    ))

    # At the start of each training iteration:
    curriculum.update(mcts_config, iteration_metrics)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = False
    """Master switch.  When False, the curriculum scheduler is a no-op and
    the training loop uses fixed num_qubits/depth from MCTSConfig."""

    # --- Difficulty range ---

    start_num_qubits: int = 2
    """Number of qubits at the easiest difficulty level."""

    start_depth: int = 3
    """Circuit depth at the easiest difficulty level."""

    target_num_qubits: int = 10
    """Number of qubits at the final difficulty level."""

    target_depth: int = 10
    """Circuit depth at the final difficulty level."""

    # --- Advancement criteria (performance strategy) ---

    strategy: str = 'performance'
    """Advancement strategy: 'performance' or 'linear'.

    - 'performance': advance when avg_t_gates_reduced exceeds
      advance_threshold for advance_window consecutive iterations.
    - 'linear': advance every linear_advance_every iterations."""

    advance_threshold: float = 0.5
    """For 'performance' strategy: minimum average T-gate reduction
    (as a fraction of initial T-gates) required to advance.  0.5 means
    the agent must on average reduce T-gates by at least 50%."""

    advance_window: int = 3
    """For 'performance' strategy: number of consecutive iterations
    that must meet the threshold before advancing."""

    # --- Linear strategy ---

    linear_advance_every: int = 10
    """For 'linear' strategy: advance every N iterations."""

    # --- Step sizes ---

    qubit_step: int = 1
    """How many qubits to add per advancement."""

    depth_step: int = 2
    """How much depth to add per advancement."""

    # --- Mixed-difficulty sampling ---

    mix_easier_fraction: float = 0.1
    """Fraction of self-play games to generate at an easier difficulty
    (one level below current).  Helps prevent catastrophic forgetting."""

    mix_harder_fraction: float = 0.1
    """Fraction of self-play games to generate at the next difficulty
    level (one level above current).  Provides exposure to upcoming
    challenges before the formal advancement."""


class CurriculumScheduler:
    """Manages progressive difficulty increases during training.

    Call :meth:`update` at the start of each training iteration to
    (potentially) advance the difficulty level and adjust the MCTSConfig.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_level: int = 0
        self._performance_history: deque[float] = deque(
            maxlen=config.advance_window,
        )

        # Pre-compute the sequence of (num_qubits, depth) levels
        self.levels: list[tuple[int, int]] = self._compute_levels()
        self.max_level: int = len(self.levels) - 1

        if config.enabled:
            q, d = self.levels[0]
            logger.info(
                f"Curriculum enabled: {len(self.levels)} difficulty levels, "
                f"starting at ({q}q, d={d}), "
                f"target ({config.target_num_qubits}q, d={config.target_depth})"
            )

    def _compute_levels(self) -> list[tuple[int, int]]:
        """Build the ordered list of (num_qubits, depth) difficulty levels.

        Qubits and depth advance in tandem.  If the step sizes don't evenly
        divide the range, the final level is clamped to the target.
        """
        cfg = self.config
        levels = []
        q, d = cfg.start_num_qubits, cfg.start_depth

        while q <= cfg.target_num_qubits or d <= cfg.target_depth:
            levels.append((
                min(q, cfg.target_num_qubits),
                min(d, cfg.target_depth),
            ))
            q += cfg.qubit_step
            d += cfg.depth_step

        # Ensure the final target is always included
        target = (cfg.target_num_qubits, cfg.target_depth)
        if not levels or levels[-1] != target:
            levels.append(target)

        return levels

    @property
    def current_num_qubits(self) -> int:
        return self.levels[self.current_level][0]

    @property
    def current_depth(self) -> int:
        return self.levels[self.current_level][1]

    @property
    def at_target(self) -> bool:
        """True if the curriculum has reached the final difficulty level."""
        return self.current_level >= self.max_level

    def update(
        self,
        mcts_config,
        iteration: int,
        avg_t_gates_reduced: float,
        avg_initial_t_gates: float,
        simplification_rate: float,
    ) -> bool:
        """Check whether to advance the difficulty level and update config.

        :param mcts_config: The MCTSConfig to modify in place.
        :param iteration: Current training iteration number.
        :param avg_t_gates_reduced: Average T-gates reduced this iteration.
        :param avg_initial_t_gates: Average initial T-gates this iteration.
        :param simplification_rate: Fraction of games fully simplified.
        :return: True if the level was advanced.
        """
        if not self.config.enabled or self.at_target:
            return False

        advanced = False
        cfg = self.config

        if cfg.strategy == 'performance':
            advanced = self._check_performance_advance(
                avg_t_gates_reduced, avg_initial_t_gates, simplification_rate,
            )
        elif cfg.strategy == 'linear':
            advanced = self._check_linear_advance(iteration)
        else:
            raise ValueError(f"Unknown curriculum strategy: '{cfg.strategy}'")

        if advanced:
            old_q, old_d = self.levels[self.current_level - 1]
            new_q, new_d = self.levels[self.current_level]
            logger.info(
                f"Curriculum advanced to level {self.current_level}/{self.max_level}: "
                f"({old_q}q, d={old_d}) → ({new_q}q, d={new_d})"
            )
            self._performance_history.clear()

        # Apply current level to MCTSConfig
        mcts_config.num_qubits = self.current_num_qubits
        mcts_config.depth = self.current_depth

        return advanced

    def _check_performance_advance(
        self,
        avg_t_gates_reduced: float,
        avg_initial_t_gates: float,
        simplification_rate: float,
    ) -> bool:
        """Advance if the agent's T-gate reduction ratio exceeds the threshold
        for enough consecutive iterations."""
        if avg_initial_t_gates < 1:
            # Avoid division by zero for trivial circuits
            reduction_ratio = 0.0
        else:
            reduction_ratio = avg_t_gates_reduced / avg_initial_t_gates

        self._performance_history.append(reduction_ratio)

        if len(self._performance_history) < self.config.advance_window:
            return False

        # Check if ALL recent iterations meet the threshold
        if all(r >= self.config.advance_threshold for r in self._performance_history):
            if self.current_level < self.max_level:
                self.current_level += 1
                return True

        return False

    def _check_linear_advance(self, iteration: int) -> bool:
        """Advance on a fixed schedule."""
        if iteration > 0 and iteration % self.config.linear_advance_every == 0:
            if self.current_level < self.max_level:
                self.current_level += 1
                return True
        return False

    def get_mixed_difficulty_levels(
        self,
        num_games: int,
    ) -> list[tuple[int, int]]:
        """Return a list of (num_qubits, depth) for each self-play game,
        implementing mixed-difficulty sampling.

        Most games use the current level.  A fraction use one level easier
        (if available) and one level harder (if available).

        :param num_games: Total number of games to generate.
        :return: List of (num_qubits, depth) tuples, one per game.
        """
        if not self.config.enabled:
            # No curriculum — all games at config defaults
            return []

        cfg = self.config
        current = self.levels[self.current_level]

        n_easier = int(num_games * cfg.mix_easier_fraction)
        n_harder = int(num_games * cfg.mix_harder_fraction)
        n_current = num_games - n_easier - n_harder

        levels = [current] * n_current

        # Easier games
        if self.current_level > 0:
            easier = self.levels[self.current_level - 1]
            levels.extend([easier] * n_easier)
        else:
            # At easiest level — use current for all
            levels.extend([current] * n_easier)

        # Harder games (preview of next level)
        if self.current_level < self.max_level:
            harder = self.levels[self.current_level + 1]
            levels.extend([harder] * n_harder)
        else:
            # At hardest level — use current for all
            levels.extend([current] * n_harder)

        return levels

    def state_dict(self) -> dict:
        """Serialize curriculum state for checkpointing."""
        return {
            'current_level': self.current_level,
            'performance_history': list(self._performance_history),
            'config': self.config,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state from a checkpoint."""
        self.current_level = state['current_level']
        self._performance_history = deque(
            state['performance_history'],
            maxlen=self.config.advance_window,
        )
