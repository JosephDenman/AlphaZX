"""
Prioritized replay buffer for GFlowNet training.

Stores high-reward trajectories and samples them with priority
proportional to reward quality and action-type diversity.  This
combats mode collapse by re-exposing the model to its best
discoveries and ensuring diverse action patterns are represented.

Design decisions
----------------
- **Reward-gated admission**: Only trajectories above ``min_reward``
  are stored, preventing the buffer from filling with degenerate
  trajectories that reinforce mode collapse.
- **Priority = reward_rank + diversity_weight × diversity_score**:
  Reward rank ensures high-reward trajectories are replayed often.
  Diversity score ensures rare action-type patterns are not forgotten.
- **Action-type fingerprinting**: Diversity is measured by how rare a
  trajectory's sequence of action types is relative to other buffer
  entries.  This is cheap to compute and directly targets the observed
  failure mode (all trajectories using the same non-F-Right rewrites).
- **Stored data**: initial GameState + action tuple sequence — the
  minimum needed to replay through the current policy for fresh
  gradients.

References
----------
- Shen et al. (2023), "Towards Understanding and Improving GFlowNet
  Training", ICML.  Up to 10× sample efficiency from prioritized replay.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from alphazx.shared.game_state import GameState

logger = logging.getLogger(__name__)


@dataclass
class ReplayEntry:
    """A stored trajectory for replay.

    Stores the minimum data needed to re-run the trajectory through
    the current policy.  Two replay paths are supported:

    1. **State-based** (sequential sampling): ``initial_state`` is a live
       GameState; replay applies ``action_tuples`` step-by-step through the
       environment (``replay_rollout``).
    2. **Tensor-based** (parallel sampling): ``trajectory_record`` holds the
       pre-exported PyG tensors from the worker.  Replay teacher-forces the
       current model on those tensors (``replay_records_batched``), avoiding
       GameState reconstruction and its hash-dependent match ordering.
    """
    initial_state: GameState | None
    action_tuples: list[tuple]
    terminal_reward: float
    shaped_reward: float
    initial_t_gates: int
    final_t_gates: int
    per_rewrite_t_deltas: list[int]
    action_type_fingerprint: tuple[int, ...]
    trajectory_record: object = None  # TrajectoryRecord, typed as object to avoid circular import

    @property
    def t_gate_reduction(self) -> int:
        return self.initial_t_gates - self.final_t_gates


class PrioritizedReplayBuffer:
    """Replay buffer that prioritizes high-reward, diverse trajectories.

    Priority scoring
    ~~~~~~~~~~~~~~~~
    Each entry receives a priority score:

        priority = reward_score + diversity_weight × diversity_score

    ``reward_score`` is the entry's shaped reward normalized to [0, 1]
    by the buffer's maximum shaped reward.

    ``diversity_score`` is 1 / (count of entries sharing the same
    action-type fingerprint).  Rare fingerprints score higher.

    Sampling
    ~~~~~~~~
    Entries are sampled proportionally to their priority scores
    (softmax with temperature=1).  Ties are broken randomly.

    Eviction
    ~~~~~~~~
    When the buffer is full, the lowest-priority entry is evicted.
    """

    def __init__(
        self,
        max_size: int = 1000,
        min_reward: float = 0.02,
        diversity_weight: float = 0.1,
        min_reduction_ratio: float = 0.0,
    ):
        self.max_size = max_size
        self.min_reward = min_reward
        self.diversity_weight = diversity_weight
        self.min_reduction_ratio = min_reduction_ratio

        self._entries: list[ReplayEntry] = []
        self._fingerprint_counts: Counter = Counter()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def max_reward(self) -> float:
        """Maximum shaped reward in the buffer."""
        if not self._entries:
            return 0.0
        return max(e.shaped_reward for e in self._entries)

    @property
    def mean_reward(self) -> float:
        if not self._entries:
            return 0.0
        return sum(e.shaped_reward for e in self._entries) / len(self._entries)

    @property
    def num_unique_fingerprints(self) -> int:
        return len(self._fingerprint_counts)

    def add(self, entry: ReplayEntry) -> bool:
        """Add a trajectory to the buffer if it meets the quality threshold.

        Returns True if the entry was accepted, False if rejected.

        When ``min_reduction_ratio > 0``, admission is gated on the raw
        T-count reduction ratio ``(initial_T - final_T) / initial_T``,
        which is independent of the reward exponent.  Otherwise falls
        back to the legacy ``shaped_reward > min_reward`` check.
        """
        # Gate: reject degenerate trajectories
        if self.min_reduction_ratio > 0:
            # Use exponent-independent reduction ratio
            if entry.initial_t_gates <= 0:
                return False
            ratio = (entry.initial_t_gates - entry.final_t_gates) / entry.initial_t_gates
            if ratio < self.min_reduction_ratio:
                return False
        else:
            # Legacy: filter on shaped reward (exponent-dependent)
            if entry.shaped_reward <= self.min_reward:
                return False

        # Update fingerprint counts
        self._fingerprint_counts[entry.action_type_fingerprint] += 1

        if len(self._entries) < self.max_size:
            self._entries.append(entry)
            return True

        # Buffer full → evict lowest priority entry
        priorities = self._compute_priorities()
        min_idx = min(range(len(priorities)), key=lambda i: priorities[i])

        # Only evict if new entry has higher priority than the worst
        new_priority = self._entry_priority(entry)
        if new_priority > priorities[min_idx]:
            evicted = self._entries[min_idx]
            self._fingerprint_counts[evicted.action_type_fingerprint] -= 1
            if self._fingerprint_counts[evicted.action_type_fingerprint] <= 0:
                del self._fingerprint_counts[evicted.action_type_fingerprint]
            self._entries[min_idx] = entry
            return True
        else:
            # New entry is worse than everything in the buffer — undo fingerprint count
            self._fingerprint_counts[entry.action_type_fingerprint] -= 1
            if self._fingerprint_counts[entry.action_type_fingerprint] <= 0:
                del self._fingerprint_counts[entry.action_type_fingerprint]
            return False

    def sample(self, n: int) -> list[ReplayEntry]:
        """Sample n entries proportionally to their priority scores.

        Returns fewer than n entries if the buffer is smaller.
        """
        if not self._entries:
            return []

        n = min(n, len(self._entries))
        priorities = self._compute_priorities()

        # Convert to sampling weights (softmax-like, but just normalize)
        total = sum(priorities)
        if total <= 0:
            # Uniform fallback
            return random.sample(self._entries, n)

        weights = [p / total for p in priorities]
        # Weighted sampling without replacement
        indices = []
        remaining_weights = list(weights)
        remaining_indices = list(range(len(self._entries)))

        for _ in range(n):
            if not remaining_indices:
                break
            total_w = sum(remaining_weights)
            if total_w <= 0:
                # Uniform fallback for remaining
                idx = random.choice(remaining_indices)
            else:
                r = random.random() * total_w
                cumulative = 0.0
                idx = remaining_indices[-1]
                for j, (ri, w) in enumerate(zip(remaining_indices, remaining_weights)):
                    cumulative += w
                    if cumulative >= r:
                        idx = ri
                        remaining_weights.pop(j)
                        remaining_indices.pop(j)
                        break
                else:
                    remaining_weights.pop(-1)
                    remaining_indices.pop(-1)
            indices.append(idx)

        return [self._entries[i] for i in indices]

    def _compute_priorities(self) -> list[float]:
        """Compute priority scores for all entries."""
        return [self._entry_priority(e) for e in self._entries]

    def _entry_priority(self, entry: ReplayEntry) -> float:
        """Compute priority for a single entry."""
        # Reward score: normalized by buffer max (or 1.0 if buffer empty)
        max_r = self.max_reward
        if max_r > 0:
            reward_score = entry.shaped_reward / max_r
        else:
            reward_score = 0.0

        # Diversity score: inverse of fingerprint count
        fp_count = self._fingerprint_counts.get(entry.action_type_fingerprint, 1)
        diversity_score = 1.0 / max(1, fp_count)

        return reward_score + self.diversity_weight * diversity_score

    def stats(self) -> dict:
        """Return summary statistics for logging."""
        if not self._entries:
            return {
                'size': 0, 'max_reward': 0.0, 'mean_reward': 0.0,
                'unique_fingerprints': 0,
            }
        rewards = [e.shaped_reward for e in self._entries]
        return {
            'size': len(self._entries),
            'max_reward': max(rewards),
            'mean_reward': sum(rewards) / len(rewards),
            'min_reward': min(rewards),
            'unique_fingerprints': self.num_unique_fingerprints,
            'mean_t_reduction': sum(e.t_gate_reduction for e in self._entries) / len(self._entries),
        }
