"""Tests for the GFlowNet PrioritizedReplayBuffer.

Focuses on admission gating: the new ``min_reduction_ratio`` filter
ensures trajectories are admitted based on the raw T-count reduction
fraction rather than the exponent-dependent shaped reward.
"""

import unittest

from alphazx.gflownet.replay_buffer import PrioritizedReplayBuffer, ReplayEntry


def _make_entry(
    initial_t: int = 20,
    final_t: int = 18,
    shaped_reward: float = 0.001,
    terminal_reward: float = 0.001,
    fingerprint: tuple[int, ...] = (0, 1),
) -> ReplayEntry:
    """Create a minimal ReplayEntry for testing."""
    return ReplayEntry(
        initial_state=None,  # type: ignore[arg-type]
        action_tuples=[(0, 1, 2)],
        terminal_reward=terminal_reward,
        shaped_reward=shaped_reward,
        initial_t_gates=initial_t,
        final_t_gates=final_t,
        per_rewrite_t_deltas=[initial_t - final_t],
        action_type_fingerprint=fingerprint,
    )


class TestReductionRatioAdmission(unittest.TestCase):
    """Tests for min_reduction_ratio gating in PrioritizedReplayBuffer."""

    def test_admits_trajectory_above_ratio_threshold(self):
        """A trajectory with 10% reduction should be admitted at 2% threshold."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        # 20 → 18 = 10% reduction, well above 2%
        entry = _make_entry(initial_t=20, final_t=18)
        self.assertTrue(buf.add(entry))
        self.assertEqual(len(buf), 1)

    def test_rejects_trajectory_below_ratio_threshold(self):
        """A trajectory with 1% reduction should be rejected at 2% threshold."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        # 100 → 99 = 1% reduction, below 2%
        entry = _make_entry(initial_t=100, final_t=99)
        self.assertFalse(buf.add(entry))
        self.assertEqual(len(buf), 0)

    def test_rejects_no_reduction(self):
        """A trajectory with 0% reduction should always be rejected."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        entry = _make_entry(initial_t=20, final_t=20)
        self.assertFalse(buf.add(entry))

    def test_rejects_negative_reduction(self):
        """A trajectory that increased T-count should be rejected."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        entry = _make_entry(initial_t=20, final_t=22)
        self.assertFalse(buf.add(entry))

    def test_rejects_zero_initial_t_gates(self):
        """A trajectory with 0 initial T-gates has no meaningful ratio."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        entry = _make_entry(initial_t=0, final_t=0)
        self.assertFalse(buf.add(entry))

    def test_ratio_independent_of_reward_exponent(self):
        """Admission should work regardless of how small the shaped reward is.

        This is the core scenario: with reward_exponent=4, a 10% reduction
        gives shaped_reward ≈ 0.10^4 = 1e-4, far below the legacy
        min_reward=0.02.  But the reduction ratio (0.10) is well above 0.02.
        """
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.02,
        )
        # 10% reduction, but tiny shaped_reward (as if exponent=4)
        entry = _make_entry(
            initial_t=20, final_t=18,
            shaped_reward=1e-4, terminal_reward=1e-4,
        )
        self.assertTrue(buf.add(entry))
        self.assertEqual(len(buf), 1)

    def test_exact_threshold_boundary(self):
        """A trajectory exactly at the threshold should be admitted (>=)."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.10,
        )
        # Exactly 10% reduction
        entry = _make_entry(initial_t=20, final_t=18)
        self.assertTrue(buf.add(entry))

    def test_just_below_threshold_rejected(self):
        """A trajectory just below the threshold should be rejected."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.10,
        )
        # 9.5% reduction — below 10%
        entry = _make_entry(initial_t=200, final_t=181)
        self.assertFalse(buf.add(entry))


class TestLegacyAdmission(unittest.TestCase):
    """Tests that legacy min_reward gating still works when ratio=0."""

    def test_legacy_admits_above_min_reward(self):
        """When min_reduction_ratio=0, falls back to shaped_reward filter."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reward=0.02, min_reduction_ratio=0.0,
        )
        entry = _make_entry(shaped_reward=0.05)
        self.assertTrue(buf.add(entry))

    def test_legacy_rejects_below_min_reward(self):
        """When min_reduction_ratio=0, shaped_reward below threshold is rejected."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reward=0.02, min_reduction_ratio=0.0,
        )
        entry = _make_entry(shaped_reward=0.01)
        self.assertFalse(buf.add(entry))


class TestBufferEvictionWithRatio(unittest.TestCase):
    """Tests that eviction still works correctly with the new admission."""

    def test_full_buffer_evicts_lowest_priority(self):
        """When buffer is full, lowest-priority entry is evicted."""
        buf = PrioritizedReplayBuffer(
            max_size=3, min_reduction_ratio=0.01,
        )
        # Fill with 5% reduction entries
        for i in range(3):
            entry = _make_entry(
                initial_t=20, final_t=19,
                shaped_reward=0.05 + i * 0.01,
                fingerprint=(i,),
            )
            buf.add(entry)
        self.assertEqual(len(buf), 3)

        # Add a much better entry (50% reduction, high reward)
        better = _make_entry(
            initial_t=20, final_t=10,
            shaped_reward=1.0,
            fingerprint=(99,),
        )
        self.assertTrue(buf.add(better))
        self.assertEqual(len(buf), 3)  # still 3, one was evicted

    def test_sample_returns_entries(self):
        """Basic sanity: sample returns the right number of entries."""
        buf = PrioritizedReplayBuffer(
            max_size=10, min_reduction_ratio=0.01,
        )
        for i in range(5):
            entry = _make_entry(
                initial_t=20, final_t=19 - i,
                shaped_reward=0.1 * (i + 1),
                fingerprint=(i,),
            )
            buf.add(entry)
        samples = buf.sample(3)
        self.assertEqual(len(samples), 3)


if __name__ == '__main__':
    unittest.main()
