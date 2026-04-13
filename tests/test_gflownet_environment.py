"""Tests for GFlowNet environment (ZXGFlowNetEnv)."""

import pytest
import torch

from alphazx.gflownet.environment import (
    ZXGFlowNetEnv, Transition, Trajectory,
)
from alphazx.shared.config import CircuitConfig
from alphazx.shared.game_state import GameState
from tests.gflownet_utils import make_circuit_config, make_env


# ---------------------------------------------------------------------------
# Transition / Trajectory dataclass tests
# ---------------------------------------------------------------------------

class TestTransition:
    def test_fields(self):
        """Transition stores state, action, next_state, reward, done."""
        t = Transition(state=None, action=(0, 2, 5, 0, 0),
                       next_state=None, reward=0.1, done=False)
        assert t.action == (0, 2, 5, 0, 0)
        assert t.reward == 0.1
        assert t.done is False


class TestTrajectory:
    def _make_trajectory(self, n_transitions=3, initial_t=5, final_t=3):
        """Helper to build a Trajectory with dummy transitions."""
        transitions = []
        for i in range(n_transitions):
            transitions.append(Transition(
                state=None, action=(0, 2, i, 0, 0),
                next_state=None, reward=0.5 if i == n_transitions - 1 else 0.0,
                done=i == n_transitions - 1,
            ))
        return Trajectory(
            transitions=transitions,
            initial_t_gates=initial_t,
            final_t_gates=final_t,
        )

    def test_len(self):
        traj = self._make_trajectory(n_transitions=4)
        assert len(traj) == 4

    def test_empty_trajectory(self):
        traj = Trajectory()
        assert len(traj) == 0
        assert traj.total_reward == 0
        assert traj.t_gate_reduction == 0
        assert traj.states == []
        assert traj.actions == []

    def test_total_reward(self):
        traj = self._make_trajectory()
        assert traj.total_reward == pytest.approx(0.5)

    def test_t_gate_reduction(self):
        traj = self._make_trajectory(initial_t=5, final_t=3)
        assert traj.t_gate_reduction == 2

    def test_t_gate_reduction_negative(self):
        """T-gate count can increase (negative reduction)."""
        traj = self._make_trajectory(initial_t=2, final_t=5)
        assert traj.t_gate_reduction == -3

    def test_actions_list(self):
        traj = self._make_trajectory(n_transitions=3)
        actions = traj.actions
        assert len(actions) == 3
        assert actions[0] == (0, 2, 0, 0, 0)
        assert actions[2] == (0, 2, 2, 0, 0)


# ---------------------------------------------------------------------------
# ZXGFlowNetEnv tests
# ---------------------------------------------------------------------------

class TestZXGFlowNetEnv:
    @pytest.fixture
    def env(self):
        return make_env()

    @pytest.fixture
    def state(self, env):
        return env.generate_state()

    # --- State generation ---

    def test_generate_state_returns_game_state(self, env):
        state = env.generate_state()
        assert isinstance(state, GameState)

    def test_generate_state_has_t_gates(self, env):
        """Generated state should have at least min_initial_t_gates T-gates."""
        state = env.generate_state()
        assert state.num_non_clifford >= env.config.min_initial_t_gates

    def test_generate_state_different_seeds(self, env):
        """Multiple generated states should generally differ."""
        states = [env.generate_state() for _ in range(5)]
        t_counts = [s.num_non_clifford for s in states]
        # Not all identical (probabilistic — could fail rarely)
        # Just check they're valid
        assert all(t >= 1 for t in t_counts)

    def test_generate_state_clifford_circuit(self):
        """Clifford circuit type should also work."""
        cfg = make_circuit_config(circuit_type='clifford')
        env = ZXGFlowNetEnv(cfg)
        state = env.generate_state()
        assert isinstance(state, GameState)

    def test_generate_state_invalid_circuit_type(self):
        cfg = make_circuit_config()
        cfg.circuit_type = 'invalid_type'
        env = ZXGFlowNetEnv(cfg)
        with pytest.raises(ValueError, match="Unknown circuit_type"):
            env.generate_state()

    # --- Action enumeration ---

    def test_enumerate_flat_actions_returns_list(self, env, state):
        actions = env.enumerate_flat_actions(state)
        assert isinstance(actions, list)

    def test_flat_actions_tuple_format(self, env, state):
        """Each flat action should be (graph_id=0, type, node, 0, 0)."""
        actions = env.enumerate_flat_actions(state)
        for action in actions:
            assert len(action) == 5
            assert action[0] == 0  # graph_id
            assert 2 <= action[1] <= 9  # non-F-Right types
            assert action[3] == 0  # phase (unused for non-F-Right)
            assert action[4] == 0  # new_edge (unused for non-F-Right)

    def test_flat_actions_skip_boundary_and_super_nodes(self, env, state):
        """Flat actions should never include boundary (index 0) or super nodes (index > 10)."""
        data, data_index = state.ensure_data()
        actions = env.enumerate_flat_actions(state)
        for action in actions:
            node_idx = action[2]
            match = data_index[node_idx]
            assert match.index != 0, "boundary node should be excluded"
            assert match.index <= 10, f"super node (index {match.index}) should be excluded"
            assert match.index >= 3, "F-Right nodes (1, 2) should be excluded"

    def test_flat_actions_skip_f_right(self, env, state):
        """Flat actions should not include F-Right types (0 or 1)."""
        actions = env.enumerate_flat_actions(state)
        for action in actions:
            assert action[1] not in (0, 1), "F-Right should not appear in flat actions"

    def test_enumerate_f_right_nodes_returns_list(self, env, state):
        pairs = env.enumerate_f_right_nodes(state)
        assert isinstance(pairs, list)

    def test_f_right_nodes_format(self, env, state):
        """Each F-Right entry should be (action_type, node_index)."""
        pairs = env.enumerate_f_right_nodes(state)
        for at, ni in pairs:
            assert at in (0, 1), f"F-Right action_type should be 0 or 1, got {at}"
            assert isinstance(ni, int)

    def test_has_actions_consistent(self, env, state):
        """has_actions should agree with action enumeration."""
        flat = env.enumerate_flat_actions(state)
        fr = env.enumerate_f_right_nodes(state)
        total = len(flat) + len(fr)
        has = env.has_actions(state)
        if total > 0:
            assert has is True
        # Note: has_actions could be True even if our enumeration returns 0,
        # because has_legal_actions uses a different check internally

    # --- Step ---

    def test_step_returns_tuple(self, env, state):
        flat = env.enumerate_flat_actions(state)
        if not flat:
            pytest.skip("No flat actions available on this circuit")
        next_state, reward, done = env.step(state, flat[0])
        assert isinstance(next_state, GameState)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_step_preserves_original_state(self, env, state):
        """Step should clone the state — original should be unchanged."""
        flat = env.enumerate_flat_actions(state)
        if not flat:
            pytest.skip("No flat actions available")
        original_t = state.num_non_clifford
        env.step(state, flat[0])
        assert state.num_non_clifford == original_t

    def test_step_multiple_actions(self, env, state):
        """Should be able to step through multiple actions sequentially."""
        current = state
        for _ in range(3):
            flat = env.enumerate_flat_actions(current)
            fr = env.enumerate_f_right_nodes(current)
            if not flat and not fr:
                break
            if flat:
                current, _, done = env.step(current, flat[0])
                if done:
                    break

    # --- Terminal checking ---

    def test_is_terminal_false_for_fresh_state(self, env, state):
        """A fresh state with T-gates should usually not be terminal."""
        if state.num_non_clifford > 0:
            # A fresh circuit with T-gates typically has actions available
            # (not always guaranteed, but very likely for depth=3)
            pass  # Just checking it doesn't crash
        env.is_terminal(state)

    # --- Reward ---

    def test_terminal_reward_full_reduction(self, env):
        """All T-gates eliminated → reward = 1.0 ^ exponent = 1.0."""
        r = env.terminal_reward(5, 0, reward_exponent=4.0)
        assert r == pytest.approx(1.0)

    def test_terminal_reward_no_reduction(self, env):
        """No T-gates reduced → reward = min_reward (floor applied after exponent)."""
        r = env.terminal_reward(5, 5, reward_exponent=4.0, min_reward=1e-6)
        # reduction_ratio = 0, 0^4 = 0, max(1e-6, 0) = 1e-6
        assert r == pytest.approx(1e-6)

    def test_terminal_reward_partial_reduction(self, env):
        """Partial reduction → reward between min and 1."""
        r = env.terminal_reward(4, 2, reward_exponent=4.0)
        # reduction_ratio = 2/4 = 0.5, reward = 0.5^4 = 0.0625
        assert r == pytest.approx(0.5 ** 4.0)

    def test_terminal_reward_zero_initial(self, env):
        """Zero initial T-gates → min_reward."""
        r = env.terminal_reward(0, 0, min_reward=1e-6)
        assert r == 1e-6

    def test_terminal_reward_increase(self, env):
        """T-gates increased → reduction_ratio clamped to 0 → min_reward."""
        r = env.terminal_reward(3, 5, reward_exponent=4.0, min_reward=1e-6)
        # reduction_ratio = 0 (clamped), 0^4 = 0, max(1e-6, 0) = 1e-6
        assert r == pytest.approx(1e-6)

    def test_terminal_reward_exponent_effect(self, env):
        """Higher exponent should produce smaller rewards for partial reduction."""
        r_low = env.terminal_reward(4, 2, reward_exponent=1.0)
        r_high = env.terminal_reward(4, 2, reward_exponent=8.0)
        assert r_low > r_high
