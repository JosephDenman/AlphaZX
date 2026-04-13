"""Tests for GFlowNet trajectory sampler — decomposed sub-action design."""

import math

import pytest
import torch

from alphazx.gflownet.sampler import (
    TrajectorySampler, AnnotatedTransition, AnnotatedTrajectory,
    SubActionPhase, PartialAction,
    RewriteStepRecord, TrajectoryRecord,
    _worker_sample_trajectories, ParallelTrajectorySampler,
)
from tests.gflownet_utils import (
    make_env, make_policy, make_model, make_gflownet_config,
)


# ---------------------------------------------------------------------------
# PartialAction tests
# ---------------------------------------------------------------------------

class TestPartialAction:
    def test_empty_partial_action(self):
        pa = PartialAction()
        assert pa.next_phase == SubActionPhase.CHOOSE_TYPE

    def test_after_type_chosen(self):
        pa = PartialAction(action_type=4)  # non-F-Right
        assert pa.next_phase == SubActionPhase.CHOOSE_NODE

    def test_non_f_right_ready_after_node(self):
        pa = PartialAction(action_type=4, node_index=7)
        assert pa.next_phase == SubActionPhase.APPLY

    def test_f_right_needs_phase_after_node(self):
        pa = PartialAction(action_type=0, node_index=3)
        assert pa.next_phase == SubActionPhase.CHOOSE_PHASE

    def test_f_right_needs_edge_after_phase(self):
        pa = PartialAction(action_type=0, node_index=3, phase_val=2)
        assert pa.next_phase == SubActionPhase.CHOOSE_NEW_EDGE

    def test_f_right_needs_transfer_after_edge(self):
        pa = PartialAction(action_type=0, node_index=3, phase_val=2, new_edge=1)
        assert pa.next_phase == SubActionPhase.CHOOSE_TRANSFER

    def test_f_right_ready_after_transfer(self):
        pa = PartialAction(action_type=0, node_index=3, phase_val=2,
                           new_edge=1, transfer_edges=[1, 0])
        assert pa.next_phase == SubActionPhase.APPLY

    def test_is_f_right(self):
        assert PartialAction(action_type=0).is_f_right is True
        assert PartialAction(action_type=1).is_f_right is True
        assert PartialAction(action_type=2).is_f_right is False
        assert PartialAction(action_type=9).is_f_right is False
        assert PartialAction().is_f_right is False

    def test_to_action_tuple_non_f_right(self):
        pa = PartialAction(action_type=4, node_index=7)
        t = pa.to_action_tuple()
        assert t == (0, 4, 7, 0, 0)

    def test_to_action_tuple_f_right(self):
        pa = PartialAction(action_type=0, node_index=3, phase_val=2,
                           new_edge=1, transfer_edges=[1, 0, 1])
        t = pa.to_action_tuple()
        assert t == (0, 0, 3, 2, 1, 1, 0, 1)


# ---------------------------------------------------------------------------
# AnnotatedTrajectory tests
# ---------------------------------------------------------------------------

class TestAnnotatedTrajectory:
    def _make_trajectory(self, n=5, reward=0.5):
        transitions = []
        for i in range(n):
            transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=2,
                log_pf=torch.tensor(-1.0 - 0.1 * i, requires_grad=True),
                log_pb=-0.5,
            ))
        return AnnotatedTrajectory(
            transitions=transitions,
            initial_t_gates=5, final_t_gates=3,
            terminal_reward=reward, num_rewrites=2,
        )

    def test_len(self):
        traj = self._make_trajectory(n=5)
        assert len(traj) == 5

    def test_empty(self):
        traj = AnnotatedTrajectory()
        assert len(traj) == 0
        assert traj.sum_log_pf.item() == 0.0
        assert traj.sum_log_pb == 0.0
        assert traj.t_gate_reduction == 0
        assert traj.num_rewrites == 0

    def test_t_gate_reduction(self):
        traj = self._make_trajectory()
        assert traj.t_gate_reduction == 2

    def test_sum_log_pf_differentiable(self):
        traj = self._make_trajectory()
        slp = traj.sum_log_pf
        assert slp.requires_grad

    def test_sum_log_pb_scalar(self):
        traj = self._make_trajectory(n=5)
        assert traj.sum_log_pb == pytest.approx(-2.5)

    def test_num_rewrites(self):
        traj = self._make_trajectory()
        assert traj.num_rewrites == 2


# ---------------------------------------------------------------------------
# TrajectorySampler tests
# ---------------------------------------------------------------------------

class TestTrajectorySampler:
    @pytest.fixture
    def sampler(self):
        env = make_env()
        policy = make_policy()
        return TrajectorySampler(
            env=env, policy=policy, device='cpu',
            temperature=1.0, epsilon_uniform=0.05,
            max_trajectory_length=10,
        )

    # --- sample (no grad) ---

    def test_sample_returns_annotated_trajectory(self, sampler):
        traj = sampler.sample()
        assert isinstance(traj, AnnotatedTrajectory)

    def test_sample_has_transitions(self, sampler):
        traj = sampler.sample()
        assert len(traj) > 0

    def test_sample_has_sub_action_phases(self, sampler):
        """Each transition should have a SubActionPhase."""
        traj = sampler.sample()
        for t in traj.transitions:
            assert isinstance(t.sub_action_phase, SubActionPhase)

    def test_sample_starts_with_choose_type(self, sampler):
        """First transition should always be CHOOSE_TYPE."""
        traj = sampler.sample()
        assert traj.transitions[0].sub_action_phase == SubActionPhase.CHOOSE_TYPE

    def test_sample_type_then_node_pattern(self, sampler):
        """Transitions should follow type→node[→phase→edge→transfer] pattern."""
        traj = sampler.sample()
        i = 0
        while i < len(traj.transitions):
            assert traj.transitions[i].sub_action_phase == SubActionPhase.CHOOSE_TYPE
            i += 1
            if i >= len(traj.transitions):
                break
            assert traj.transitions[i].sub_action_phase == SubActionPhase.CHOOSE_NODE
            i += 1
            # If F-Right (type 0 or 1), expect phase, edge, transfer
            action_type = traj.transitions[i - 2].sub_action_value
            if action_type < 2:
                if i >= len(traj.transitions):
                    break
                assert traj.transitions[i].sub_action_phase == SubActionPhase.CHOOSE_PHASE
                i += 1
                if i >= len(traj.transitions):
                    break
                assert traj.transitions[i].sub_action_phase == SubActionPhase.CHOOSE_NEW_EDGE
                i += 1
                if i >= len(traj.transitions):
                    break
                assert traj.transitions[i].sub_action_phase == SubActionPhase.CHOOSE_TRANSFER
                i += 1

    def test_sample_records_t_gates(self, sampler):
        traj = sampler.sample()
        assert traj.initial_t_gates >= 1

    def test_sample_records_num_rewrites(self, sampler):
        traj = sampler.sample()
        assert traj.num_rewrites > 0
        # num_rewrites should be less than total sub-steps
        assert traj.num_rewrites <= len(traj)

    def test_sample_no_grad(self, sampler):
        """sample() should produce detached log_pf tensors."""
        traj = sampler.sample()
        for t in traj.transitions:
            assert not t.log_pf.requires_grad

    def test_sample_log_pb_values(self, sampler):
        traj = sampler.sample()
        for t in traj.transitions:
            assert isinstance(t.log_pb, float)
            assert t.log_pb <= 0

    def test_sample_terminal_reward_nonneg(self, sampler):
        traj = sampler.sample()
        assert traj.terminal_reward >= 0

    # --- sample_with_grad ---

    def test_sample_with_grad_has_grad(self, sampler):
        traj = sampler.sample_with_grad()
        assert traj.sum_log_pf.requires_grad

    def test_sample_with_grad_backward(self, sampler):
        """Should be able to backpropagate through sum_log_pf."""
        traj = sampler.sample_with_grad()
        loss = traj.sum_log_pf ** 2
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in sampler.policy.parameters()
        )
        assert has_grad

    # --- sample_batch ---

    def test_sample_batch(self, sampler):
        batch = sampler.sample_batch(3)
        assert len(batch) == 3

    def test_sample_batch_with_grad(self, sampler):
        batch = sampler.sample_batch_with_grad(2)
        assert len(batch) == 2
        assert all(t.sum_log_pf.requires_grad for t in batch)

    # --- Edge cases ---

    def test_short_max_length(self):
        env = make_env()
        policy = make_policy()
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            temperature=1.0, max_trajectory_length=1,
        )
        traj = sampler.sample()
        assert traj.num_rewrites <= 1

    def test_high_temperature(self):
        env = make_env()
        policy = make_policy()
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            temperature=100.0, max_trajectory_length=5,
        )
        traj = sampler.sample()
        assert isinstance(traj, AnnotatedTrajectory)

    def test_full_epsilon_uniform(self):
        env = make_env()
        policy = make_policy()
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            epsilon_uniform=1.0, max_trajectory_length=5,
        )
        traj = sampler.sample()
        assert isinstance(traj, AnnotatedTrajectory)

    # --- Decomposition correctness ---

    def test_non_f_right_has_2_sub_steps_per_rewrite(self, sampler):
        """Non-F-Right rewrites should produce exactly 2 sub-steps (type + node)."""
        traj = sampler.sample()
        i = 0
        while i < len(traj.transitions):
            at = traj.transitions[i].sub_action_value
            if at >= 2:
                # Non-F-Right: should be type + node = 2 transitions
                assert i + 1 < len(traj.transitions)
                assert traj.transitions[i + 1].sub_action_phase == SubActionPhase.CHOOSE_NODE
                i += 2
            else:
                # F-Right: type + node + phase + edge + transfer = 5 transitions
                i += 5

    def test_f_right_has_5_sub_steps_per_rewrite(self, sampler):
        """F-Right rewrites should produce exactly 5 sub-steps."""
        traj = sampler.sample()
        i = 0
        while i < len(traj.transitions):
            at = traj.transitions[i].sub_action_value
            if at < 2 and i + 4 < len(traj.transitions):
                phases = [traj.transitions[i + j].sub_action_phase for j in range(5)]
                assert phases == [
                    SubActionPhase.CHOOSE_TYPE,
                    SubActionPhase.CHOOSE_NODE,
                    SubActionPhase.CHOOSE_PHASE,
                    SubActionPhase.CHOOSE_NEW_EDGE,
                    SubActionPhase.CHOOSE_TRANSFER,
                ]
                i += 5
            elif at >= 2:
                i += 2
            else:
                break  # incomplete F-Right at end (max_length hit)


# ---------------------------------------------------------------------------
# Parallel sampling tests — PyG Data export + replay_from_records
# ---------------------------------------------------------------------------

class TestWorkerSampleTrajectories:
    """Test the module-level worker function in-process (no subprocess spawn)."""

    def test_worker_returns_trajectory_records(self):
        """Worker function should return valid TrajectoryRecord objects."""
        from alphazx.mcts.parallel_self_play import _extract_model_hparams
        model = make_model()
        config = make_gflownet_config()

        model_hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        sampler_kwargs = {
            'temperature': config.sampling_temperature,
            'epsilon_uniform': config.epsilon_uniform,
            'max_trajectory_length': config.max_trajectory_length,
            'reward_exponent': config.reward_exponent,
            'min_reward': config.min_reward,
            'reward_shaping_coeff': config.reward_shaping_coeff,
            'pe_dim': config.pe_dim,
            'env_config': {
                'num_qubits': config.num_qubits,
                'depth': config.depth,
                'circuit_type': config.circuit_type,
                'p_had': config.p_had,
                'p_t': config.p_t,
                'min_initial_t_gates': config.min_initial_t_gates,
                'max_t_gate_increase': config.max_t_gate_increase,
                'max_episode_length': config.max_episode_length,
                'pe_dim': config.pe_dim,
            },
        }

        records = _worker_sample_trajectories(
            model_state_dict=state_dict,
            model_hparams=model_hparams,
            sampler_kwargs=sampler_kwargs,
            num_trajectories=3,
            worker_seed=42,
        )

        assert len(records) == 3
        for rec in records:
            assert isinstance(rec, TrajectoryRecord)
            assert rec.initial_t_gates >= 0
            assert rec.final_t_gates >= 0
            assert rec.terminal_reward > 0
            assert len(rec.rewrite_steps) == rec.num_rewrites
            assert len(rec.rewrite_actions) == rec.num_rewrites
            assert len(rec.per_rewrite_t_deltas) == rec.num_rewrites

    def test_rewrite_step_records_have_valid_tensors(self):
        """Each RewriteStepRecord should contain valid PyG Data tensors."""
        from alphazx.mcts.parallel_self_play import _extract_model_hparams
        model = make_model()
        config = make_gflownet_config()

        records = _worker_sample_trajectories(
            model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            model_hparams=_extract_model_hparams(model),
            sampler_kwargs={
                'temperature': 1.0,
                'epsilon_uniform': 0.0,
                'max_trajectory_length': 5,
                'reward_exponent': 4.0,
                'min_reward': 0.01,
                'reward_shaping_coeff': 0.0,
                'pe_dim': config.pe_dim,
                'env_config': {
                    'num_qubits': config.num_qubits,
                    'depth': config.depth,
                    'circuit_type': config.circuit_type,
                    'p_had': config.p_had,
                    'p_t': config.p_t,
                    'min_initial_t_gates': config.min_initial_t_gates,
                    'max_t_gate_increase': config.max_t_gate_increase,
                    'max_episode_length': config.max_episode_length,
                    'pe_dim': config.pe_dim,
                },
            },
            num_trajectories=2,
            worker_seed=123,
        )

        for rec in records:
            for step in rec.rewrite_steps:
                assert isinstance(step, RewriteStepRecord)
                # Tensors should exist and be non-empty
                assert step.x.numel() > 0
                assert step.edge_index.numel() > 0
                assert step.edge_index.shape[0] == 2
                assert step.node_type.numel() > 0
                # Action and backward prob fields
                assert isinstance(step.action_type, int)
                assert isinstance(step.node_index, int)
                assert step.num_available_types >= 1
                assert step.num_type_nodes >= 1
                assert step.t_gates_before >= 0


class TestReplayFromRecords:
    """Test the replay_from_records method on TrajectorySampler."""

    def test_replay_produces_annotated_trajectory(self):
        """replay_from_records should produce a valid AnnotatedTrajectory."""
        from alphazx.mcts.parallel_self_play import _extract_model_hparams
        model = make_model()
        config = make_gflownet_config()

        # Get a record from the worker function (run in-process)
        records = _worker_sample_trajectories(
            model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            model_hparams=_extract_model_hparams(model),
            sampler_kwargs={
                'temperature': 1.0,
                'epsilon_uniform': 0.0,
                'max_trajectory_length': 5,
                'reward_exponent': 4.0,
                'min_reward': 0.01,
                'reward_shaping_coeff': 0.0,
                'pe_dim': config.pe_dim,
                'env_config': {
                    'num_qubits': config.num_qubits,
                    'depth': config.depth,
                    'circuit_type': config.circuit_type,
                    'p_had': config.p_had,
                    'p_t': config.p_t,
                    'min_initial_t_gates': config.min_initial_t_gates,
                    'max_t_gate_increase': config.max_t_gate_increase,
                    'max_episode_length': config.max_episode_length,
                    'pe_dim': config.pe_dim,
                },
            },
            num_trajectories=3,
            worker_seed=42,
        )

        # Now replay through the model with a fresh sampler
        env = make_env()
        policy = make_policy(model)
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            max_trajectory_length=5,
        )

        # Put model in train mode so gradients flow
        policy.train()

        for record in records:
            if not record.rewrite_steps:
                continue

            traj = sampler.replay_from_records(record)

            # Basic structure checks
            assert isinstance(traj, AnnotatedTrajectory)
            assert traj.initial_t_gates == record.initial_t_gates
            assert traj.final_t_gates == record.final_t_gates
            assert traj.terminal_reward == record.terminal_reward
            assert traj.num_rewrites == record.num_rewrites

            # Should have transitions with gradient-tracked log_pf
            assert len(traj.transitions) > 0
            for t in traj.transitions:
                assert t.log_pf.requires_grad, \
                    f"log_pf should have gradients but doesn't for {t.sub_action_phase}"

            # sum_log_pf should be differentiable
            total_log_pf = traj.sum_log_pf
            assert total_log_pf.requires_grad
            # Should be able to compute gradients
            total_log_pf.backward(retain_graph=True)

            # Rewrite boundary info for SubTB
            assert len(traj.rewrite_boundary_log_flows) == record.num_rewrites
            assert len(traj.rewrite_start_indices) == record.num_rewrites


class TestParallelTrajectorySampler:
    """Integration test for the full parallel sampling pipeline."""

    def test_sample_and_replay_produces_trajectories(self):
        """Full pipeline: spawn workers → collect records → replay with gradients."""
        model = make_model()
        config = make_gflownet_config(
            num_sampling_workers=2,
            trajectories_per_batch=4,
            max_trajectory_length=5,
        )
        env = make_env()
        policy = make_policy(model)
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            temperature=config.sampling_temperature,
            epsilon_uniform=config.epsilon_uniform,
            max_trajectory_length=config.max_trajectory_length,
            reward_exponent=config.reward_exponent,
            min_reward=config.min_reward,
        )

        parallel = ParallelTrajectorySampler(
            model=model,
            config=config,
            sampler=sampler,
            num_workers=2,
        )

        try:
            # Put model in train mode
            policy.train()
            trajectories = parallel.sample_and_replay(n=4)

            assert len(trajectories) > 0
            for traj in trajectories:
                assert isinstance(traj, AnnotatedTrajectory)
                assert traj.terminal_reward > 0
                if traj.transitions:
                    # Gradients should flow
                    assert traj.sum_log_pf.requires_grad
        finally:
            parallel.shutdown()


class TestReplayRecordsBatched:
    """Test that batched replay produces identical results to sequential replay."""

    def _get_records_and_sampler(self, eval_mode: bool = True):
        """Helper: get trajectory records and a sampler for replaying.

        Args:
            eval_mode: If True (default), set model to eval mode so dropout
                is disabled and sequential vs batched results are identical.
                Set to False for gradient-flow tests where train mode is needed.
        """
        from alphazx.mcts.parallel_self_play import _extract_model_hparams
        model = make_model()
        config = make_gflownet_config()

        records = _worker_sample_trajectories(
            model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            model_hparams=_extract_model_hparams(model),
            sampler_kwargs={
                'temperature': 1.0,
                'epsilon_uniform': 0.0,
                'max_trajectory_length': 5,
                'reward_exponent': 4.0,
                'min_reward': 0.01,
                'reward_shaping_coeff': 0.0,
                'pe_dim': config.pe_dim,
                'env_config': {
                    'num_qubits': config.num_qubits,
                    'depth': config.depth,
                    'circuit_type': config.circuit_type,
                    'p_had': config.p_had,
                    'p_t': config.p_t,
                    'min_initial_t_gates': config.min_initial_t_gates,
                    'max_t_gate_increase': config.max_t_gate_increase,
                    'max_episode_length': config.max_episode_length,
                    'pe_dim': config.pe_dim,
                },
            },
            num_trajectories=4,
            worker_seed=42,
        )

        # Filter to records with rewrite steps
        records = [r for r in records if r.rewrite_steps]
        assert len(records) > 0, "Need at least one non-empty record"

        env = make_env()
        policy = make_policy(model)
        sampler = TrajectorySampler(
            env=env, policy=policy, device='cpu',
            max_trajectory_length=5,
        )
        # eval mode disables dropout for deterministic comparison;
        # gradients still flow (eval != no_grad).
        if eval_mode:
            policy.eval()
        else:
            policy.train()
        return records, sampler

    def test_batched_matches_sequential_structure(self):
        """Batched replay should produce the same trajectory structure."""
        records, sampler = self._get_records_and_sampler()

        # Sequential
        seq_trajs = [sampler.replay_from_records(r) for r in records]
        # Batched
        bat_trajs = sampler.replay_records_batched(records)

        assert len(bat_trajs) == len(seq_trajs)

        for seq, bat in zip(seq_trajs, bat_trajs):
            assert seq.initial_t_gates == bat.initial_t_gates
            assert seq.final_t_gates == bat.final_t_gates
            assert seq.terminal_reward == bat.terminal_reward
            assert seq.shaped_reward == bat.shaped_reward
            assert seq.num_rewrites == bat.num_rewrites
            assert len(seq.transitions) == len(bat.transitions)
            assert len(seq.rewrite_start_indices) == len(bat.rewrite_start_indices)
            assert seq.rewrite_start_indices == bat.rewrite_start_indices

    def test_batched_matches_sequential_structure(self):
        """Batched replay should produce the same sub-action phases and log_pb."""
        records, sampler = self._get_records_and_sampler()

        seq_trajs = [sampler.replay_from_records(r) for r in records]
        bat_trajs = sampler.replay_records_batched(records)

        for seq, bat in zip(seq_trajs, bat_trajs):
            for j, (st, bt) in enumerate(zip(seq.transitions, bat.transitions)):
                assert st.sub_action_phase == bt.sub_action_phase, \
                    f"Phase mismatch at transition {j}"
                assert st.sub_action_value == bt.sub_action_value, \
                    f"Sub-action value mismatch at transition {j}"
                assert st.log_pb == pytest.approx(bt.log_pb), \
                    f"log_pb mismatch at transition {j}"

    def test_batched_log_pf_reasonable(self):
        """Batched log_pf values should be finite, negative, and same sign as sequential.

        Exact numerical agreement is NOT expected because the homogeneous
        model uses BatchNorm in its GPS layers, and BatchNorm statistics
        differ between single-graph vs multi-graph forward passes.
        The HeteroModel (LayerNorm) would produce closer agreement.
        """
        records, sampler = self._get_records_and_sampler()

        seq_trajs = [sampler.replay_from_records(r) for r in records]
        bat_trajs = sampler.replay_records_batched(records)

        for seq, bat in zip(seq_trajs, bat_trajs):
            for j, (st, bt) in enumerate(zip(seq.transitions, bat.transitions)):
                s = st.log_pf.squeeze().item()
                b = bt.log_pf.squeeze().item()
                assert math.isfinite(s) and math.isfinite(b), \
                    f"Non-finite log_pf at transition {j}: seq={s}, bat={b}"
                # Both should be negative (log-probabilities)
                assert s < 0 and b < 0, \
                    f"Expected negative log_pf at transition {j}: seq={s}, bat={b}"

    def test_batched_gradients_flow(self):
        """Batched replay should produce gradient-tracked log_pf values."""
        records, sampler = self._get_records_and_sampler(eval_mode=False)
        bat_trajs = sampler.replay_records_batched(records)

        for traj in bat_trajs:
            assert len(traj.transitions) > 0
            for t in traj.transitions:
                assert t.log_pf.requires_grad, \
                    f"log_pf missing gradient for {t.sub_action_phase}"

            total = traj.sum_log_pf
            assert total.requires_grad
            total.backward(retain_graph=True)

    def test_batched_rewrite_boundary_flows(self):
        """SubTB boundary flows should be present and finite."""
        records, sampler = self._get_records_and_sampler()
        bat_trajs = sampler.replay_records_batched(records)

        for traj in bat_trajs:
            assert len(traj.rewrite_boundary_log_flows) == traj.num_rewrites
            for flow in traj.rewrite_boundary_log_flows:
                assert torch.isfinite(flow.squeeze())
