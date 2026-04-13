"""Tests for GFlowNet configuration."""

import pytest
from dataclasses import fields

from alphazx.shared.config import CircuitConfig
from alphazx.gflownet.config import GFlowNetConfig


class TestGFlowNetConfig:
    """Tests for GFlowNetConfig dataclass."""

    def test_inherits_circuit_config(self):
        """GFlowNetConfig should inherit from CircuitConfig."""
        assert issubclass(GFlowNetConfig, CircuitConfig)

    def test_default_values(self):
        cfg = GFlowNetConfig()
        # Inherited defaults
        assert cfg.num_qubits == 5
        assert cfg.depth == 5
        assert cfg.pe_dim == 20
        # GFlowNet-specific defaults
        assert cfg.loss_type == 'trajectory_balance'
        assert cfg.reward_exponent == 4.0
        assert cfg.reward_exponent_initial == 1.0
        assert cfg.reward_exponent_warmup_iters == 200
        assert cfg.min_reward == 0.01
        assert cfg.learn_log_Z is True
        assert cfg.trajectories_per_batch == 64
        assert cfg.epsilon_uniform == 0.0
        assert cfg.max_trajectory_length == 20
        assert cfg.learn_backward_policy is True
        assert cfg.sampling_temperature == 1.0
        assert cfg.eval_temperature == 1.0
        assert cfg.learning_rate == 1e-3
        assert cfg.log_z_learning_rate == 5e-2
        assert cfg.weight_decay == 1e-4
        assert cfg.lr_schedule == 'cosine_restarts'
        assert cfg.lr_restart_period == 50
        assert cfg.reward_shaping_coeff == 0.0
        assert cfg.replay_buffer_size == 1000
        assert cfg.replay_ratio == 0.0
        assert cfg.replay_min_reward == 0.02
        assert cfg.replay_diversity_weight == 0.1
        assert cfg.subtb_lambda == 0.9
        assert cfg.grad_clip_max_norm == 1.0

    def test_override_inherited_params(self):
        cfg = GFlowNetConfig(num_qubits=10, depth=8)
        assert cfg.num_qubits == 10
        assert cfg.depth == 8

    def test_override_gflownet_params(self):
        cfg = GFlowNetConfig(
            loss_type='detailed_balance',
            reward_exponent=2.0,
            trajectories_per_batch=32,
            sampling_temperature=0.5,
        )
        assert cfg.loss_type == 'detailed_balance'
        assert cfg.reward_exponent == 2.0
        assert cfg.trajectories_per_batch == 32
        assert cfg.sampling_temperature == 0.5

    def test_effective_max_episode_length_inherited(self):
        """Should inherit effective_max_episode_length from CircuitConfig."""
        cfg = GFlowNetConfig(max_episode_length=100, num_qubits=3, depth=3)
        assert cfg.effective_max_episode_length > 0

    def test_has_all_gflownet_fields(self):
        """All expected GFlowNet fields should be present."""
        field_names = {f.name for f in fields(GFlowNetConfig)}
        expected_gfn_fields = {
            'loss_type', 'reward_exponent', 'min_reward', 'learn_log_Z',
            'reward_exponent_initial', 'reward_exponent_warmup_iters',
            'trajectories_per_batch', 'epsilon_uniform', 'max_trajectory_length',
            'learn_backward_policy', 'sampling_temperature', 'eval_temperature',
            'learning_rate', 'log_z_learning_rate', 'weight_decay',
            'lr_schedule', 'lr_restart_period',
            'reward_shaping_coeff',
            'subtb_lambda', 'grad_clip_max_norm',
            'replay_buffer_size', 'replay_ratio',
            'replay_min_reward', 'replay_diversity_weight',
        }
        assert expected_gfn_fields.issubset(field_names)
