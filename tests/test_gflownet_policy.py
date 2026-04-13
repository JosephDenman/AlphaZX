"""Tests for GFlowNet forward and backward policies."""

import math

import pytest
import torch

from alphazx.gflownet.policy import GFlowNetForwardPolicy, UniformBackwardPolicy
from tests.gflownet_utils import make_model, make_env, make_policy


class TestGFlowNetForwardPolicy:
    @pytest.fixture
    def policy(self):
        return make_policy()

    @pytest.fixture
    def env(self):
        return make_env()

    @pytest.fixture
    def state(self, env):
        return env.generate_state()

    # --- Construction ---

    def test_has_log_Z_parameter(self, policy):
        assert hasattr(policy, 'log_Z')
        assert isinstance(policy.log_Z, torch.nn.Parameter)
        assert policy.log_Z.item() == pytest.approx(0.0)

    def test_log_Z_in_parameters(self, policy):
        """log_Z should be included in policy.parameters() for optimization."""
        param_names = {name for name, _ in policy.named_parameters()}
        assert 'log_Z' in param_names

    def test_model_params_in_parameters(self, policy):
        """Model parameters should be included in policy.parameters()."""
        param_names = {name for name, _ in policy.named_parameters()}
        assert any(name.startswith('model.') for name in param_names)

    # --- Forward pass ---

    def test_forward_returns_distribution_and_value(self, policy, state):
        from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution
        dist, value = policy(state, 'cpu')
        assert isinstance(dist, AlphaZXDistribution)
        assert isinstance(value, torch.Tensor)

    def test_forward_with_grad(self, policy, state):
        """Forward pass should preserve gradients (no @torch.no_grad)."""
        dist, value = policy(state, 'cpu')
        assert value.requires_grad

    def test_forward_no_grad_when_wrapped(self, policy, state):
        """Caller can still disable gradients with torch.no_grad()."""
        with torch.no_grad():
            dist, value = policy(state, 'cpu')
            assert not value.requires_grad

    # --- Distribution provides per-component log-probs ---

    def test_distribution_has_action_type_log_probs(self, policy, state):
        dist, _ = policy(state, 'cpu')
        log_p = dist.action_type_log_probs(torch.tensor([2]))
        assert isinstance(log_p, torch.Tensor)
        assert log_p.item() < 0

    def test_distribution_has_node_log_probs(self, policy, state):
        dist, _ = policy(state, 'cpu')
        log_p = dist.node_log_probs(torch.tensor([[2]]), torch.tensor([[3]]))
        assert isinstance(log_p, torch.Tensor)

    def test_distribution_has_phase_log_probs(self, policy, state):
        dist, _ = policy(state, 'cpu')
        log_p = dist.new_phase_log_probs(
            torch.tensor([[0]]), torch.tensor([[3]]), torch.tensor([[0]]),
        )
        assert isinstance(log_p, torch.Tensor)


class TestUniformBackwardPolicy:
    def test_log_prob_single_action(self):
        lp = UniformBackwardPolicy.log_prob(1)
        assert lp == pytest.approx(0.0)  # -log(1) = 0

    def test_log_prob_multiple_actions(self):
        lp = UniformBackwardPolicy.log_prob(10)
        assert lp == pytest.approx(-math.log(10))

    def test_log_prob_zero_actions(self):
        lp = UniformBackwardPolicy.log_prob(0)
        assert lp == 0.0

    def test_log_prob_negative_actions(self):
        lp = UniformBackwardPolicy.log_prob(-5)
        assert lp == 0.0

    def test_log_prob_is_negative_for_k_gt_1(self):
        for k in range(2, 20):
            assert UniformBackwardPolicy.log_prob(k) < 0

    def test_log_prob_decreases_with_more_actions(self):
        lp5 = UniformBackwardPolicy.log_prob(5)
        lp10 = UniformBackwardPolicy.log_prob(10)
        lp100 = UniformBackwardPolicy.log_prob(100)
        assert lp5 > lp10 > lp100
