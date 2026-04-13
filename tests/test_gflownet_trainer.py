"""Tests for GFlowNet trainer (TB loss, training step, evaluation)."""

import math

import pytest
import torch

from alphazx.gflownet.trainer import (
    GFlowNetTrainer, GFlowNetTrainMetrics, GFlowNetEvalMetrics,
)
from alphazx.gflownet.sampler import (
    AnnotatedTrajectory, AnnotatedTransition, SubActionPhase,
)
from tests.gflownet_utils import make_model, make_gflownet_config


# ---------------------------------------------------------------------------
# Metrics dataclass tests
# ---------------------------------------------------------------------------

class TestGFlowNetTrainMetrics:
    def test_default_values(self):
        m = GFlowNetTrainMetrics()
        assert m.tb_loss == 0.0
        assert m.log_Z == 0.0
        assert m.num_trajectories == 0
        assert m.mean_sub_steps == 0.0
        assert m.mean_rewrites == 0.0

    def test_custom_values(self):
        m = GFlowNetTrainMetrics(
            tb_loss=42.5, log_Z=1.3, mean_t_gate_reduction=2.5,
            num_trajectories=16, mean_sub_steps=30.0, mean_rewrites=10.0,
        )
        assert m.tb_loss == 42.5
        assert m.mean_rewrites == 10.0


class TestGFlowNetEvalMetrics:
    def test_default_values(self):
        m = GFlowNetEvalMetrics()
        assert m.mean_t_gate_reduction == 0.0
        assert m.num_games == 0


# ---------------------------------------------------------------------------
# GFlowNetTrainer tests
# ---------------------------------------------------------------------------

class TestGFlowNetTrainer:
    @pytest.fixture
    def trainer(self):
        model = make_model()
        config = make_gflownet_config(
            trajectories_per_batch=2,
            max_trajectory_length=10,
        )
        return GFlowNetTrainer(model, config, device='cpu')

    # --- Construction ---

    def test_has_policy(self, trainer):
        assert trainer.policy is not None

    def test_has_optimizer(self, trainer):
        assert trainer.optimizer is not None

    def test_has_scheduler(self, trainer):
        assert trainer.scheduler is not None

    def test_optimizer_has_two_param_groups(self, trainer):
        """log_Z should have its own param group with a higher LR."""
        groups = trainer.optimizer.param_groups
        assert len(groups) == 2
        # First group: model params
        assert groups[0]['lr'] == trainer.config.learning_rate
        # Second group: log_Z with higher LR
        assert groups[1]['lr'] == trainer.config.log_z_learning_rate
        assert groups[1]['lr'] >= groups[0]['lr']
        # log_Z should have no weight decay
        assert groups[1]['weight_decay'] == 0.0

    def test_scheduler_is_cosine_warm_restarts(self, trainer):
        """Default schedule should be CosineAnnealingWarmRestarts."""
        import torch.optim.lr_scheduler as sched
        assert isinstance(trainer.scheduler, sched.CosineAnnealingWarmRestarts)

    # --- TB loss ---

    def test_tb_loss_basic(self, trainer):
        traj = AnnotatedTrajectory(
            transitions=[
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                    sub_action_value=4,
                    log_pf=torch.tensor(-1.0, requires_grad=True),
                    log_pb=-0.5,
                ),
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NODE,
                    sub_action_value=7,
                    log_pf=torch.tensor(-0.5, requires_grad=True),
                    log_pb=-0.3,
                ),
            ],
            initial_t_gates=3, final_t_gates=1,
            terminal_reward=0.5, num_rewrites=1,
        )
        loss = trainer._trajectory_balance_loss(traj)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_tb_loss_formula(self, trainer):
        """Verify the TB loss formula: (log_Z + sum_log_pf - log_R - sum_log_pb)^2."""
        log_pf_1 = torch.tensor(-1.0, requires_grad=True)
        log_pf_2 = torch.tensor(-0.5, requires_grad=True)
        traj = AnnotatedTrajectory(
            transitions=[
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                    sub_action_value=4,
                    log_pf=log_pf_1, log_pb=-0.8,
                ),
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NODE,
                    sub_action_value=7,
                    log_pf=log_pf_2, log_pb=-0.3,
                ),
            ],
            initial_t_gates=4, final_t_gates=2,
            terminal_reward=0.25, num_rewrites=1,
        )
        loss = trainer._trajectory_balance_loss(traj)

        log_Z = trainer.policy.log_Z.item()
        sum_log_pf = -1.0 + -0.5
        log_R = math.log(0.25)
        sum_log_pb = -0.8 + -0.3
        expected = (log_Z + sum_log_pf - log_R - sum_log_pb) ** 2

        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_tb_loss_zero_reward_returns_none(self, trainer):
        traj = AnnotatedTrajectory(
            transitions=[
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                    sub_action_value=4,
                    log_pf=torch.tensor(-1.0, requires_grad=True),
                    log_pb=-0.5,
                ),
            ],
            terminal_reward=0.0, num_rewrites=1,
        )
        assert trainer._trajectory_balance_loss(traj) is None

    def test_tb_loss_empty_trajectory_returns_none(self, trainer):
        traj = AnnotatedTrajectory(terminal_reward=0.5)
        assert trainer._trajectory_balance_loss(traj) is None

    def test_tb_loss_gradient_flows_to_log_Z(self, trainer):
        traj = AnnotatedTrajectory(
            transitions=[
                AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                    sub_action_value=4,
                    log_pf=torch.tensor(-1.5, requires_grad=True),
                    log_pb=-0.5,
                ),
            ],
            initial_t_gates=3, final_t_gates=1,
            terminal_reward=0.5, num_rewrites=1,
        )
        loss = trainer._trajectory_balance_loss(traj)
        loss.backward()
        assert trainer.policy.log_Z.grad is not None
        assert trainer.policy.log_Z.grad.abs().item() > 0

    # --- train_step ---

    def test_train_step_returns_metrics(self, trainer):
        metrics = trainer.train_step()
        assert isinstance(metrics, GFlowNetTrainMetrics)

    def test_train_step_produces_loss(self, trainer):
        metrics = trainer.train_step()
        assert metrics.tb_loss >= 0

    def test_train_step_records_trajectory_count(self, trainer):
        metrics = trainer.train_step()
        assert metrics.num_trajectories == 2

    def test_train_step_records_rewrites_and_substeps(self, trainer):
        metrics = trainer.train_step()
        assert metrics.mean_rewrites > 0
        assert metrics.mean_sub_steps > 0
        # sub_steps should be >= rewrites (at least 2 sub-steps per rewrite)
        assert metrics.mean_sub_steps >= metrics.mean_rewrites

    def test_train_step_updates_step_count(self, trainer):
        assert trainer._step_count == 0
        trainer.train_step()
        assert trainer._step_count == 1

    def test_multiple_train_steps(self, trainer):
        for i in range(3):
            metrics = trainer.train_step()
            assert isinstance(metrics, GFlowNetTrainMetrics)
        assert trainer._step_count == 3

    # --- evaluate ---

    def test_evaluate_returns_metrics(self, trainer):
        metrics = trainer.evaluate(num_games=2)
        assert isinstance(metrics, GFlowNetEvalMetrics)

    def test_evaluate_records_game_count(self, trainer):
        metrics = trainer.evaluate(num_games=3)
        assert metrics.num_games == 3

    def test_evaluate_frac_positive_bounded(self, trainer):
        metrics = trainer.evaluate(num_games=2)
        assert 0.0 <= metrics.frac_positive_reduction <= 1.0

    def test_evaluate_no_grad(self, trainer):
        trainer.policy.zero_grad()
        trainer.evaluate(num_games=2)
        for p in trainer.policy.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    # --- Integration ---

    def test_train_then_evaluate(self, trainer):
        for _ in range(2):
            trainer.train_step()
        eval_metrics = trainer.evaluate(num_games=2)
        assert isinstance(eval_metrics, GFlowNetEvalMetrics)
