"""
Training loop for AlphaZero-style learning on ZX-calculus simplification.

The training loop alternates between:
1. Self-play: generate training data using MCTS guided by the current network.
2. Network updates: train the policy and value heads on replay buffer data.
3. Evaluation: measure T-gate reduction vs baseline (PyZX or previous best).
4. Checkpointing: save model weights periodically.

The loss function is:
    L = policy_loss + c_value * value_loss

Policy loss: cross-entropy between the MCTS visit-count distribution and the
model's predicted action probabilities. Since each example has a different set
of actions (from MCTS progressive widening), this is computed per-example.

Value loss: MSE between the model's predicted state value and the actual
episode outcome (normalized T-gate reduction).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.evaluate import evaluate_state, compute_action_prior
from alphazx.mcts.replay_buffer import ReplayBuffer, TrainingExample
from alphazx.mcts.self_play import SelfPlayManager, EpisodeResult
from alphazx.models.pre_process import pre_process_single

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the AlphaZero training loop."""

    # --- Self-play ---
    num_self_play_games: int = 100
    """Number of self-play games per training iteration."""

    # --- Training ---
    training_steps: int = 1000
    """Number of gradient steps per training iteration."""

    batch_size: int = 32
    """Number of examples per training minibatch."""

    learning_rate: float = 1e-3
    """Initial learning rate."""

    weight_decay: float = 1e-4
    """L2 regularization (weight decay) coefficient."""

    lr_schedule: str = 'cosine'
    """Learning rate schedule: 'cosine', 'constant', or 'step'."""

    c_value: float = 1.0
    """Weight of value loss relative to policy loss.
    L = policy_loss + c_value * value_loss"""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping. 0 to disable."""

    # --- Evaluation ---
    eval_interval: int = 1
    """Run evaluation every N training iterations."""

    eval_games: int = 20
    """Number of games to play for evaluation."""

    # --- Checkpointing ---
    checkpoint_interval: int = 1
    """Save a checkpoint every N training iterations."""

    checkpoint_dir: str = 'checkpoints'
    """Directory to save model checkpoints."""

    # --- Misc ---
    num_iterations: int = 100
    """Total number of train iterations (self-play + gradient steps)."""

    min_buffer_size: int = 256
    """Minimum replay buffer size before training starts.
    This prevents training on too few examples early on."""


@dataclass
class TrainStepMetrics:
    """Metrics from a single training step."""
    policy_loss: float
    value_loss: float
    total_loss: float


@dataclass
class IterationMetrics:
    """Metrics from a full training iteration (self-play + training)."""
    iteration: int
    # Self-play
    num_games: int
    avg_steps: float
    avg_t_gates_reduced: float
    simplification_rate: float
    self_play_time: float
    # Training
    avg_policy_loss: float
    avg_value_loss: float
    avg_total_loss: float
    training_time: float
    # Buffer
    buffer_size: int


class Trainer:
    """AlphaZero training loop for ZX-calculus simplification.

    Alternates between self-play data generation and network updates.
    """

    def __init__(
        self,
        model: nn.Module,
        mcts_config: MCTSConfig,
        trainer_config: TrainerConfig,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device('cpu'),
        evaluator=None,  # Optional: alphazx.mcts.evaluator.Evaluator
    ):
        self.model = model
        self.mcts_config = mcts_config
        self.trainer_config = trainer_config
        self.replay_buffer = replay_buffer
        self.device = device
        self.evaluator = evaluator

        # Self-play manager
        self.self_play_manager = SelfPlayManager(
            model=model,
            config=mcts_config,
            replay_buffer=replay_buffer,
            device=device,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        self.current_iteration = 0
        self.total_training_steps = 0

    def _build_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        cfg = self.trainer_config
        if cfg.lr_schedule == 'cosine':
            total_steps = cfg.num_iterations * cfg.training_steps
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-6
            )
        elif cfg.lr_schedule == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.training_steps * 10, gamma=0.5
            )
        else:
            return None

    def train(self) -> list[IterationMetrics]:
        """Run the full training loop for num_iterations."""
        all_metrics = []
        checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, self.trainer_config.num_iterations + 1):
            self.current_iteration = iteration
            logger.info(f"=== Iteration {iteration}/{self.trainer_config.num_iterations} ===")

            metrics = self._run_iteration()
            all_metrics.append(metrics)

            # Log iteration summary
            logger.info(
                f"Iter {iteration}: "
                f"policy_loss={metrics.avg_policy_loss:.4f}, "
                f"value_loss={metrics.avg_value_loss:.4f}, "
                f"total_loss={metrics.avg_total_loss:.4f}, "
                f"avg_t_reduced={metrics.avg_t_gates_reduced:.2f}, "
                f"simplification={metrics.simplification_rate:.1%}, "
                f"buffer={metrics.buffer_size}"
            )

            # Evaluation
            if self.evaluator and iteration % self.trainer_config.eval_interval == 0:
                eval_results = self.evaluator.evaluate(
                    self.model,
                    num_games=self.trainer_config.eval_games,
                )
                logger.info(f"Eval: {eval_results}")

            # Checkpoint
            if iteration % self.trainer_config.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration)

        return all_metrics

    def _run_iteration(self) -> IterationMetrics:
        """Run one iteration: self-play followed by training steps."""
        cfg = self.trainer_config

        # --- Self-play phase ---
        self.model.eval()
        sp_start = time.time()
        results = self.self_play_manager.generate_games(cfg.num_self_play_games)
        sp_time = time.time() - sp_start

        # Compute self-play statistics
        avg_steps = sum(r.num_steps for r in results) / max(1, len(results))
        avg_t_reduced = sum(r.t_gates_reduced for r in results) / max(1, len(results))
        simplification_rate = sum(1 for r in results if r.simplified) / max(1, len(results))

        # --- Training phase ---
        self.model.train()
        train_start = time.time()

        policy_losses = []
        value_losses = []
        total_losses = []

        # Only train if we have enough data
        if len(self.replay_buffer) >= cfg.min_buffer_size:
            for step in range(cfg.training_steps):
                metrics = self._train_step()
                policy_losses.append(metrics.policy_loss)
                value_losses.append(metrics.value_loss)
                total_losses.append(metrics.total_loss)
                self.total_training_steps += 1
        else:
            logger.info(
                f"Buffer size {len(self.replay_buffer)} < min {cfg.min_buffer_size}, "
                f"skipping training"
            )

        train_time = time.time() - train_start

        return IterationMetrics(
            iteration=self.current_iteration,
            num_games=len(results),
            avg_steps=avg_steps,
            avg_t_gates_reduced=avg_t_reduced,
            simplification_rate=simplification_rate,
            self_play_time=sp_time,
            avg_policy_loss=_safe_mean(policy_losses),
            avg_value_loss=_safe_mean(value_losses),
            avg_total_loss=_safe_mean(total_losses),
            training_time=train_time,
            buffer_size=len(self.replay_buffer),
        )

    def _train_step(self) -> TrainStepMetrics:
        """Execute a single training step on a minibatch from the replay buffer.

        Policy loss: For each example, we compute the KL divergence (equivalent to
        cross-entropy up to a constant) between the MCTS policy and the model's
        predicted action probabilities.

        Specifically, for MCTS policy π and model probability p:
            policy_loss = -Σ_a π(a) * log p(a)

        where the sum is over all actions in the MCTS policy (the actions that
        were sampled during search). We compute p(a) using the model's distribution
        component methods, same as compute_action_prior().

        Value loss: MSE between predicted value and episode outcome.
        """
        cfg = self.trainer_config
        examples = self.replay_buffer.sample(cfg.batch_size)

        # Process examples one at a time because each has a different graph structure
        # and different action sets. Batching the forward pass would require the
        # evaluate_states_batch infrastructure, which adds complexity for uncertain
        # gain at this stage. Sequential per-example processing is clearer and correct.
        #
        # IMPORTANT: we accumulate losses into a Python list and sum at the end
        # rather than doing in-place +=/-= on leaf tensors, because in-place ops
        # on leaf tensors detach from the computation graph and break gradient flow.
        policy_loss_terms: list[torch.Tensor] = []
        value_loss_terms: list[torch.Tensor] = []

        for example in examples:
            data = example.state_data.clone().to(self.device)
            batch_tensor = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

            # Forward pass
            dist_params, pred_value = self.model(
                data.x, data.edge_index, data.edge_attr,
                data.node_type,
                batch_tensor,
                data.pe,
                data.id.to(self.device),
            )
            distribution = AlphaZXDistribution(dist_params)

            # --- Policy loss ---
            # Cross-entropy: -Σ_a π(a) * log p(a)
            # where π(a) is the MCTS visit-count probability
            # and p(a) is the model's predicted probability for action a
            action_log_probs: list[torch.Tensor] = []
            for action, mcts_prob in example.mcts_policy.items():
                if mcts_prob < 1e-8:
                    continue  # Skip near-zero entries
                log_prob = self._compute_log_prob(distribution, action)
                action_log_probs.append(-mcts_prob * log_prob)

            if action_log_probs:
                example_policy_loss = torch.stack(action_log_probs).sum()
                policy_loss_terms.append(example_policy_loss)

            # --- Value loss ---
            value_target = torch.tensor(
                example.value_target, dtype=torch.float32, device=self.device
            )
            example_value_loss = (pred_value.squeeze() - value_target) ** 2
            value_loss_terms.append(example_value_loss)

        if not value_loss_terms:
            return TrainStepMetrics(0.0, 0.0, 0.0)

        # Average over the batch — torch.stack keeps the computation graph intact
        num_valid = len(value_loss_terms)
        avg_policy_loss = (
            torch.stack(policy_loss_terms).sum() / num_valid
            if policy_loss_terms
            else torch.tensor(0.0, device=self.device)
        )
        avg_value_loss = torch.stack(value_loss_terms).sum() / num_valid
        total_loss = avg_policy_loss + cfg.c_value * avg_value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return TrainStepMetrics(
            policy_loss=avg_policy_loss.item(),
            value_loss=avg_value_loss.item(),
            total_loss=total_loss.item(),
        )

    def _compute_log_prob(
        self,
        distribution: AlphaZXDistribution,
        action: tuple,
    ) -> torch.Tensor:
        """Compute log p(action) from the model's distribution, with gradients.

        This mirrors compute_action_prior() from evaluate.py but keeps tensors
        on the computation graph so gradients flow through the model.
        """
        action_type = torch.tensor([action[1]], dtype=torch.long, device=self.device)
        node = torch.tensor([action[2]], dtype=torch.long, device=self.device)
        phase = torch.tensor([action[3]], dtype=torch.long, device=self.device)
        new_edges = torch.tensor([action[4]], dtype=torch.long, device=self.device)
        transfer_edges = torch.tensor(
            [list(action[5:])], dtype=torch.float32, device=self.device
        )

        # Sum log probabilities of each component
        log_prob = (
            distribution.action_type_log_probs(action_type)
            + distribution.node_log_probs(action_type, node)
            + distribution.new_phase_log_probs(node, phase)
            + distribution.new_edge_log_probs(node, new_edges)
            + distribution.transfer_edge_log_probs(node, transfer_edges)
        )

        return log_prob

    def _save_checkpoint(self, checkpoint_dir: Path, iteration: int) -> None:
        """Save model and optimizer state."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_training_steps': self.total_training_steps,
            'mcts_config': self.mcts_config,
            'trainer_config': self.trainer_config,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = checkpoint_dir / f'checkpoint_iter_{iteration:04d}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Also save a 'latest' symlink/copy for convenience
        latest_path = checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

    @classmethod
    def load_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        mcts_config: MCTSConfig,
        trainer_config: TrainerConfig,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device('cpu'),
    ) -> 'Trainer':
        """Load a trainer from a checkpoint.

        :param model: An uninitialized model of the correct architecture.
        :param checkpoint_path: Path to the checkpoint file.
        :return: A Trainer with restored state.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        trainer = cls(model, mcts_config, trainer_config, replay_buffer, device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_iteration = checkpoint.get('iteration', 0)
        trainer.total_training_steps = checkpoint.get('total_training_steps', 0)

        if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return trainer


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a list, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
