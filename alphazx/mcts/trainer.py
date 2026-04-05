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

import math

import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter

from torch_geometric.data import Batch

from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution, AlphaZXDistributionParams
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.curriculum import CurriculumScheduler, CurriculumConfig
from alphazx.mcts.evaluate import evaluate_state, compute_action_prior
from alphazx.mcts.replay_buffer import ReplayBuffer, TrainingExample
from alphazx.mcts.self_play import SelfPlayManager, EpisodeResult, ACTION_TYPE_NAMES
from alphazx.mcts.parallel_self_play import ParallelSelfPlayManager
from alphazx.mcts.tb_logger import TBLogger, SelfPlayStats, TrainStepDiagnostics
from alphazx.models.pre_process import pre_process_single

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the AlphaZero training loop."""

    # --- Self-play ---
    num_self_play_games: int = 100
    """Number of self-play games per training iteration."""

    # --- Training ---
    training_steps: int = 100
    """Number of gradient steps per training iteration.
    Reduced from 1000 to prevent overfitting on small replay buffers.
    With 50 games/iter and ~20 examples each, 100 steps × 32 batch = ~3200
    sample draws ≈ 3 passes through new data, which is reasonable."""

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

    # --- Parallelism ---
    num_self_play_workers: int = 1
    """Number of parallel worker processes for self-play game generation.
    1 = serial (no multiprocessing overhead, uses SelfPlayManager).
    >1 = spawns worker processes via ParallelSelfPlayManager.
    Recommended: os.cpu_count() - 1 to leave one core for the main process."""

    # --- Misc ---
    num_iterations: int = 100
    """Total number of train iterations (self-play + gradient steps)."""

    min_buffer_size: int = 256
    """Minimum replay buffer size before training starts.
    This prevents training on too few examples early on."""

    # --- TensorBoard ---
    tensorboard: bool = True
    """Enable TensorBoard logging."""

    tensorboard_dir: str = 'runs/alphazx'
    """Directory for TensorBoard event files."""


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
        tb_logger: Optional[TBLogger] = None,
        curriculum: Optional[CurriculumScheduler] = None,
    ):
        self.model = model
        self.mcts_config = mcts_config
        self.trainer_config = trainer_config
        self.replay_buffer = replay_buffer
        self.device = device
        self.evaluator = evaluator
        self.tb_logger = tb_logger
        self.curriculum = curriculum

        # Self-play manager: parallel if num_self_play_workers > 1
        if trainer_config.num_self_play_workers > 1:
            self.self_play_manager = ParallelSelfPlayManager(
                model=model,
                config=mcts_config,
                replay_buffer=replay_buffer,
                device=device,
                num_workers=trainer_config.num_self_play_workers,
            )
        else:
            self.self_play_manager = SelfPlayManager(
                model=model,
                config=mcts_config,
                replay_buffer=replay_buffer,
                device=device,
            )

        # If curriculum is enabled, apply the initial difficulty level
        if self.curriculum and self.curriculum.config.enabled:
            mcts_config.num_qubits = self.curriculum.current_num_qubits
            mcts_config.depth = self.curriculum.current_depth

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
                if self.tb_logger:
                    self.tb_logger.log_evaluation(iteration, eval_results)

            # Model histograms (once per iteration, not per step)
            if self.tb_logger:
                self.tb_logger.log_model_histograms(iteration, self.model)
                self.tb_logger.flush()

            # Checkpoint
            if iteration % self.trainer_config.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration)

        # Close TensorBoard writer at the end of training
        if self.tb_logger:
            self.tb_logger.close()

        return all_metrics

    def _run_iteration(self) -> IterationMetrics:
        """Run one iteration: self-play followed by training steps."""
        cfg = self.trainer_config

        # --- Self-play phase ---
        # If curriculum is active with mixed-difficulty sampling, generate
        # games at different difficulty levels within the same iteration.
        self.model.eval()
        sp_start = time.time()

        if self.curriculum and self.curriculum.config.enabled:
            results = self._generate_curriculum_games(cfg.num_self_play_games)
        else:
            results = self.self_play_manager.generate_games(cfg.num_self_play_games)

        sp_time = time.time() - sp_start

        # Compute self-play statistics
        n_games = max(1, len(results))
        avg_steps = sum(r.num_steps for r in results) / n_games
        avg_t_reduced = sum(r.t_gates_reduced for r in results) / n_games
        simplification_rate = sum(1 for r in results if r.simplified) / n_games

        # --- Curriculum update ---
        if self.curriculum:
            avg_initial_t = sum(r.initial_t_gates for r in results) / n_games
            advanced = self.curriculum.update(
                self.mcts_config,
                self.current_iteration,
                avg_t_reduced,
                avg_initial_t,
                simplification_rate,
            )
            if self.tb_logger:
                self._log_curriculum(advanced)

        # Collect extended self-play stats for TensorBoard
        if self.tb_logger:
            sp_stats = self._collect_self_play_stats(results, sp_time)
            self.tb_logger.log_self_play(self.current_iteration, sp_stats)

        # --- Training phase ---
        self.model.train()
        train_start = time.time()

        policy_losses = []
        value_losses = []
        total_losses = []

        # Only train if we have enough data
        if len(self.replay_buffer) >= cfg.min_buffer_size:
            for step in range(cfg.training_steps):
                diag = self._train_step()
                policy_losses.append(diag.policy_loss)
                value_losses.append(diag.value_loss)
                total_losses.append(diag.total_loss)
                self.total_training_steps += 1

                # Log per-step metrics to TensorBoard
                if self.tb_logger:
                    self.tb_logger.log_train_step(self.total_training_steps, diag)
        else:
            logger.info(
                f"Buffer size {len(self.replay_buffer)} < min {cfg.min_buffer_size}, "
                f"skipping training"
            )

        train_time = time.time() - train_start

        # Log iteration-level training aggregates
        if self.tb_logger:
            self.tb_logger.log_iteration_training(
                iteration=self.current_iteration,
                avg_policy_loss=_safe_mean(policy_losses),
                avg_value_loss=_safe_mean(value_losses),
                avg_total_loss=_safe_mean(total_losses),
                training_time=train_time,
                buffer_size=len(self.replay_buffer),
                buffer_total_added=self.replay_buffer.total_added,
            )

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

    def _train_step(self) -> TrainStepDiagnostics:
        """Execute a single training step on a minibatch from the replay buffer.

        Uses a BATCHED forward pass through the GNN for efficiency: all examples
        in the minibatch are packed into a single PyG Batch and processed in one
        forward pass.  The distribution parameters are then sliced per-example
        for policy loss computation (each example has a different MCTS action set).

        Policy loss: cross-entropy between MCTS visit-count distribution π and
        the model's predicted action probabilities p:
            policy_loss = -Σ_a π(a) * log p(a)

        Value loss: MSE between predicted value and episode outcome.

        Returns TrainStepDiagnostics with extended metrics for TensorBoard.
        """
        cfg = self.trainer_config
        examples = self.replay_buffer.sample(cfg.batch_size)

        # --- Batched forward pass ---
        # Pack all examples into a single PyG Batch.  Batch.from_data_list
        # concatenates node/edge features and offsets edge indices correctly
        # for variable-size graphs.
        data_list = [ex.state_data.clone().to(self.device) for ex in examples]
        batch = Batch.from_data_list(data_list).to(self.device)
        graph_ids = torch.stack([d.id for d in data_list]).to(self.device)

        dist_params, pred_values = self.model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.node_type,
            batch.batch,
            batch.pe,
            graph_ids,
        )

        # --- Value loss (fully batched) ---
        value_targets = torch.tensor(
            [ex.value_target for ex in examples], dtype=torch.float32, device=self.device
        )
        pred_values_flat = pred_values.squeeze(-1)
        value_loss = ((pred_values_flat - value_targets) ** 2).mean()

        # --- Policy loss (per-example, on sliced distribution params) ---
        # Each example has a different MCTS action set, so we split the batched
        # distribution params and compute cross-entropy individually.  The tensor
        # slicing preserves the computation graph for gradient flow.
        policy_loss_terms: list[torch.Tensor] = []
        policy_entropies: list[float] = []

        for i, example in enumerate(examples):
            single_params = AlphaZXDistributionParams(
                graph_ids=dist_params.graph_ids[i:i+1],
                mixture_dist_probs=dist_params.mixture_dist_probs[i:i+1],
                node_dist_probs=dist_params.node_dist_probs[i:i+1],
                phase_dist_probs=dist_params.phase_dist_probs[i:i+1],
                new_edge_dist_probs=dist_params.new_edge_dist_probs[i:i+1],
                transfer_edge_dist_probs=dist_params.transfer_edge_dist_probs[i:i+1],
            )
            distribution = AlphaZXDistribution(single_params)

            # MCTS policy entropy: H(π) = -Σ π(a) log π(a)
            entropy = 0.0
            for prob in example.mcts_policy.values():
                if prob > 1e-8:
                    entropy -= prob * math.log(prob)
            policy_entropies.append(entropy)

            # Cross-entropy: -Σ_a π(a) * log p(a)
            action_log_probs: list[torch.Tensor] = []
            for action, mcts_prob in example.mcts_policy.items():
                if mcts_prob < 1e-8:
                    continue
                log_prob = self._compute_log_prob(distribution, action)
                action_log_probs.append(-mcts_prob * log_prob)

            if action_log_probs:
                policy_loss_terms.append(torch.stack(action_log_probs).sum())

        if not policy_loss_terms:
            return TrainStepDiagnostics()

        num_examples = len(examples)
        avg_policy_loss = torch.stack(policy_loss_terms).sum() / num_examples
        total_loss = avg_policy_loss + cfg.c_value * value_loss

        # --- Backward pass ---
        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute gradient norm before clipping (for diagnostics)
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

        self.optimizer.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            self.scheduler.step()

        # --- Diagnostics ---
        value_preds_list = pred_values_flat.detach().cpu().flatten().tolist()
        value_targets_list = value_targets.detach().cpu().flatten().tolist()
        vt_mean = _safe_mean(value_targets_list)
        vt_std = (
            (sum((v - vt_mean) ** 2 for v in value_targets_list) / max(1, len(value_targets_list))) ** 0.5
            if value_targets_list else 0.0
        )

        return TrainStepDiagnostics(
            policy_loss=avg_policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=total_loss.item(),
            grad_norm=grad_norm,
            learning_rate=current_lr,
            value_target_mean=vt_mean,
            value_target_std=vt_std,
            value_prediction_mean=_safe_mean(value_preds_list),
            mcts_policy_entropy=_safe_mean(policy_entropies),
            num_valid_examples=num_examples,
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

        # Sum log probabilities of each component.
        # Phase, edge, and transfer selectors are conditioned on action_type.
        log_prob = (
            distribution.action_type_log_probs(action_type)
            + distribution.node_log_probs(action_type, node)
            + distribution.new_phase_log_probs(action_type, node, phase)
            + distribution.new_edge_log_probs(action_type, node, new_edges)
            + distribution.transfer_edge_log_probs(action_type, node, transfer_edges)
        )

        return log_prob

    def _collect_self_play_stats(
        self,
        results: list[EpisodeResult],
        sp_time: float,
    ) -> SelfPlayStats:
        """Collect extended statistics from self-play results for TensorBoard.

        Extracts per-node-type counts from initial and final game states,
        action type distributions, and other detailed metrics that go beyond
        the basic IterationMetrics.
        """
        from alphazx.diagram.match import METADATA

        n = max(1, len(results))
        total_examples = sum(len(r.examples) for r in results)

        # Aggregate action type counts across all games
        action_type_counts = Counter()
        for r in results:
            for ex in r.examples:
                for action in ex.mcts_policy.keys():
                    if len(action) > 1:
                        action_type_idx = action[1]
                        action_name = ACTION_TYPE_NAMES.get(
                            action_type_idx, f"unknown({action_type_idx})"
                        )
                        action_type_counts[action_name] += 1

        # Per-node-type counts from the match diagram
        # We can extract these from the EpisodeResult's first and last examples
        # by counting node_type values in the PyG data
        initial_counts: dict[str, float] = {}
        final_counts: dict[str, float] = {}

        abbrevs = METADATA.node_type_abbrevs
        abbrev_index = METADATA.node_type_abbrev_index_dict

        for r in results:
            if r.examples:
                # Initial state: first example's state_data
                first_data = r.examples[0].state_data
                if hasattr(first_data, 'node_type') and first_data.node_type is not None:
                    nt = first_data.node_type
                    for abbrev in abbrevs:
                        idx = abbrev_index[abbrev]
                        count = (nt == idx).sum().item()
                        initial_counts[abbrev] = initial_counts.get(abbrev, 0) + count

                # Final state: last example's state_data
                last_data = r.examples[-1].state_data
                if hasattr(last_data, 'node_type') and last_data.node_type is not None:
                    nt = last_data.node_type
                    for abbrev in abbrevs:
                        idx = abbrev_index[abbrev]
                        count = (nt == idx).sum().item()
                        final_counts[abbrev] = final_counts.get(abbrev, 0) + count

        # Average over games
        initial_avg = {k: v / n for k, v in initial_counts.items()}
        final_avg = {k: v / n for k, v in final_counts.items()}

        return SelfPlayStats(
            num_games=len(results),
            total_examples=total_examples,
            avg_steps=sum(r.num_steps for r in results) / n,
            avg_t_gates_reduced=sum(r.t_gates_reduced for r in results) / n,
            simplification_rate=sum(1 for r in results if r.simplified) / n,
            avg_initial_t_gates=sum(r.initial_t_gates for r in results) / n,
            avg_final_t_gates=sum(r.final_t_gates for r in results) / n,
            avg_reward=sum(r.total_reward for r in results) / n,
            games_with_t_increase=sum(
                1 for r in results if r.final_t_gates > r.initial_t_gates
            ),
            self_play_time=sp_time,
            initial_node_type_counts=initial_avg,
            final_node_type_counts=final_avg,
            action_type_counts=action_type_counts,
        )

    def _generate_curriculum_games(
        self, num_games: int,
    ) -> list[EpisodeResult]:
        """Generate self-play games with mixed-difficulty sampling.

        Most games use the current curriculum level.  A fraction use easier
        or harder levels to smooth transitions and prevent forgetting.

        When using ParallelSelfPlayManager, difficulty overrides are passed
        as a batch so workers can apply them per-game without serial dispatch.
        When using serial SelfPlayManager, falls back to the original loop.
        """
        difficulty_levels = self.curriculum.get_mixed_difficulty_levels(num_games)

        # Parallel path: pass all difficulty overrides in one call
        if isinstance(self.self_play_manager, ParallelSelfPlayManager):
            return self.self_play_manager.generate_games(
                num_games,
                difficulty_overrides=difficulty_levels,
            )

        # Serial path: original per-game loop
        results = []
        for i, (nq, depth) in enumerate(difficulty_levels):
            saved_q = self.mcts_config.num_qubits
            saved_d = self.mcts_config.depth
            self.mcts_config.num_qubits = nq
            self.mcts_config.depth = depth

            try:
                game_results = self.self_play_manager.generate_games(1)
                results.extend(game_results)
            finally:
                self.mcts_config.num_qubits = saved_q
                self.mcts_config.depth = saved_d

        return results

    def _log_curriculum(self, advanced: bool) -> None:
        """Log curriculum state to TensorBoard."""
        if not self.tb_logger or not self.tb_logger._writer:
            return
        iteration = self.current_iteration
        c = self.curriculum
        w = self.tb_logger._writer
        w.add_scalar('curriculum/level', c.current_level, iteration)
        w.add_scalar('curriculum/num_qubits', c.current_num_qubits, iteration)
        w.add_scalar('curriculum/depth', c.current_depth, iteration)
        w.add_scalar('curriculum/at_target', int(c.at_target), iteration)
        if advanced:
            w.add_scalar('curriculum/advanced_at_iteration', iteration, iteration)

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
        if self.curriculum is not None:
            checkpoint['curriculum_state_dict'] = self.curriculum.state_dict()

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

        if 'curriculum_state_dict' in checkpoint and trainer.curriculum is not None:
            trainer.curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
            # Re-apply the restored curriculum level to MCTSConfig
            mcts_config.num_qubits = trainer.curriculum.current_num_qubits
            mcts_config.depth = trainer.curriculum.current_depth

        return trainer


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a list, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
